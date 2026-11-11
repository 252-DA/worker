"""
Static guard cho ranh giới dữ liệu (docs/ke-hoach-ranh-gioi-du-lieu.md).

Quy tắc: Python (worker, packages-ai, mcp-service) chỉ ghi bảng AI. Bảng nghiệp
vụ do core-api ghi; kết quả của worker đi về core-api qua hàng đợi
``worker_results``. Test này quét mã nguồn, tìm mọi câu SQL INSERT/UPDATE/DELETE
(và TRUNCATE/MERGE/COPY ... FROM) nằm trong chuỗi ký tự của code chạy thật, và
FAIL nếu bảng đích không thuộc miền AI. Nó chặn việc ai đó lại thêm một câu ghi
chéo vào bảng của core-api mà không ai để ý.

Cách quét: duyệt AST, lấy các hằng chuỗi và f-string (bỏ docstring), nên chú
thích và docstring nhắc tới câu SQL không bị tính. Bảng nghiệp vụ = mọi model
trong core-api/prisma/schema.prisma trừ bảng AI và outbox_events; câu văn xuôi
kiểu "insert into the queue" vì thế không bị báo nhầm (``queue`` không phải
bảng). Tên bảng động (``INSERT INTO {table}``) bị coi là vi phạm vì không kiểm
được bảng đích; nối chuỗi cũng vậy: hằng chuỗi kết thúc bằng từ khoá ghi mà
chưa có tên bảng (``"UPDATE " + table + " SET ..."``) là đích động. Không thấy
schema.prisma thì test bị bỏ qua.

Chỉ quét ba cây nguồn chạy thật: worker/worker, packages-ai/src, mcp-service/src.
Test (kể cả seed dữ liệu của test tích hợp) không bị quét.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "core-api" / "prisma" / "schema.prisma"

# Bảng AI: worker/packages-ai/mcp-service được ghi trực tiếp.
AI_TABLES = frozenset(
    {
        "chunks",
        "chunk_lo_mappings",
        "concepts",
        "chunk_concepts",
        "knowledge_relations",
        "llm_usage_logs",
    }
)

# Ngoại lệ có chủ đích: worker ghi sự kiện graph của chính nó (HEADING_GRAPH_PROJECT,
# CONCEPT_GRAPH_PROJECT) cùng transaction với chunks — transactional outbox. Việc
# giới hạn theo event_type là của RLS ở script phân quyền, không phải của test này.
OUTBOX_TABLE = "outbox_events"

WRITABLE_TABLES = AI_TABLES | {OUTBOX_TABLE}

SCAN_ROOTS = {
    "worker/worker": REPO_ROOT / "worker" / "worker",
    "packages-ai/src": REPO_ROOT / "packages-ai" / "src",
    "mcp-service/src": REPO_ROOT / "mcp-service" / "src",
}

_DOCUMENTS_REPO = "packages-ai/src/document_chunk/adapters/metadata/postgres_document_repository.py"
_PLACEMENT_REPO = "packages-ai/src/document_chunk/adapters/metadata/postgres_document_placement_repository.py"

# Ngoại lệ tường minh, theo cặp (file, bảng): các hàm legacy còn được delivery
# gRPC/HTTP cũ của packages-ai (compose profile "legacy", không chạy mặc định) dùng.
# Khi xoá delivery legacy thì xoá hàm và GỠ ngoại lệ ở đây;
# test_legacy_exceptions_are_still_needed fail nếu ngoại lệ không còn câu SQL nào khớp.
LEGACY_EXCEPTIONS: dict[tuple[str, str], str] = {
    (_DOCUMENTS_REPO, "documents"): (
        "PostgresDocumentRepository.upsert_document / update_document_status / delete: "
        "EnqueueDocument, ProcessDocument, DeleteDocument của delivery gRPC/HTTP legacy"
    ),
    (_PLACEMENT_REPO, "documents"): (
        "PostgresDocumentPlacementRepository.set_document_chapter: "
        "MetadataStoreDocumentReporter (delivery legacy, script backfill)"
    ),
}

# --- Nhận diện câu ghi -------------------------------------------------------

_IDENT = r'(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_$]*)'
# [schema.]table — nhóm 1 là tên bảng.
# ONLY/TABLE trần không phải tên bảng: ``TRUNCATE TABLE `` + t không được đọc thành bảng "table".
_TABLE = rf"(?:{_IDENT}\s*\.\s*)?((?!(?:ONLY|TABLE)\b){_IDENT})"

_WRITE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("INSERT", re.compile(rf"\bINSERT\s+INTO\s+(?:ONLY\s+)?{_TABLE}", re.I)),
    ("DELETE", re.compile(rf"\bDELETE\s+FROM\s+(?:ONLY\s+)?{_TABLE}", re.I)),
    # `DO UPDATE SET` (ON CONFLICT) không phải câu UPDATE: loại bằng (?!SET\b).
    (
        "UPDATE",
        re.compile(
            rf"\bUPDATE\s+(?:ONLY\s+)?(?!SET\b){_TABLE}(?:\s+(?:AS\s+)?[A-Za-z_]\w*)?\s+SET\b",
            re.I,
        ),
    ),
    ("TRUNCATE", re.compile(rf"\bTRUNCATE\s+(?:TABLE\s+)?(?:ONLY\s+)?{_TABLE}", re.I)),
    ("MERGE", re.compile(rf"\bMERGE\s+INTO\s+(?:ONLY\s+)?{_TABLE}", re.I)),
    ("COPY", re.compile(rf"\bCOPY\s+{_TABLE}\s*(?:\([^)]*\))?\s+FROM\b", re.I)),
)

# Sau INSERT INTO / DELETE FROM / UPDATE không có tên bảng đọc được (placeholder,
# nối chuỗi): không kiểm được bảng đích nên coi là vi phạm.
_DYNAMIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("INSERT", re.compile(r"\bINSERT\s+INTO(?:\s+ONLY)?\s*(?:$|[^\sA-Za-z_\"])", re.I)),
    ("DELETE", re.compile(r"\bDELETE\s+FROM(?:\s+ONLY)?\s*(?:$|[^\sA-Za-z_\"])", re.I)),
    ("UPDATE", re.compile(r"\bUPDATE\s*(?:\{|%s|%\()", re.I)),
)


# Nối chuỗi: ``"UPDATE " + table + " SET ..."``, ``"TRUNCATE " + t``. Từng hằng chuỗi
# không chứa tên bảng nên các mẫu trên không thấy gì; hằng kết thúc bằng từ khoá ghi
# (tên bảng nằm ở phần nối sau) là đích động. INSERT INTO / DELETE FROM đã được
# _DYNAMIC_PATTERNS bắt qua ``$``.
_TRAILING_KEYWORD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("UPDATE", re.compile(r"\bUPDATE(?:\s+ONLY)?\s+$", re.I)),
    ("TRUNCATE", re.compile(r"\bTRUNCATE(?:\s+TABLE)?(?:\s+ONLY)?\s+$", re.I)),
    ("MERGE", re.compile(r"\bMERGE\s+INTO(?:\s+ONLY)?\s+$", re.I)),
)
# Bắt buộc có khoảng trắng cuối: tên bảng phải tách khỏi từ khoá nên chỗ nối luôn có nó.
# Nhờ vậy giá trị kiểu enum ``op = "UPDATE"`` hay văn xuôi "Failed to update" không bị báo nhầm.
# Không bắt "COPY " cuối chuỗi: "copy" còn là codec của ffmpeg (video_composer.py).
# COPY ghi bảng luôn đi với FROM STDIN; thấy FROM STDIN mà không có tên bảng đứng
# trước trong cùng hằng (``"COPY " + t + " (a) FROM STDIN"``) thì là đích động.
_COPY_FROM_STDIN = re.compile(r"\bFROM\s+STDIN\b", re.I)
_COPY_WITH_TABLE = re.compile(rf"\bCOPY\s+{_TABLE}\s*(?:\([^)]*\))?\s+FROM\b", re.I)
# UPDATE cuối chuỗi vẫn có thể là mệnh đề của câu khác, không phải câu ghi:
# ``FOR [NO KEY] UPDATE``, ``ON CONFLICT DO UPDATE``, ``ON UPDATE CASCADE``, ``BEFORE/AFTER
# UPDATE`` (trigger), ``INSERT OR UPDATE``, ``GRANT SELECT, UPDATE``.
_NOT_A_STATEMENT_BEFORE = frozenset({"FOR", "DO", "ON", "KEY", "BEFORE", "AFTER", "OF", "OR"})
# UPDATE <bảng> rồi mới nối `" SET ..."`: có tên bảng nhưng thiếu SET nên mẫu UPDATE ở trên trượt.
_UPDATE_SPLIT_BEFORE_SET = re.compile(
    rf"\bUPDATE\s+(?:ONLY\s+)?(?!SET\b){_TABLE}(?:\s+(?:AS\s+)?(?!SET\b)[A-Za-z_]\w*)?\s*$", re.I
)


def _trailing_write_keyword(text: str) -> list[tuple[str, str | None, re.Match[str]]]:
    """(loại, bảng, khớp) cho câu ghi bị cắt bởi nối chuỗi ở cuối hằng ``text``."""
    found: list[tuple[str, str | None, re.Match[str]]] = []
    for kind, pattern in _TRAILING_KEYWORD_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        if kind == "UPDATE":
            before = text[: match.start()].rstrip()
            last_word = before.rsplit(None, 1)[-1] if before else ""
            if before.endswith(",") or last_word.upper() in _NOT_A_STATEMENT_BEFORE:
                continue
        found.append((kind, None, match))
    stdin = _COPY_FROM_STDIN.search(text)
    if stdin is not None and _COPY_WITH_TABLE.search(text) is None:
        found.append(("COPY", None, stdin))
    split = _UPDATE_SPLIT_BEFORE_SET.search(text)
    if split is not None:
        found.append(("UPDATE", _normalize_table(split.group(1)), split))
    return found


@dataclass(frozen=True)
class SqlWrite:
    kind: str
    table: str | None  # None: tên bảng động
    line: int
    snippet: str

    def describe(self) -> str:
        target = self.table if self.table is not None else "<tên bảng động>"
        return f"dòng {self.line}: {self.kind} {target}  ({self.snippet})"


def _normalize_table(raw: str) -> str:
    return raw.strip('"').lower()


def _string_pieces(tree: ast.AST) -> list[tuple[int, str]]:
    """(dòng, nội dung) của mọi chuỗi trong code, trừ docstring."""
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))

    pieces: list[tuple[int, str]] = []
    inside_fstring: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            text = "".join(
                part.value if isinstance(part, ast.Constant) else "{}" for part in node.values
            )
            pieces.append((node.lineno, text))
            inside_fstring.update(id(part) for part in node.values)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and id(node) not in inside_fstring
        ):
            pieces.append((node.lineno, node.value))
    return pieces


def find_sql_writes(source: str, filename: str = "<source>") -> list[SqlWrite]:
    """Mọi câu ghi SQL trong chuỗi của mã nguồn Python ``source``."""
    writes: list[SqlWrite] = []
    for lineno, text in _string_pieces(ast.parse(source, filename=filename)):
        for kind, pattern in _WRITE_PATTERNS:
            for match in pattern.finditer(text):
                writes.append(
                    SqlWrite(
                        kind=kind,
                        table=_normalize_table(match.group(1)),
                        line=lineno + text.count("\n", 0, match.start()),
                        snippet=" ".join(match.group(0).split()),
                    )
                )
        for kind, pattern in _DYNAMIC_PATTERNS:
            for match in pattern.finditer(text):
                writes.append(
                    SqlWrite(
                        kind=kind,
                        table=None,
                        line=lineno + text.count("\n", 0, match.start()),
                        snippet=" ".join(match.group(0).split()),
                    )
                )
        for kind, table, match in _trailing_write_keyword(text):
            writes.append(
                SqlWrite(
                    kind=kind,
                    table=table,
                    line=lineno + text.count("\n", 0, match.start()),
                    snippet=" ".join(match.group(0).split()),
                )
            )
    return sorted(writes, key=lambda w: (w.line, w.kind))


def scan_tree(root: Path) -> dict[str, list[SqlWrite]]:
    """Đường dẫn tương đối với ``root`` → các câu ghi tìm được trong file đó."""
    found: dict[str, list[SqlWrite]] = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        writes = find_sql_writes(path.read_text(encoding="utf-8"), str(path))
        if writes:
            found[path.relative_to(root).as_posix()] = writes
    return found


def is_allowed(rel_path: str, write: SqlWrite, business_tables: frozenset[str]) -> bool:
    """``business_tables``: bảng của core-api (đã trừ bảng AI và outbox_events)."""
    if write.table is None:
        return False
    if write.table not in business_tables:
        return True
    return (rel_path, write.table) in LEGACY_EXCEPTIONS


def violations_in(label: str, root: Path, business_tables: frozenset[str]) -> list[str]:
    out: list[str] = []
    for rel, writes in scan_tree(root).items():
        full = f"{label}/{rel}"
        for write in writes:
            if not is_allowed(full, write, business_tables):
                out.append(f"{full} {write.describe()}")
    return out


def business_tables_from_schema() -> frozenset[str] | None:
    """Bảng nghiệp vụ theo schema.prisma; None khi không thấy schema."""
    if not SCHEMA_PATH.is_file():
        return None
    tables = prisma_tables(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert len(tables) > 30, f"đọc schema.prisma thất bại, chỉ thấy {len(tables)} bảng"
    return frozenset(tables - WRITABLE_TABLES)


def prisma_tables(schema_text: str) -> set[str]:
    """Tên bảng của mọi model trong schema.prisma (``@@map`` nếu có, không thì tên model)."""
    tables: set[str] = set()
    for match in re.finditer(r"^model\s+(\w+)\s*\{(.*?)^\}", schema_text, re.M | re.S):
        mapped = re.search(r'@@map\(\s*"([^"]+)"\s*\)', match.group(2))
        tables.add((mapped.group(1) if mapped else match.group(1)).lower())
    return tables


# --- Test trên mã nguồn thật -------------------------------------------------


@pytest.mark.parametrize("label", sorted(SCAN_ROOTS))
def test_python_writes_only_ai_tables(label):
    root = SCAN_ROOTS[label]
    if not root.is_dir():
        pytest.skip(f"không có {label} trong môi trường này")
    business = business_tables_from_schema()
    if business is None:
        pytest.skip("không thấy core-api/prisma/schema.prisma")

    violations = violations_in(label, root, business)

    assert violations == [], (
        "Python chỉ được ghi bảng AI "
        f"({', '.join(sorted(AI_TABLES))}) và outbox_events. Bảng nghiệp vụ do core-api ghi: "
        "gửi kết quả qua hàng đợi worker_results thay vì viết SQL.\n  "
        + "\n  ".join(violations)
    )


def test_legacy_exceptions_are_still_needed():
    """Xoá delivery legacy rồi thì phải gỡ ngoại lệ, không để sót lỗ hổng."""
    stale: list[str] = []
    for (rel, table), reason in LEGACY_EXCEPTIONS.items():
        root_key = next((k for k in SCAN_ROOTS if rel.startswith(k + "/")), None)
        assert root_key is not None, f"ngoại lệ ngoài các cây được quét: {rel}"
        root = SCAN_ROOTS[root_key]
        if not root.is_dir():
            pytest.skip(f"không có {root_key} trong môi trường này")
        path = root / rel[len(root_key) + 1 :]
        writes = find_sql_writes(path.read_text(encoding="utf-8"), str(path)) if path.is_file() else []
        if not any(w.table == table for w in writes):
            stale.append(f"{rel} ({table}): {reason}")
    assert stale == [], "Ngoại lệ legacy không còn câu SQL nào khớp, hãy gỡ:\n  " + "\n  ".join(stale)


def test_ai_tables_exist_in_prisma_schema():
    """Danh sách bảng AI không được lệch khỏi schema (đổi tên bảng thì guard phải biết)."""
    if not SCHEMA_PATH.is_file():
        pytest.skip("không thấy core-api/prisma/schema.prisma")

    tables = prisma_tables(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert len(tables) > 30, f"đọc schema.prisma thất bại, chỉ thấy {len(tables)} bảng"
    assert WRITABLE_TABLES <= tables, f"bảng không có trong schema: {sorted(WRITABLE_TABLES - tables)}"
    assert {"documents", "quiz_attempts", "quiz_items", "courses"} <= tables


def test_legacy_exceptions_target_business_tables_from_the_schema():
    if not SCHEMA_PATH.is_file():
        pytest.skip("không thấy core-api/prisma/schema.prisma")
    tables = prisma_tables(SCHEMA_PATH.read_text(encoding="utf-8"))
    business = tables - WRITABLE_TABLES

    for (_rel, table), _reason in LEGACY_EXCEPTIONS.items():
        assert table in business, f"{table} không phải bảng nghiệp vụ trong schema.prisma"


# --- Bộ quét tự kiểm tra -----------------------------------------------------


_BUSINESS = frozenset(
    {"documents", "quiz_attempts", "quiz_items", "courses", "lessons", "lesson_cards"}
)


class TestScanner:
    @staticmethod
    def _writes(sql: str) -> list[tuple[str, str | None]]:
        return [(w.kind, w.table) for w in find_sql_writes(f"Q = {sql!r}\n")]

    def test_flags_each_statement_kind_on_business_tables(self):
        assert self._writes("INSERT INTO documents (a) VALUES (1)") == [("INSERT", "documents")]
        assert self._writes("UPDATE quiz_attempts SET score = 1") == [("UPDATE", "quiz_attempts")]
        assert self._writes("DELETE FROM quiz_items WHERE x") == [("DELETE", "quiz_items")]
        assert self._writes("TRUNCATE TABLE courses") == [("TRUNCATE", "courses")]
        assert self._writes("MERGE INTO lessons USING x") == [("MERGE", "lessons")]
        assert self._writes("COPY lessons (a, b) FROM STDIN") == [("COPY", "lessons")]

    def test_handles_case_schema_quotes_alias_and_newlines(self):
        assert self._writes("insert\n  into   public.\"Documents\" (a)") == [("INSERT", "documents")]
        assert self._writes("UPDATE chunks c SET deleted_at = now()") == [("UPDATE", "chunks")]
        assert self._writes("UPDATE ONLY quiz_attempts AS qa\n SET score = 1") == [
            ("UPDATE", "quiz_attempts")
        ]

    def test_on_conflict_do_update_is_not_an_update_statement(self):
        sql = "INSERT INTO chunks (a) VALUES (1) ON CONFLICT (a) DO UPDATE SET a = 2"
        assert self._writes(sql) == [("INSERT", "chunks")]

    def test_select_for_update_and_reads_are_ignored(self):
        assert self._writes("SELECT * FROM documents WHERE x FOR UPDATE") == []
        assert self._writes("SELECT count(*) FROM quiz_attempts") == []

    def test_cte_write_inside_with_is_found(self):
        sql = "WITH r AS (INSERT INTO lessons (a) VALUES (1) RETURNING a) SELECT * FROM r"
        assert self._writes(sql) == [("INSERT", "lessons")]

    def test_dynamic_table_name_is_flagged(self):
        [write] = find_sql_writes('def f(t):\n    return f"DELETE FROM {t} WHERE id = 1"\n')
        assert (write.kind, write.table) == ("DELETE", None)
        [write] = find_sql_writes('Q = "INSERT INTO " + table\n')
        assert write.table is None
        assert not is_allowed("x.py", write, _BUSINESS)

    @pytest.mark.parametrize(
        "source",
        [
            'Q = "UPDATE " + table + " SET score = 1"\n',
            'Q = "UPDATE ONLY " + table + " SET score = 1"\n',
            'Q = ("UPDATE "\n     + table\n     + " SET score = 1")\n',
            'Q = "INSERT INTO " + table + " (a) VALUES (1)"\n',
            'Q = "DELETE FROM " + table + " WHERE id = 1"\n',
            'Q = "TRUNCATE " + table\n',
            'Q = "TRUNCATE TABLE " + table\n',
            'Q = "MERGE INTO " + table + " USING x"\n',
            'Q = "COPY " + table + " (a, b) FROM STDIN"\n',
            'Q = "x; UPDATE " + table\n',
        ],
    )
    def test_string_concatenated_table_name_is_a_dynamic_target(self, source):
        writes = find_sql_writes(source)

        assert writes, source
        assert all(w.table is None for w in writes), [w.describe() for w in writes]
        assert not any(is_allowed("x.py", w, _BUSINESS) for w in writes)

    def test_update_table_followed_by_concatenated_set_names_the_table(self):
        [write] = find_sql_writes('Q = "UPDATE documents" + " SET status = 1"\n')
        assert (write.kind, write.table) == ("UPDATE", "documents")
        assert not is_allowed("x.py", write, _BUSINESS)
        # Bảng AI vẫn được phép dù bị cắt bởi nối chuỗi.
        [write] = find_sql_writes('Q = "UPDATE chunks c" + " SET deleted_at = now()"\n')
        assert (write.kind, write.table) == ("UPDATE", "chunks")
        assert is_allowed("x.py", write, _BUSINESS - WRITABLE_TABLES)

    @pytest.mark.parametrize(
        "text",
        [
            "SELECT * FROM documents WHERE id = 1 FOR UPDATE",
            "SELECT * FROM documents FOR NO KEY UPDATE",
            "SELECT 1 ON CONFLICT (a) DO UPDATE ",
            "ALTER TABLE x ADD FOREIGN KEY (a) REFERENCES y ON UPDATE",
            "CREATE TRIGGER t BEFORE UPDATE",
            "CREATE TRIGGER t AFTER INSERT OR UPDATE",
            "GRANT SELECT, UPDATE",
            # "copy" là codec ffmpeg, "update" văn xuôi: không phải câu ghi.
            "copy",
            "-c:a copy",
            "Please update",
            "Failed to update",
            # Giá trị kiểu enum, không có khoảng trắng cuối nên không thể đứng trước tên bảng.
            "UPDATE",
            "TRUNCATE",
        ],
    )
    def test_clause_and_prose_endings_are_not_mistaken_for_writes(self, text):
        assert find_sql_writes(f"Q = {text!r}\n") == []

    def test_adjacent_literals_without_a_plus_are_one_constant(self):
        # Python gộp ngay lúc parse: `"UPDATE " "documents SET a = 1"` là một chuỗi hoàn chỉnh.
        assert [(w.kind, w.table) for w in find_sql_writes('Q = "UPDATE " "documents SET a = 1"\n')] == [
            ("UPDATE", "documents")
        ]

    def test_docstrings_and_comments_are_not_sql(self):
        source = (
            '"""Module doc: INSERT INTO documents mô tả."""\n'
            "# UPDATE courses SET name = 1\n"
            "def f():\n"
            '    """DELETE FROM quiz_items trong docstring."""\n'
            "    return 1\n"
        )
        assert find_sql_writes(source) == []

    def test_prose_is_not_mistaken_for_sql(self):
        assert self._writes("Please update the settings; truncated output") == []
        # Văn xuôi có thể khớp mẫu nhưng "queue" không phải bảng nghiệp vụ nên không bị báo.
        [write] = find_sql_writes("Q = 'cannot insert into queue: closed'\n")
        assert is_allowed("x.py", write, _BUSINESS)

    def test_reports_the_line_of_the_statement(self):
        source = 'X = 1\nQ = """\nSELECT 1;\nDELETE FROM documents;\n"""\n'
        [write] = find_sql_writes(source)
        assert write.line == 4

    def test_allowlist_covers_ai_tables_outbox_and_explicit_legacy_pairs_only(self):
        def w(table):
            return SqlWrite("INSERT", table, 1, "")

        # Tập bảng nghiệp vụ thật luôn đã trừ bảng AI và outbox_events.
        business = _BUSINESS - WRITABLE_TABLES
        for table in sorted(AI_TABLES) + ["outbox_events"]:
            assert is_allowed("any.py", w(table), business)
        assert not is_allowed("any.py", w("documents"), business)
        assert is_allowed(_DOCUMENTS_REPO, w("documents"), business)
        assert not is_allowed(_DOCUMENTS_REPO, w("quiz_attempts"), business)
        assert not is_allowed("other/file.py", w("documents"), business)

    def test_prisma_parser_uses_map_when_present(self):
        schema = (
            "model documents {\n  id Int @id\n}\n"
            'model Foo {\n  id Int @id\n  @@map("foo_tbl")\n}\n'
            "enum X {\n  A\n}\n"
        )
        assert prisma_tables(schema) == {"documents", "foo_tbl"}

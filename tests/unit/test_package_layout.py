"""
Khoá cách xếp gói ``worker`` (xem docstring ``__init__.py`` của từng gói).

Phụ thuộc chỉ đi một chiều:

    runners → container → use_cases → reporters, services → adapters
                          use_cases → use_cases/shared

- ``adapters``: không import gì khác trong ``worker``.
- ``services``: không import ``use_cases``, ``runners``, ``container``.
- ``reporters``: không import ``runners``, ``container``, ``services``; kiểu của
  use case chỉ được lấy trong ``if TYPE_CHECKING``.
- ``use_cases``: không import ``runners``/``container``; một use case không
  import use case khác — phần dùng chung nằm ở ``use_cases/shared``, và
  ``shared`` không import use case nào.
- Không ai ngoài ``runners`` import ``runners``.
- Chỉ ``reporters`` (và chính adapter) được dùng ``make_result_id`` hay hằng
  ``kind`` của ``worker.adapters.result_publisher``: mọi kết quả về core-api
  đi qua reporter.

Quét cả import nằm trong hàm (import lười), trừ khối ``if TYPE_CHECKING``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "worker"
RESULT_PUBLISHER = "worker.adapters.result_publisher"
# Ngoài reporters/adapters chỉ được lấy các tên này từ result_publisher.
RESULT_PUBLISHER_PUBLIC = {"InvalidWorkerResult", "WorkerResultPublisher"}


def _module_name(path: Path) -> str:
    rel = path.relative_to(PACKAGE_ROOT.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _is_type_checking(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _imports(tree: ast.AST) -> list[tuple[str, tuple[str, ...]]]:
    """(module, tên được import) cho mọi import ``worker.*``, bỏ khối TYPE_CHECKING."""
    found: list[tuple[str, tuple[str, ...]]] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.If) and _is_type_checking(node):
            for child in node.orelse:
                visit(child)
            return
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "worker" or alias.name.startswith("worker."):
                    found.append((alias.name, ()))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "worker" or node.module.startswith("worker."):
                names = tuple(alias.name for alias in node.names)
                if node.module == "worker":
                    # ``from worker import x`` = import gói con ``worker.x``.
                    found.extend((f"worker.{name}", ()) for name in names)
                else:
                    found.append((node.module, names))
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return found


def _modules() -> list[tuple[str, list[tuple[str, tuple[str, ...]]]]]:
    return [
        (_module_name(path), _imports(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))))
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        if "__pycache__" not in path.parts
    ]


def _under(module: str, package: str) -> bool:
    return module == package or module.startswith(package + ".")


def _use_case_of(module: str) -> str | None:
    """Tên use case (module trực tiếp dưới ``use_cases``), None nếu là ``shared``/gói."""
    parts = module.split(".")
    if len(parts) >= 3 and parts[1] == "use_cases" and parts[2] != "shared":
        return parts[2]
    return None


def _violations() -> list[str]:
    problems: list[str] = []
    for module, imports in _modules():
        for target, names in imports:
            def bad(reason: str) -> None:
                problems.append(f"{module} imports {target}: {reason}")

            if _under(module, "worker.adapters") and not _under(target, "worker.adapters"):
                bad("adapters không import phần khác của worker")
            if _under(module, "worker.services") and any(
                _under(target, p) for p in ("worker.use_cases", "worker.runners", "worker.container")
            ):
                bad("services không import use_cases/runners/container")
            if _under(module, "worker.reporters") and any(
                _under(target, p)
                for p in ("worker.runners", "worker.container", "worker.services", "worker.use_cases")
            ):
                bad("reporters chỉ được lấy kiểu use case trong if TYPE_CHECKING")
            if _under(module, "worker.use_cases") and any(
                _under(target, p) for p in ("worker.runners", "worker.container")
            ):
                bad("use_cases không import runners/container")
            if _under(module, "worker.use_cases.shared") and _use_case_of(target):
                bad("use_cases/shared không import use case")
            source_case, target_case = _use_case_of(module), _use_case_of(target)
            if source_case and target_case and source_case != target_case:
                bad("use case không import use case khác; chuyển phần chung vào use_cases/shared")
            if _under(target, "worker.runners") and not _under(module, "worker.runners"):
                bad("chỉ runners import runners")
            if (
                target == RESULT_PUBLISHER
                and not (_under(module, "worker.reporters") or _under(module, "worker.adapters"))
                and (not names or set(names) - RESULT_PUBLISHER_PUBLIC)
            ):
                bad("kind/result_id chỉ dùng trong reporters; gửi kết quả qua reporter")
    return problems


def test_package_dependencies_go_one_way():
    assert _violations() == []


@pytest.mark.parametrize(
    "module, source",
    [
        ("worker.services.x", "from worker.use_cases.run_pipeline import RunPipelineUseCase"),
        ("worker.use_cases.generate_micro_quiz", "from worker.use_cases.generate_curriculum_quiz import X"),
        ("worker.use_cases.run_enrichment", "from worker.adapters.result_publisher import make_result_id"),
        ("worker.reporters.grading", "from worker.use_cases.grade_short_answers import X"),
        ("worker.adapters.x", "import worker.config"),
        ("worker.use_cases.shared.x", "from worker.use_cases.run_pipeline import X"),
    ],
)
def test_checker_catches_a_forbidden_import(monkeypatch, module, source):
    """Bộ kiểm phải thật sự bắt được từng luật, không chỉ trả rỗng."""
    monkeypatch.setattr(
        "tests.unit.test_package_layout._modules",
        lambda: [(module, _imports(ast.parse(source)))],
    )
    assert _violations()


def test_type_checking_imports_are_allowed_in_reporters(monkeypatch):
    source = "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from worker.use_cases.x import Y\n"
    monkeypatch.setattr(
        "tests.unit.test_package_layout._modules",
        lambda: [("worker.reporters.x", _imports(ast.parse(source)))],
    )
    assert _violations() == []

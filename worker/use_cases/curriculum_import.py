"""
Nạp đề cương — phần của worker: parse và map chunk → LO. Worker không ghi bảng
đề cương nào (docs/ke-hoach-ranh-gioi-du-lieu.md):

- ``curriculum_preview``: tải file core-api đã lưu, parse + trích xuất, gửi
  CURRICULUM_IMPORT_RESULT (bản xem trước + bản đề cương chuẩn hoá đầy đủ, hoặc
  lỗi) về core-api qua ``worker_results``. core-api lưu lại cho giảng viên duyệt
  và khi áp dụng thì ghi đúng bản đó — bản được duyệt chính là bản được ghi.
- ``COURSE_CHUNK_LO_REMAP_REQUESTED`` (outbox, sau khi core-api ghi đề cương):
  map lại các chunk đã index của học phần sang LO mới; ``chunk_lo_mappings`` là
  dữ liệu AI nên worker vẫn tự ghi.
- ``curriculum_apply``: lệnh của luồng cũ (worker tự ghi đề cương). Job còn sót
  trong hàng đợi chỉ được báo lỗi về để lần nạp không treo ở APPLYING.
"""
from __future__ import annotations

import tempfile
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from document_chunk.application.use_cases.ingest_curriculum import IngestCurriculumRequest
from document_chunk.application.use_cases.map_chunks_to_los import MapChunksToLosRequest
from document_chunk.domain.bloom import bloom_to_level
from document_chunk.domain.entities.curriculum import Curriculum, SourceRef
from document_chunk.domain.exceptions import DocumentStaleError
from document_chunk.shared.logger import get_logger

from worker.services.result_publisher import (
    CURRICULUM_IMPORT_RESULT,
    InvalidWorkerResult,
    make_result_id,
)

logger = get_logger(__name__)

CURRICULUM_PREVIEW_JOB = "curriculum_preview"
LEGACY_CURRICULUM_APPLY_JOB = "curriculum_apply"
COURSE_CHUNK_LO_REMAP_REQUESTED = "COURSE_CHUNK_LO_REMAP_REQUESTED"
CURRICULUM_JOB_NAMES = frozenset(
    {CURRICULUM_PREVIEW_JOB, LEGACY_CURRICULUM_APPLY_JOB, COURSE_CHUNK_LO_REMAP_REQUESTED}
)

_MAX_ERROR_CHARS = 2000
# Job xếp trước khi đổi luồng chỉ mang import_id; worker không còn đọc
# curriculum_imports để lấy phần còn lại.
_LEGACY_PREVIEW_ERROR = (
    "Yêu cầu trích xuất này được tạo trước khi đổi cách nạp đề cương. "
    "Hãy đồng bộ lại đề cương từ Canvas."
)
_LEGACY_APPLY_ERROR = (
    "Lệnh áp dụng theo cách cũ không còn chạy (core-api nay tự ghi đề cương). "
    "Hãy đồng bộ lại đề cương từ Canvas rồi áp dụng bản xem trước mới."
)


class InvalidCurriculumJob(ValueError):
    """Dữ liệu job sai dạng: chạy lại cũng sai y như vậy."""


def _source(ref: SourceRef | None) -> dict[str, Any] | None:
    if ref is None:
        return None
    return {"section": ref.section, "page": ref.page}


def _source_columns(ref: SourceRef | None) -> dict[str, Any]:
    return {
        "source_section": ref.section if ref else None,
        "source_page": ref.page if ref else None,
    }


def _cdio_level(value: str | int | None) -> str | None:
    """Như ``_cdio_level`` của repository cũ: chỉ I/II/III, không biết thì None."""
    if str(value) in {"I", "II", "III"}:
        return str(value)
    return {1: "I", 2: "II", 3: "III"}.get(value)


def curriculum_preview(curriculum: Curriculum) -> dict[str, Any]:
    """Dạng JSON cho màn hình duyệt: đủ để giảng viên đối chiếu với đề cương gốc."""
    course = curriculum.course
    los_by_chapter: dict[str, list[str]] = {}
    for link in curriculum.chapter_lo_links:
        codes = los_by_chapter.setdefault(link.chapter_code, [])
        if link.lo_code not in codes:
            codes.append(link.lo_code)

    return {
        "course": {
            "code": course.code,
            "title_vi": course.title_vi,
            "title_en": course.title_en,
            "credits": course.credits,
            "semester": course.semester,
            "syllabus_version": course.syllabus_version,
        },
        "counts": {
            "chapters": len(curriculum.chapters),
            "learning_outcomes": len(curriculum.learning_outcomes),
            "assessments": len(curriculum.assessments),
            "sessions": len(curriculum.sessions),
            "goals": len(curriculum.goals),
            "chapter_lo_links": len(curriculum.chapter_lo_links),
            "lo_assessment_links": len(curriculum.lo_assessment_links),
        },
        "chapters": [
            {
                "code": ch.code,
                "title": ch.title,
                "title_en": ch.title_en,
                "lo_codes": los_by_chapter.get(ch.code, []),
            }
            for ch in sorted(curriculum.chapters, key=lambda c: c.order_index)
        ],
        "learning_outcomes": [
            {
                "code": lo.code,
                "parent_code": lo.parent_code,
                "statement_vi": lo.statement_vi,
                "statement_en": lo.statement_en,
                "bloom_level": lo.bloom_level,
                "bloom_provenance": lo.bloom_provenance,
                "cdio_level": lo.cdio_level,
            }
            for lo in curriculum.learning_outcomes
        ],
        "assessments": [
            {
                "code": a.code,
                "name_vi": a.name_vi,
                "parent_code": a.parent_code,
                "category": a.category,
                "weight": a.weight,
            }
            for a in curriculum.assessments
        ],
        "sessions": [
            {
                "order_index": s.order_index,
                "session_no": s.session_no,
                "chapter_code": s.chapter_code,
                "title_vi": s.title_vi,
            }
            for s in curriculum.sessions
        ],
        "issues": [
            {
                "code": i.code,
                "severity": i.severity,
                "message": i.message,
                "source": _source(i.source),
            }
            for i in curriculum.issues
        ],
        "blocking": bool(curriculum.blocking_issues),
    }


def curriculum_payload(curriculum: Curriculum) -> dict[str, Any]:
    """
    Bản đề cương chuẩn hoá core-api ghi khi áp dụng (contracts/integration-events/
    v1/curriculum-import-result.json, definitions.curriculum).

    Giữ mã thay cho id: core-api tự sinh id dòng tất định từ course_id của lần
    nạp và mã, như ``stable_uuid`` cũ. Giá trị đã quy về dạng cột (Bloom 1–6,
    CDIO I/II/III) ở đây vì bảng quy đổi nằm ở ``domain.bloom``; ánh xạ thuần
    về DB (loại đánh giá → assessments.type, scope/session_order) do core-api làm.
    Giữ nguyên thứ tự phần tử: sort_order của assessments là vị trí trong danh sách.
    """
    course = curriculum.course
    return {
        "course": {
            "code": course.code,
            "title_vi": course.title_vi,
            "title_en": course.title_en,
            "credits": course.credits,
            "semester": course.semester,
            "syllabus_version": course.syllabus_version,
            "lo_year": course.lo_year,
        },
        "source_document_id": curriculum.source_document_id,
        "chapters": [
            {
                "code": ch.code,
                "title": ch.title,
                "title_en": ch.title_en,
                "order_index": ch.order_index,
                "topics": list(ch.topics),
                **_source_columns(ch.source),
            }
            for ch in curriculum.chapters
        ],
        "learning_outcomes": [
            {
                "code": lo.code,
                "parent_code": lo.parent_code,
                "statement_vi": lo.statement_vi,
                "statement_en": lo.statement_en,
                "bloom_level": bloom_to_level(lo.bloom_level),
                "bloom_provenance": lo.bloom_provenance,
                "cdio_level": _cdio_level(lo.cdio_level),
                "cdio_provenance": lo.cdio_provenance,
                **_source_columns(lo.source),
            }
            for lo in curriculum.learning_outcomes
        ],
        "chapter_lo_links": [
            {
                "chapter_code": link.chapter_code,
                "lo_code": link.lo_code,
                "provenance": link.provenance,
                **_source_columns(link.source),
            }
            for link in curriculum.chapter_lo_links
        ],
        "goals": [
            {
                "code": g.code,
                "statement_vi": g.statement_vi,
                "statement_en": g.statement_en,
                **_source_columns(g.source),
            }
            for g in curriculum.goals
        ],
        "sessions": [
            {
                "order_index": row.order_index,
                "session_no": row.session_no,
                "chapter_code": row.chapter_code,
                "title_vi": row.title_vi,
                "title_en": row.title_en,
                **_source_columns(row.source),
            }
            for row in curriculum.sessions
        ],
        "assessments": [
            {
                "code": a.code,
                "parent_code": a.parent_code,
                "name_vi": a.name_vi,
                "name_en": a.name_en,
                "activity_type": a.activity_type,
                "category": a.category,
                "weight": a.weight,
                "weight_provenance": a.weight_provenance,
                **_source_columns(a.source),
            }
            for a in curriculum.assessments
        ],
        "lo_assessment_links": [
            {
                "lo_code": link.lo_code,
                "assessment_code": link.assessment_code,
                "row_order_index": link.row_order_index,
                "chapter_code": link.chapter_code,
                "provenance": link.provenance,
                **_source_columns(link.source),
            }
            for link in curriculum.lo_assessment_links
        ],
        "issues": [
            {
                "code": i.code,
                "severity": i.severity,
                "message": i.message,
                **_source_columns(i.source),
                "source_locator": i.source.locator if i.source else None,
            }
            for i in curriculum.issues
        ],
    }


def _uuid_field(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise InvalidCurriculumJob(f"job requires {key}")
    try:
        return str(uuid.UUID(value))
    except ValueError as exc:
        raise InvalidCurriculumJob(f"{key} must be a UUID, got {value!r}") from exc


@dataclass(frozen=True)
class CurriculumPreviewJob:
    """Dữ liệu job ``curriculum_preview`` do core-api xếp (curriculum-import.service.ts)."""

    import_id: str
    course_id: str
    file_name: str
    storage_key: str


class CurriculumImportUseCase:
    def __init__(
        self,
        *,
        publisher,
        file_storage,
        ingest_curriculum_use_case,
        map_chunks_to_los_use_case,
        course_documents,
    ) -> None:
        self._publisher = publisher
        self._file_storage = file_storage
        self._ingest = ingest_curriculum_use_case
        self._map_chunks_to_los = map_chunks_to_los_use_case
        self._course_documents = course_documents

    def run(self, job_name: str, data: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(data, Mapping):
            raise InvalidCurriculumJob("job data must be an object")
        if job_name == CURRICULUM_PREVIEW_JOB:
            return self.preview(data)
        if job_name == COURSE_CHUNK_LO_REMAP_REQUESTED:
            return self.remap_course(data)
        if job_name == LEGACY_CURRICULUM_APPLY_JOB:
            return self.reject_legacy_apply(data)
        raise InvalidCurriculumJob(f"unknown curriculum job {job_name!r}")

    # ------------------------------------------------------------------
    # Xem trước
    # ------------------------------------------------------------------

    def preview(self, data: Mapping[str, Any]) -> dict[str, Any]:
        import_id = _uuid_field(data, "import_id")
        missing = [
            key for key in ("course_id", "file_name", "storage_key")
            if not isinstance(data.get(key), str) or not data.get(key)
        ]
        if missing:
            logger.warning("curriculum_import.preview.legacy_job", import_id=import_id, missing=missing)
            return self._publish_failure(import_id, _LEGACY_PREVIEW_ERROR)
        job = CurriculumPreviewJob(
            import_id=import_id,
            course_id=_uuid_field(data, "course_id"),
            file_name=data["file_name"],
            storage_key=data["storage_key"],
        )

        try:
            with self._downloaded(job) as path:
                result = self._ingest.extract(
                    IngestCurriculumRequest(
                        file_path=path, file_name=job.file_name, course_id=job.course_id,
                    )
                )
            if result.is_err():
                raise result.error
            curriculum: Curriculum = result.unwrap()
            preview = curriculum_preview(curriculum)
            payload = {
                "import_id": import_id,
                "status": "BLOCKED" if preview["blocking"] else "READY",
                "preview": preview,
                "curriculum": curriculum_payload(curriculum),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 -- lỗi parse là kết quả FAILED, không phải job lỗi
            logger.exception(
                "curriculum_import.preview.failed",
                import_id=import_id,
                course_id=job.course_id,
                error=str(exc),
            )
            return self._publish_failure(import_id, f"{type(exc).__name__}: {exc}")

        # Lỗi gửi (Redis) để nổi lên: job gốc fail và BullMQ chạy lại; parse lại
        # cho cùng kết quả và cùng result_id nên không sinh bản trùng.
        try:
            self._publisher.publish_sync(
                CURRICULUM_IMPORT_RESULT,
                payload,
                result_id=make_result_id(CURRICULUM_IMPORT_RESULT, import_id),
            )
        except InvalidWorkerResult as exc:
            # Đề cương không đóng gói được thành JSON (ví dụ trọng số NaN): chạy
            # lại cũng vậy, nên báo FAILED thay vì để lần nạp treo ở QUEUED.
            logger.exception("curriculum_import.preview.unpublishable", import_id=import_id)
            return self._publish_failure(import_id, f"{type(exc).__name__}: {exc}")
        logger.info(
            "curriculum_import.preview.done",
            import_id=import_id,
            course_id=job.course_id,
            status=payload["status"],
            **preview["counts"],
            issues=len(preview["issues"]),
        )
        return {"import_id": import_id, "status": payload["status"], **preview["counts"]}

    def reject_legacy_apply(self, data: Mapping[str, Any]) -> dict[str, Any]:
        import_id = _uuid_field(data, "import_id")
        logger.warning("curriculum_import.apply.legacy_job", import_id=import_id)
        return self._publish_failure(import_id, _LEGACY_APPLY_ERROR, attempt="legacy-apply")

    def _publish_failure(
        self, import_id: str, error: str, *, attempt: str | None = None
    ) -> dict[str, Any]:
        self._publisher.publish_sync(
            CURRICULUM_IMPORT_RESULT,
            {
                "import_id": import_id,
                "status": "FAILED",
                "preview": None,
                "curriculum": None,
                "error": error[:_MAX_ERROR_CHARS],
            },
            result_id=make_result_id(CURRICULUM_IMPORT_RESULT, import_id, attempt),
        )
        return {"import_id": import_id, "status": "FAILED"}

    # ------------------------------------------------------------------
    # Map lại chunk → LO sau khi core-api ghi đề cương
    # ------------------------------------------------------------------

    def remap_course(self, data: Mapping[str, Any]) -> dict[str, Any]:
        payload = data.get("payload")
        if not isinstance(payload, Mapping):
            raise InvalidCurriculumJob(f"{COURSE_CHUNK_LO_REMAP_REQUESTED} requires a payload object")
        course_id = _uuid_field(payload, "course_id")
        documents, mapped = self._remap_course_chunks(course_id)
        logger.info(
            "curriculum_import.remap.done",
            course_id=course_id,
            import_id=payload.get("import_id"),
            event_id=data.get("event_id"),
            documents=documents,
            chunk_lo_mappings=mapped,
        )
        return {"course_id": course_id, "documents": documents, "chunk_lo_mappings": mapped}

    def _remap_course_chunks(self, course_id: str) -> tuple[int, int]:
        """Tài liệu index trước khi có đề cương chưa có cạnh chunk → LO; map lại từng cái.

        Lỗi map của một tài liệu (DB, Qdrant, mapper) nổi lên để BullMQ chạy lại
        lệnh: nếu chỉ log thì lần nạp đề cương đó xong mà tài liệu mãi không có
        cạnh chunk → LO, không ai biết. Map idempotent (xoá cạnh suy luận rồi ghi
        lại) nên chạy lại cả lệnh an toàn. Vẫn map hết các tài liệu còn lại trước
        khi ném lỗi, để một tài liệu hỏng không giữ các tài liệu khác chờ. Tài
        liệu đã bị xoá/không còn chunk giữa chừng thì bỏ qua: chạy lại cũng vậy.
        Không đọc được danh sách tài liệu cũng ném ra.
        """
        document_ids = self._course_documents.indexed_document_ids(course_id)
        total = 0
        first_error: Exception | None = None
        failed = 0
        for document_id in document_ids:
            result = self._map_chunks_to_los.execute(
                MapChunksToLosRequest(document_id=document_id, course_id=course_id)
            )
            if result.is_err():
                if self._document_gone(course_id, document_id, result.error):
                    logger.info(
                        "curriculum_import.remap_skipped_deleted_document",
                        course_id=course_id,
                        document_id=document_id,
                        error=str(result.error),
                    )
                    continue
                logger.warning(
                    "curriculum_import.remap_failed",
                    course_id=course_id,
                    document_id=document_id,
                    error=str(result.error),
                )
                failed += 1
                first_error = first_error or result.error
                continue
            total += result.unwrap().mapping_count
        if first_error is not None:
            logger.error(
                "curriculum_import.remap_incomplete",
                course_id=course_id,
                failed_documents=failed,
                documents=len(document_ids),
            )
            raise first_error
        return len(document_ids), total

    def _document_gone(self, course_id: str, document_id: str, error: Exception) -> bool:
        """Lỗi map là do tài liệu vừa bị xoá (hoặc không còn chunk) chứ không phải hạ tầng?"""
        if isinstance(error, DocumentStaleError):
            return True
        # Đọc lại danh sách: lỗi hạ tầng thì ném ra cũng không sao (job chạy lại).
        return document_id not in self._course_documents.indexed_document_ids(course_id)

    @contextmanager
    def _downloaded(self, job: CurriculumPreviewJob) -> Iterator[Path]:
        suffix = Path(job.file_name).suffix or Path(job.storage_key).suffix or ".pdf"
        with tempfile.TemporaryDirectory(prefix="curriculum-") as tmp_dir:
            path = Path(tmp_dir) / f"syllabus{suffix}"
            result = self._file_storage.download(job.storage_key, path)
            if result.is_err():
                raise result.error
            yield path

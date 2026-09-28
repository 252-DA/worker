"""
Nạp đề cương theo hai bước, trên cùng một file core-api đã lưu vào object storage:

- xem trước: parse + trích xuất, ghi kết quả vào `curriculum_imports.preview`;
- áp dụng: sau khi giảng viên duyệt, ghi vào đề cương đang dùng rồi map lại
  các chunk đã index của học phần sang LO mới.

Hai bước đi qua cùng `IngestCurriculumUseCase`, nên bản xem trước đúng là thứ
sẽ được ghi.
"""
from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from document_chunk.application.use_cases.ingest_curriculum import IngestCurriculumRequest
from document_chunk.application.use_cases.map_chunks_to_los import MapChunksToLosRequest
from document_chunk.domain.entities.curriculum import Curriculum
from document_chunk.shared.logger import get_logger

from worker.services.curriculum_import_store import CurriculumImport, CurriculumImportStore

logger = get_logger(__name__)

CURRICULUM_PREVIEW_JOB = "curriculum_preview"
CURRICULUM_APPLY_JOB = "curriculum_apply"
CURRICULUM_JOB_NAMES = frozenset({CURRICULUM_PREVIEW_JOB, CURRICULUM_APPLY_JOB})

_MAX_ERROR_CHARS = 2000


def _source(ref) -> dict[str, Any] | None:
    if ref is None:
        return None
    return {"section": ref.section, "page": ref.page}


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


class CurriculumImportUseCase:
    def __init__(
        self,
        *,
        store: CurriculumImportStore,
        file_storage,
        ingest_curriculum_use_case,
        map_chunks_to_los_use_case,
    ) -> None:
        self._store = store
        self._file_storage = file_storage
        self._ingest = ingest_curriculum_use_case
        self._map_chunks_to_los = map_chunks_to_los_use_case

    def run(self, job_name: str, import_id: str) -> dict[str, Any]:
        if job_name == CURRICULUM_PREVIEW_JOB:
            return self.preview(import_id)
        if job_name == CURRICULUM_APPLY_JOB:
            return self.apply(import_id)
        raise ValueError(f"unknown curriculum job {job_name!r}")

    def preview(self, import_id: str) -> dict[str, Any]:
        record = self._store.get(import_id)
        if record is None:
            logger.warning("curriculum_import.preview.not_found", import_id=import_id)
            return {"import_id": import_id, "skipped": "not_found"}
        if not self._store.transition(import_id, to="EXTRACTING", expect=("QUEUED",)):
            logger.info("curriculum_import.preview.skipped", import_id=import_id, status=record.status)
            return {"import_id": import_id, "skipped": record.status}

        try:
            with self._downloaded(record) as path:
                result = self._ingest.extract(self._request(record, path))
            if result.is_err():
                raise result.error
            curriculum: Curriculum = result.unwrap()
        except Exception as exc:
            return self._fail(record, "preview", exc, expect=("EXTRACTING",))

        preview = curriculum_preview(curriculum)
        status = "BLOCKED" if preview["blocking"] else "READY"
        self._store.transition(import_id, to=status, expect=("EXTRACTING",), preview=preview)
        logger.info(
            "curriculum_import.preview.done",
            import_id=import_id,
            course_id=record.course_id,
            status=status,
            **preview["counts"],
            issues=len(preview["issues"]),
        )
        return {"import_id": import_id, "status": status, **preview["counts"]}

    def apply(self, import_id: str) -> dict[str, Any]:
        record = self._store.get(import_id)
        if record is None:
            logger.warning("curriculum_import.apply.not_found", import_id=import_id)
            return {"import_id": import_id, "skipped": "not_found"}
        # core-api đã chuyển READY → APPLYING trước khi xếp job.
        if record.status != "APPLYING":
            logger.info("curriculum_import.apply.skipped", import_id=import_id, status=record.status)
            return {"import_id": import_id, "skipped": record.status}

        try:
            with self._downloaded(record) as path:
                result = self._ingest.execute(self._request(record, path))
            if result.is_err():
                raise result.error
            response = result.unwrap()
        except Exception as exc:
            return self._fail(record, "apply", exc, expect=("APPLYING",))

        mapped = self._remap_course_chunks(record.course_id)
        self._store.transition(import_id, to="APPLIED", expect=("APPLYING",))
        logger.info(
            "curriculum_import.apply.done",
            import_id=import_id,
            course_id=record.course_id,
            los=response.lo_count,
            chapters=response.chapter_count,
            chunk_lo_mappings=mapped,
            warnings=response.warnings,
        )
        return {
            "import_id": import_id,
            "status": "APPLIED",
            "learning_outcomes": response.lo_count,
            "chapters": response.chapter_count,
            "chunk_lo_mappings": mapped,
        }

    def _remap_course_chunks(self, course_id: str) -> int:
        """Tài liệu index trước khi có đề cương chưa có cạnh chunk → LO; map lại từng cái.

        Best-effort: lỗi ở một tài liệu không làm hỏng lần áp dụng đề cương.
        """
        total = 0
        for document_id in self._store.indexed_document_ids(course_id):
            result = self._map_chunks_to_los.execute(
                MapChunksToLosRequest(document_id=document_id, course_id=course_id)
            )
            if result.is_err():
                logger.warning(
                    "curriculum_import.remap_failed",
                    course_id=course_id,
                    document_id=document_id,
                    error=str(result.error),
                )
                continue
            total += result.unwrap().mapping_count
        return total

    def _fail(self, record: CurriculumImport, step: str, exc: Exception, *, expect: tuple[str, ...]):
        logger.exception(
            f"curriculum_import.{step}.failed",
            import_id=record.import_id,
            course_id=record.course_id,
            error=str(exc),
        )
        self._store.transition(
            record.import_id,
            to="FAILED",
            expect=expect,
            error=f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_CHARS],
        )
        return {"import_id": record.import_id, "status": "FAILED"}

    @staticmethod
    def _request(record: CurriculumImport, path: Path) -> IngestCurriculumRequest:
        return IngestCurriculumRequest(
            file_path=path,
            file_name=record.file_name,
            course_id=record.course_id,
        )

    @contextmanager
    def _downloaded(self, record: CurriculumImport) -> Iterator[Path]:
        suffix = Path(record.file_name).suffix or Path(record.storage_key).suffix or ".pdf"
        with tempfile.TemporaryDirectory(prefix="curriculum-") as tmp_dir:
            path = Path(tmp_dir) / f"syllabus{suffix}"
            result = self._file_storage.download(record.storage_key, path)
            if result.is_err():
                raise result.error
            yield path

"""
ExtractRelationsUseCase — dựng đồ thị quan hệ giữa các ý từ chunk của tài liệu.

Đường đi: chunk → LLM nêu quan hệ kèm trích dẫn → MÃ kiểm trích dẫn có nằm
nguyên văn trong chunk → ghi ``knowledge_relations``. Quan hệ không có trích
dẫn thật bị bỏ, nên mọi cạnh trong đồ thị đều trỏ được về một câu trong tài
liệu. Chạy theo lô vài chunk mỗi lần gọi và ghi từng lô; một lô lỗi không làm
mất các lô đã xong, chạy lại thì thay quan hệ của đúng các chunk đó.
"""
import time
import uuid
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from document_chunk.application.services.structured_output import generate_structured_payload
from document_chunk.domain.concept_keys import (
    MAX_LABEL_CHARS,
    RELATION_TYPES,
    concept_key,
    quote_is_verbatim,
)
from document_chunk.domain.exceptions import ProcessingError
from document_chunk.domain.ports.course_knowledge_store import (
    ICourseKnowledgeStore,
    KnowledgeChunk,
    NewRelation,
)
from document_chunk.domain.ports.llm_client import LLMUsage
from document_chunk.domain.ports.metadata_store import IMetadataStore
from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result

from worker.prompts import relations as prompts
from worker.use_cases.shared.llm_role import LlmRole

logger = get_logger(__name__)

# llm_usage_logs.status CHECK chỉ nhận ba giá trị này, phân biệt hoa thường.
_USAGE_OK = "OK"
_USAGE_ERROR = "ERROR"

CHUNKS_PER_CALL = 3
MAX_CHUNKS_PER_REQUEST = 150
MIN_CHUNK_CHARS = 80
MAX_CHUNK_CHARS = 2200
MAX_RELATIONS_PER_CHUNK = 6


class _RawRelation(BaseModel):
    source: str = ""
    relation: str = ""
    target: str = ""
    evidence_quote: str = ""


class _RawPassage(BaseModel):
    index: int
    relations: list[_RawRelation] = Field(default_factory=list)


class _Payload(BaseModel):
    passages: list[_RawPassage] = Field(default_factory=list)


@dataclass(frozen=True)
class ExtractRelationsRequest:
    course_id: str
    document_ids: list[str]


@dataclass
class ExtractRelationsResponse:
    course_id: str
    documents: int
    chunks_seen: int
    chunks_processed: int
    relations_written: int
    relations_dropped: int
    failed_batches: int = 0
    dropped_reasons: dict[str, int] = field(default_factory=dict)


def clean_relations(
    raw: _RawPassage | None, chunk: KnowledgeChunk
) -> tuple[list[NewRelation], dict[str, int]]:
    """
    Quan hệ hợp lệ của một chunk, và lý do mỗi quan hệ bị bỏ (để biết prompt hỏng ở đâu).

    Hợp lệ: loại quan hệ nằm trong danh sách, hai đầu khác rỗng và khác nhau sau
    khi chuẩn hóa, trích dẫn là chuỗi con của CHÍNH chunk này.
    """
    dropped: dict[str, int] = {}

    def drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    if raw is None:
        return [], dropped

    out: list[NewRelation] = []
    seen: set[tuple[str, str, str]] = set()
    for item in raw.relations:
        relation = item.relation.strip().lower()
        source = item.source.strip()[:MAX_LABEL_CHARS]
        target = item.target.strip()[:MAX_LABEL_CHARS]
        source_key, target_key = concept_key(source), concept_key(target)
        if relation not in RELATION_TYPES:
            drop("unknown_type")
        elif not source_key or not target_key:
            drop("empty_label")
        elif source_key == target_key:
            drop("self_loop")
        elif not quote_is_verbatim(item.evidence_quote, chunk.content):
            drop("quote_not_in_chunk")
        elif (source_key, relation, target_key) in seen:
            drop("duplicate")
        elif len(out) >= MAX_RELATIONS_PER_CHUNK:
            drop("over_limit")
        else:
            seen.add((source_key, relation, target_key))
            out.append(
                NewRelation(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source_label=source,
                    source_key=source_key,
                    relation_type=relation,
                    target_label=target,
                    target_key=target_key,
                    evidence_quote=" ".join(item.evidence_quote.split()),
                    page_number=chunk.page_number,
                )
            )
    return out, dropped


class ExtractRelationsUseCase:
    def __init__(
        self,
        knowledge_store: ICourseKnowledgeStore,
        metadata_store: IMetadataStore,
        writer: LlmRole,
        *,
        chunks_per_call: int = CHUNKS_PER_CALL,
        max_chunks: int = MAX_CHUNKS_PER_REQUEST,
    ) -> None:
        self._knowledge = knowledge_store
        self._metadata_store = metadata_store
        self._writer = writer
        self._chunks_per_call = max(1, chunks_per_call)
        self._max_chunks = max_chunks

    def execute(
        self, request: ExtractRelationsRequest
    ) -> Result[ExtractRelationsResponse, Exception]:
        if not request.document_ids:
            return Err(ProcessingError("document_ids must not be empty"))
        trace_id = str(uuid.uuid4())

        chunks: list[KnowledgeChunk] = []
        for document_id in request.document_ids:
            listed = self._knowledge.list_document_chunks(document_id)
            if listed.is_err():
                return Err(listed.error)
            chunks.extend(listed.unwrap())

        usable = [
            c for c in chunks if len(c.content.strip()) >= MIN_CHUNK_CHARS
        ][: self._max_chunks]
        if not usable:
            return Err(ProcessingError("No chunk with enough text to extract relations from"))

        response = ExtractRelationsResponse(
            course_id=request.course_id,
            documents=len(request.document_ids),
            chunks_seen=len(chunks),
            chunks_processed=0,
            relations_written=0,
            relations_dropped=0,
        )
        for start in range(0, len(usable), self._chunks_per_call):
            batch = usable[start : start + self._chunks_per_call]
            outcome = self._extract_batch(batch, request.course_id, trace_id)
            if outcome is None:
                response.failed_batches += 1
                continue
            written, dropped = outcome
            response.chunks_processed += len(batch)
            response.relations_written += written
            for reason, n in dropped.items():
                response.relations_dropped += n
                response.dropped_reasons[reason] = response.dropped_reasons.get(reason, 0) + n

        if response.chunks_processed == 0:
            return Err(
                ProcessingError(
                    f"Every one of {response.failed_batches} batches failed; nothing was saved"
                )
            )
        logger.info(
            "extract_relations.done",
            course_id=request.course_id,
            chunks=response.chunks_processed,
            written=response.relations_written,
            dropped=response.dropped_reasons,
            failed_batches=response.failed_batches,
        )
        return Ok(response)

    def _extract_batch(
        self, batch: list[KnowledgeChunk], course_id: str, trace_id: str
    ) -> tuple[int, dict[str, int]] | None:
        """(số quan hệ đã ghi, lý do bỏ) hoặc None khi cả lô hỏng."""
        prompt = prompts.extraction([c.content.strip()[:MAX_CHUNK_CHARS] for c in batch])
        started = time.monotonic()
        usage = LLMUsage()

        def on_usage(value: LLMUsage) -> None:
            nonlocal usage
            usage = usage + value

        result = generate_structured_payload(
            llm_client=self._writer.client,
            prompt=prompt.user,
            system=prompt.system,
            schema_model=_Payload,
            repair_schema_name="RelationExtraction",
            on_usage=on_usage,
        )
        self._log_usage(result.is_ok(), course_id, trace_id, usage, started)
        if result.is_err():
            logger.warning("extract_relations.batch_failed", error=str(result.error))
            return None

        by_index = {p.index: p for p in result.unwrap().passages}
        relations: list[NewRelation] = []
        dropped: dict[str, int] = {}
        for index, chunk in enumerate(batch):
            kept, lost = clean_relations(by_index.get(index), chunk)
            relations.extend(kept)
            for reason, n in lost.items():
                dropped[reason] = dropped.get(reason, 0) + n

        saved = self._knowledge.replace_relations(
            course_id=course_id,
            chunk_ids=[c.chunk_id for c in batch],
            extraction_version=prompts.EXTRACTION_VERSION,
            model=self._writer.client.model_id,
            relations=relations,
        )
        if saved.is_err():
            logger.warning("extract_relations.save_failed", error=str(saved.error))
            return None
        return saved.unwrap(), dropped

    def _log_usage(
        self, ok: bool, course_id: str, trace_id: str, usage: LLMUsage, started: float
    ) -> None:
        logged = self._metadata_store.record_llm_usage(
            provider=self._writer.provider,
            model=self._writer.client.model_id,
            use_case="relation_extraction",
            status=_USAGE_OK if ok else _USAGE_ERROR,
            course_id=course_id,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=usage.cost_usd,
            latency_ms=int((time.monotonic() - started) * 1000),
            trace_id=trace_id,
        )
        if logged.is_err():
            logger.warning("extract_relations.usage_log_failed", error=str(logged.error))


__all__ = [
    "ExtractRelationsRequest",
    "ExtractRelationsResponse",
    "ExtractRelationsUseCase",
    "clean_relations",
]

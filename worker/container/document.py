"""Container của document worker: parse → chunk → embed → index, nạp đề cương."""
from document_chunk.adapters.chunkers.structural import build_chunker as create_chunker
from document_chunk.adapters.embedders.bge_embedder import BgeEmbedder
from document_chunk.adapters.parsers.adaptive_pdf_parser import AdaptivePdfParser
from document_chunk.adapters.parsers.docx_parser import DocxParser
from document_chunk.adapters.parsers.markdown_parser import MarkdownParser
from document_chunk.adapters.parsers.pptx_parser import PptxParser
from document_chunk.domain.ports.chunker import IChunker
from document_chunk.shared.logger import get_logger

from worker.config import WorkerSettings, get_worker_settings
from worker.container.base import (
    BaseWorkerContainer,
    build_file_storage,
    build_graph_store,
    build_job_queue,
    build_metadata_store,
    build_result_publisher,
    build_vector_store,
)
from worker.use_cases.run_pipeline import RunPipelineUseCase

logger = get_logger(__name__)


def build_parsers(settings: WorkerSettings) -> list:
    return [
        AdaptivePdfParser(settings.parser),
        DocxParser(settings.parser),
        PptxParser(settings.parser),
        MarkdownParser(settings.parser),
    ]


def build_chunker(settings: WorkerSettings) -> IChunker:
    return create_chunker(settings.chunker, settings.embedder)


def build_embedder(settings: WorkerSettings):
    provider = settings.embedder.provider
    if provider == "grpc":
        from document_chunk.adapters.embedders.grpc_embedder import GrpcEmbedder

        logger.debug("worker_container.init", component="GrpcEmbedder")
        return GrpcEmbedder(settings.embedder, expected_dimension=settings.qdrant.vector_size)

    if provider != "bge":
        raise ValueError(
            "worker supports only EMBEDDER_PROVIDER=bge or grpc; "
            f"received {provider!r}"
        )
    logger.debug("worker_container.init", component="BgeEmbedder")
    return BgeEmbedder(settings.embedder)


class DocumentWorkerContainer(BaseWorkerContainer):
    """Build only the dependencies needed by the document processing worker."""

    def __init__(self, settings: WorkerSettings) -> None:
        super().__init__(settings)
        self.file_storage = self._track("file_storage", build_file_storage(settings))
        self.metadata_store = self._track("metadata_store", build_metadata_store(settings))
        self.vector_store = self._track("vector_store", build_vector_store(settings))
        self.job_queue = self._track("job_queue", build_job_queue(settings))
        self.result_publisher = self._track("result_publisher", build_result_publisher(settings))
        self.parsers = build_parsers(settings)
        self.chunker = build_chunker(settings)
        self.embedder = self._track("embedder", build_embedder(settings))
        self.graph_store = self._track("graph_store", build_graph_store(settings))
        self.map_chunks_to_los_use_case = self._build_map_chunks_to_los()
        self.run_pipeline_use_case = RunPipelineUseCase(
            parsers=self.parsers,
            chunker=self.chunker,
            embedder=self.embedder,
            vector_store=self.vector_store,
            metadata_store=self.metadata_store,
            job_queue=self.job_queue,
            result_publisher=self.result_publisher,
            map_chunks_to_los_use_case=self.map_chunks_to_los_use_case,
        )
        self.curriculum_import_use_case = self._build_curriculum_import()

    def _build_curriculum_import(self):
        from document_chunk.adapters.curriculum.dcmh_extractor import DcmhExtractor
        from document_chunk.application.use_cases.ingest_curriculum import (
            IngestCurriculumUseCase,
        )
        from worker.adapters.course_document_index import CourseDocumentIndex
        from worker.use_cases.curriculum_import import CurriculumImportUseCase

        # Worker chỉ parse và gửi kết quả; core-api ghi đề cương khi áp dụng.
        return CurriculumImportUseCase(
            publisher=self.result_publisher,
            file_storage=self.file_storage,
            ingest_curriculum_use_case=IngestCurriculumUseCase(
                parsers=self.parsers,
                curriculum_extractor=DcmhExtractor(),
            ),
            map_chunks_to_los_use_case=self.map_chunks_to_los_use_case,
            course_documents=CourseDocumentIndex(self.settings.sql),
        )

    def _build_map_chunks_to_los(self):
        from document_chunk.adapters.curriculum.embedding_chapter_matcher import (
            EmbeddingChapterMatcher,
        )
        from document_chunk.adapters.curriculum.heuristic_lo_mapper import HeuristicLoMapper
        from document_chunk.application.use_cases.map_chunks_to_los import (
            MapChunksToLosUseCase,
        )
        from worker.reporters.document_processing import PublishingDocumentChapterReporter

        return MapChunksToLosUseCase(
            metadata_store=self.metadata_store,
            graph_store=self.graph_store,
            lo_mapper=HeuristicLoMapper(),
            # Khớp chương theo nội dung cho tài liệu không nằm trong module theo chương.
            chapter_matcher=EmbeddingChapterMatcher(self.embedder),
            # Chương suy luận → core-api ghi documents (không ghi thẳng).
            chapter_reporter=PublishingDocumentChapterReporter(self.result_publisher),
        )


def build_document_container(settings: WorkerSettings | None = None) -> DocumentWorkerContainer:
    return DocumentWorkerContainer(settings or get_worker_settings("document"))

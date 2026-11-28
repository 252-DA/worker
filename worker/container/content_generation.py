"""Container của content-generation worker: quiz, card, quan hệ, Micro Content."""
import threading

from ai_runtime.catalog import QUIZ_GENERATION, QUIZ_VERIFICATION
from document_chunk.shared.logger import get_logger

from worker.config import WorkerSettings, get_worker_settings
from worker.container.base import (
    BaseWorkerContainer,
    build_learning_context_client,
    build_metadata_store,
    build_result_publisher,
)
from worker.container.llm import TaskRoles, build_model_registry

logger = get_logger(__name__)


class CurriculumQuizContainer(BaseWorkerContainer):
    """Dependencies for queued, LO-scoped quiz and lesson-card generation."""

    def __init__(self, settings: WorkerSettings) -> None:
        super().__init__(settings)
        self.metadata_store = self._track("metadata_store", build_metadata_store(settings))
        self.result_publisher = self._track("result_publisher", build_result_publisher(settings))
        self.model_registry = build_model_registry(settings)
        self.roles = TaskRoles(self.model_registry)
        self.context_client = build_learning_context_client(settings)
        self._warned_same_model: str | None = None
        self._warn_lock = threading.Lock()

    def _writer_and_judge(self):
        """
        Người viết câu và người chấm là hai model khác nhau trong cùng một
        process: model vừa viết một câu sai có xu hướng chấm là câu đúng.
        Hai task trỏ cùng profile thì registry trả lại đúng một client.
        """
        writer = self.roles.role(QUIZ_GENERATION)
        judge = self.roles.role(QUIZ_VERIFICATION)
        same = writer.profile if writer.profile == judge.profile else None
        with self._warn_lock:
            if same and same != self._warned_same_model:
                logger.warning(
                    "worker_container.quiz_judge_is_writer",
                    profile=same,
                    hint="chọn model kiểm chứng khác model viết nội dung",
                )
            self._warned_same_model = same
        return writer, judge

    @property
    def generate_curriculum_quiz_use_case(self):
        from worker.use_cases.generate_curriculum_quiz import GenerateCurriculumQuizUseCase

        writer, judge = self._writer_and_judge()
        return GenerateCurriculumQuizUseCase(
            metadata_store=self.metadata_store,
            llm_client=writer.client,
            context_client=self.context_client,
            fallback_to_local_context=self.settings.learning_context_mcp.fallback_to_local,
            llm_provider=writer.provider,
            llm_profile=writer.profile,
            verifier=judge,
        )

    @property
    def generate_lesson_cards_use_case(self):
        """
        Card dùng chung cặp writer/judge của quiz: catalog của ai-sdk chưa có
        task riêng cho card, và cả hai đều là "viết nội dung bám nguồn rồi để
        model khác chấm". Chi phí vẫn tách được qua use_case card_*.
        """
        from worker.use_cases.generate_lesson_cards import GenerateLessonCardsUseCase

        writer, judge = self._writer_and_judge()
        return GenerateLessonCardsUseCase(
            metadata_store=self.metadata_store,
            llm_client=writer.client,
            context_client=self.context_client,
            fallback_to_local_context=self.settings.learning_context_mcp.fallback_to_local,
            llm_provider=writer.provider,
            llm_profile=writer.profile,
            verifier=judge,
        )

    @property
    def extract_relations_use_case(self):
        """
        Trích quan hệ giữa các ý cho chatbot. Việc này là "viết nội dung bám
        nguồn" nên dùng model viết của quiz; chi phí vẫn tách được qua use_case
        relation_extraction. PostgresMetadataStore cũng là ICourseKnowledgeStore.
        """
        from worker.use_cases.extract_relations import ExtractRelationsUseCase

        return ExtractRelationsUseCase(
            knowledge_store=self.metadata_store,
            metadata_store=self.metadata_store,
            writer=self.roles.role(QUIZ_GENERATION),
        )

    @property
    def generate_micro_outline_use_case(self):
        from worker.use_cases.generate_micro_outline import GenerateMicroOutlineUseCase

        writer = self.roles.role(QUIZ_GENERATION)
        return GenerateMicroOutlineUseCase(
            metadata_store=self.metadata_store,
            llm_client=writer.client,
            llm_provider=writer.provider,
            llm_profile=writer.profile,
        )

    @property
    def generate_video_script_use_case(self):
        from worker.use_cases.generate_video_script import GenerateVideoScriptUseCase

        writer, judge = self._writer_and_judge()
        return GenerateVideoScriptUseCase(
            metadata_store=self.metadata_store,
            llm_client=writer.client,
            llm_provider=writer.provider,
            llm_profile=writer.profile,
            verifier=judge,
        )

    @property
    def generate_micro_quiz_use_case(self):
        from worker.use_cases.generate_micro_quiz import GenerateMicroQuizUseCase

        writer, judge = self._writer_and_judge()
        return GenerateMicroQuizUseCase(
            metadata_store=self.metadata_store,
            llm_client=writer.client,
            llm_provider=writer.provider,
            llm_profile=writer.profile,
            verifier=judge,
        )


def build_curriculum_quiz_container(
    settings: WorkerSettings | None = None,
) -> CurriculumQuizContainer:
    return CurriculumQuizContainer(settings or get_worker_settings())

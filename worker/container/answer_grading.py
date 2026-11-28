"""Container của answer-grading worker."""
from ai_runtime.catalog import QUIZ_VERIFICATION

from worker.config import WorkerSettings, get_worker_settings
from worker.container.base import (
    BaseWorkerContainer,
    build_file_storage,
    build_metadata_store,
    build_result_publisher,
)
from worker.container.llm import TaskRoles, build_model_registry


class AnswerGradingContainer(BaseWorkerContainer):
    """
    Chấm bài của người học: tự luận ngắn trong quiz, câu tự luận Micro Content
    và bài luận nộp file qua Canvas (Asset Processor). Lệnh mang đủ dữ liệu
    chấm, kết quả về core-api qua ``result_publisher``: không đọc/ghi bảng bài
    làm. metadata store chỉ còn để ghi llm_usage_logs; MinIO để đọc file bài
    luận; không MCP, không writer.
    """

    def __init__(self, settings: WorkerSettings) -> None:
        super().__init__(settings)
        self.metadata_store = self._track("metadata_store", build_metadata_store(settings))
        self.file_storage = self._track("file_storage", build_file_storage(settings))
        self.result_publisher = self._track("result_publisher", build_result_publisher(settings))
        self.model_registry = build_model_registry(settings)
        self.roles = TaskRoles(self.model_registry)

    @property
    def grade_essay_submission_use_case(self):
        """Bài luận cũng là việc phán xét: dùng model judge như chấm tự luận ngắn."""
        from worker.use_cases.grade_essay_submission import GradeEssaySubmissionUseCase

        return GradeEssaySubmissionUseCase(
            file_storage=self.file_storage,
            metadata_store=self.metadata_store,
            grader=self.roles.role(QUIZ_VERIFICATION),
        )

    @property
    def grade_short_answers_use_case(self):
        """
        Chấm tự luận là việc phán xét như bước kiểm tra câu hỏi, nên dùng
        profile của judge. Chi phí tách được qua use_case answer_grading.
        """
        from worker.use_cases.grade_short_answers import GradeShortAnswersUseCase

        return GradeShortAnswersUseCase(
            metadata_store=self.metadata_store,
            grader=self.roles.role(QUIZ_VERIFICATION),
        )

    @property
    def grade_micro_content_answers_use_case(self):
        """Chấm câu tự luận Micro Content theo rubric, cùng prompt và model judge với quiz."""
        from worker.use_cases.grade_micro_content_answers import GradeMicroContentAnswersUseCase

        return GradeMicroContentAnswersUseCase(
            metadata_store=self.metadata_store,
            grader=self.roles.role(QUIZ_VERIFICATION),
        )


def build_answer_grading_container(
    settings: WorkerSettings | None = None,
) -> AnswerGradingContainer:
    return AnswerGradingContainer(settings or get_worker_settings("answer_grading"))

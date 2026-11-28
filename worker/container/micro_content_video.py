"""Container của micro-content-video worker."""
from worker.config import WorkerSettings, get_worker_settings
from worker.container.base import BaseWorkerContainer, build_file_storage, build_result_publisher


class MicroContentVideoContainer(BaseWorkerContainer):
    """Phục vụ sinh video AI cho Micro Content (slide/Manim + giọng đọc + render)."""

    def __init__(self, settings: WorkerSettings) -> None:
        super().__init__(settings)
        from worker.services.video.slide_generator import SlideGenerator
        from worker.services.video.tts_client import get_default_tts_client
        from worker.services.video.video_composer import VideoComposer

        self.file_storage = self._track("file_storage", build_file_storage(settings))
        self.result_publisher = self._track("result_publisher", build_result_publisher(settings))
        self.tts_client = get_default_tts_client()
        self.slide_generator = SlideGenerator()
        self.video_composer = VideoComposer()

    @property
    def render_micro_content_video_use_case(self):
        from worker.use_cases.render_micro_content_video import RenderMicroContentVideoUseCase

        return RenderMicroContentVideoUseCase(
            result_publisher=self.result_publisher,
            file_storage=self.file_storage,
            tts_client=self.tts_client,
            slide_generator=self.slide_generator,
            video_composer=self.video_composer,
        )


def build_micro_content_video_container(
    settings: WorkerSettings | None = None,
) -> MicroContentVideoContainer:
    return MicroContentVideoContainer(settings or get_worker_settings("micro_content_video"))

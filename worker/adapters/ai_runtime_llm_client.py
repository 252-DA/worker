from ai_runtime import AIRuntime
from ai_runtime.errors import AIRuntimeError

from document_chunk.domain.exceptions import LLMError
from document_chunk.domain.ports.llm_client import ILLMClient
from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result

logger = get_logger(__name__)


class AIRuntimeLLMClient(ILLMClient):
    """Compatibility adapter while existing use cases still depend on ILLMClient."""

    def __init__(self, runtime: AIRuntime) -> None:
        self._runtime = runtime

    @property
    def model_id(self) -> str:
        return self._runtime.model_id

    def generate(
        self,
        prompt: str,
        system: str | None = None,
    ) -> Result[str, Exception]:
        try:
            response = self._runtime.generate_text(prompt=prompt, system=system)
        except AIRuntimeError as exc:
            return Err(LLMError("AI runtime generation failed", cause=exc))

        logger.info(
            "ai_runtime.generate.completed",
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.total_tokens,
        )
        return Ok(response.text)


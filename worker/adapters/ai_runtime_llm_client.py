from ai_runtime import AIRuntime
from ai_runtime.errors import AIRuntimeError

from document_chunk.domain.exceptions import LLMError
from document_chunk.domain.ports.llm_client import ILLMClient, LLMUsage
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
        result = self.generate_with_usage(prompt, system=system)
        if result.is_err():
            return result
        return Ok(result.unwrap()[0])

    def generate_with_usage(
        self,
        prompt: str,
        system: str | None = None,
    ) -> Result[tuple[str, LLMUsage], Exception]:
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
        # Runtime vẫn luôn trả token; trước đây chúng chỉ được log rồi bỏ đi nên
        # llm_usage_logs ghi 0 token cho mọi lần gọi.
        usage = LLMUsage(
            prompt_tokens=int(getattr(response.usage, "input_tokens", 0) or 0),
            completion_tokens=int(getattr(response.usage, "output_tokens", 0) or 0),
            cost_usd=float(getattr(response.usage, "cost_usd", 0) or 0),
        )
        return Ok((response.text, usage))


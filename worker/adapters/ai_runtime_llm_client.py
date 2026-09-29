from ai_runtime import AIRuntime, ModelProfile
from ai_runtime.errors import AIRuntimeError

from document_chunk.domain.exceptions import LLMError
from document_chunk.domain.ports.llm_client import ILLMClient, LLMUsage
from document_chunk.shared.logger import get_logger
from document_chunk.shared.result import Err, Ok, Result

logger = get_logger(__name__)


class AIRuntimeLLMClient(ILLMClient):
    """Compatibility adapter while existing use cases still depend on ILLMClient."""

    def __init__(self, runtime: AIRuntime, profile: ModelProfile | None = None) -> None:
        self._runtime = runtime
        self._profile = profile
        self._warned_unpriced = False

    @property
    def model_id(self) -> str:
        return self._runtime.model_id

    @property
    def provider(self) -> str:
        return self._profile.provider if self._profile else "unknown"

    @property
    def profile_name(self) -> str | None:
        return self._profile.name if self._profile else None

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
            profile=self.profile_name,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.total_tokens,
        )
        # Runtime vẫn luôn trả token; trước đây chúng chỉ được log rồi bỏ đi nên
        # llm_usage_logs ghi 0 token cho mọi lần gọi.
        usage = LLMUsage(
            prompt_tokens=int(response.usage.input_tokens or 0),
            completion_tokens=int(response.usage.output_tokens or 0),
            cost_usd=self._cost_usd(response.usage),
        )
        return Ok((response.text, usage))

    def _cost_usd(self, usage) -> float:
        """
        Quy token thành USD theo bảng giá của profile.

        Trước đây chỗ này đọc ``usage.cost_usd`` — một field ``TokenUsage``
        không có — nên mọi dòng ``llm_usage_logs`` ghi chi phí 0. Profile chưa
        khai giá thì vẫn ghi 0, nhưng nói ra một lần trong log: 0 vì "chưa biết
        giá" khác hẳn 0 vì "miễn phí".
        """
        if self._profile is None:
            return 0.0
        cost = self._profile.cost_usd(usage.input_tokens, usage.output_tokens)
        if cost is None:
            if not self._warned_unpriced:
                self._warned_unpriced = True
                logger.warning(
                    "llm.cost.unpriced",
                    profile=self._profile.name,
                    model=self._profile.model,
                    reason="profile has no rate card; llm_usage_logs.cost_usd stays 0",
                )
            return 0.0
        return cost

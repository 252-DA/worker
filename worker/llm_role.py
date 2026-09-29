"""Một vai trò gọi model, kèm nhãn để ghi sổ chi phí.

Từ khi model được chọn theo task (docs/model-routing-design.md), hai vai trò
trong cùng một process có thể là hai model khác nhau. Provider và profile vì
thế phải đi theo từng vai trò: ghi provider của người viết câu cho dòng chấm
là ghi sai tiền vào ``llm_usage_logs``.
"""

from dataclasses import dataclass

from document_chunk.domain.ports.llm_client import ILLMClient


@dataclass(frozen=True)
class LlmRole:
    client: ILLMClient
    provider: str = "unknown"
    profile: str | None = None

    @property
    def model_id(self) -> str:
        return self.client.model_id


__all__ = ["LlmRole"]

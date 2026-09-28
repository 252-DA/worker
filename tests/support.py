"""Test double dùng chung."""
from document_chunk.domain.ports.llm_client import LLMUsage
from document_chunk.shared.result import Ok


def as_llm_client(mock):
    """
    Cho một MagicMock hành xử đúng như ``ILLMClient``.

    Chỗ gọi nào cần token sẽ dùng ``generate_with_usage``; test chỉ stub
    ``generate`` nên cần nối hai đường lại, nếu không mock trả về MagicMock và
    lỗi chỉ hiện ra ở tận chỗ giải nén tuple. Usage để rỗng: test double không
    có số liệu thật, và 0 token nghĩa là "không đo được", không phải "miễn phí".
    """
    def _with_usage(prompt, system=None):
        result = mock.generate(prompt, system=system)
        if result.is_err():
            return result
        return Ok((result.unwrap(), LLMUsage()))

    mock.generate_with_usage.side_effect = _with_usage
    return mock

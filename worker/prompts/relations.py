"""
Prompt trích quan hệ giữa các ý trong tài liệu (``ExtractRelationsUseCase``).

Một lần gọi xử lý vài đoạn liền nhau. Mô hình chỉ được nêu quan hệ mà đoạn
*phát biểu rõ*, kèm một trích dẫn nguyên văn; mã kiểm trích dẫn có nằm trong
đoạn không rồi mới giữ (xem ``use_cases/extract_relations.py``).
"""
from collections.abc import Sequence

from document_chunk.domain.concept_keys import RELATION_TYPES

from worker.prompts._common import Prompt, json_system

# Đổi khi đổi prompt hoặc luật kiểm: quan hệ ghi bằng phiên bản cũ vẫn truy được.
EXTRACTION_VERSION = "rel-v1"

_RELATION_HELP = (
    "- cause: the source brings about, leads to or enables the target\n"
    "- part_of: the source is a component, factor, step or member of the target\n"
    "- contrast: the source differs from or is the opposite of the target\n"
    "- prerequisite: the source must be understood or true before the target\n"
    "- example_of: the source is an example or instance of the target\n"
    "- defines: the source is defined as, or is equivalent to, the target\n"
    "- related: an explicit link that fits none of the above"
)


def extraction(passages: Sequence[str]) -> Prompt:
    numbered = "\n\n".join(f"[{index}]\n{text}" for index, text in enumerate(passages))
    return Prompt(
        name="relation_extraction",
        system=json_system(
            "You extract explicit relations between concepts from university course material."
        ),
        user=(
            "Below are numbered passages from course slides. For each passage list up to 6 "
            "relations between two concepts that the passage STATES EXPLICITLY.\n\n"
            f"Allowed relation types ({', '.join(RELATION_TYPES)}):\n{_RELATION_HELP}\n\n"
            "Rules:\n"
            "- source and target are short noun phrases (at most 6 words) written as in the "
            "passage, in the language of the passage. Use the same phrase every time the same "
            "concept appears.\n"
            "- evidence_quote is a contiguous quote copied EXACTLY from that passage (at least 12 "
            "characters) that states the relation. Do not paraphrase, translate or merge sentences.\n"
            "- Skip a passage that states no relation. Never infer a relation the passage does not "
            "say, and never use outside knowledge.\n\n"
            f"Passages:\n{numbered}\n\n"
            "Return strict JSON:\n"
            '{"passages":[{"index":0,"relations":[{"source":"...","relation":"cause",'
            '"target":"...","evidence_quote":"..."}]}]}'
        ),
    )


__all__ = ["EXTRACTION_VERSION", "extraction"]

"""
Mọi prompt worker gửi cho LLM.

Mỗi module là một họ prompt và trả về ``Prompt``. Use case chỉ gửi
``system``/``user`` đi và lấy ``name`` làm ``use_case`` khi ghi chi phí; sửa câu
chữ của prompt thì sửa ở đây.

- ``curriculum_quiz``: câu trắc nghiệm theo LO (sinh, kiểm tra, sửa)
- ``lesson_cards``: micro-content card theo LO (sinh, kiểm tra, sửa)
- ``enrichment``: card và quiz theo section tài liệu, đường cũ không kiểm tra

Prompt sửa JSON hỏng nằm ở ``document_chunk.application.services.structured_output``
vì dùng chung cho mọi schema; prompt tutor nằm ở ``ai_tutor.prompts``.
"""
from worker.prompts._common import Prompt

__all__ = ["Prompt"]

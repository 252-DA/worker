"""
Phần dùng chung giữa các use case: vai trò model (``llm_role``), schema câu hỏi
quiz, chấm theo rubric, vòng kiểm chứng bằng judge, ngữ cảnh LO, cắt/định dạng
chunk.

Use case chỉ import nhau qua gói này, không import thẳng một use case khác.
Gói này không import use case nào.
"""

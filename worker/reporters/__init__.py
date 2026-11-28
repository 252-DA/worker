"""
Mọi kết quả worker gửi về core-api qua hàng đợi ``worker_results``, mỗi loại
kết quả một module: module đó biết ``kind``, cách dựng ``result_id`` và hình
dạng payload (khớp contracts/integration-events/v1/<kind>.json).

Use case và runner không tự gọi ``publisher.publish*`` hay ``make_result_id``:
chúng gọi reporter ở đây. Muốn biết một job báo gì về core-api thì đọc module
tương ứng:

- ``document_processing``: trạng thái và chương của tài liệu (pipeline, enrichment, outbox)
- ``enrichment_content``: học liệu nháp của enrichment
- ``curriculum_import``: bản xem trước đề cương
- ``content_generation``: vòng đời một yêu cầu sinh nội dung
- ``grading``: chấm tự luận quiz, Micro Content, bài luận
- ``micro_content_video``: kết quả render video

Gửi đi bằng ``worker.adapters.result_publisher``. Reporter không import use
case hay runner (chỉ nhận đối tượng của chúng qua tham số).
"""

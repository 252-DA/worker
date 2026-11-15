"""
Prompt cho luồng micro-content video & quiz từ slide chương:
  - ``outline_suggestion``: tách bộ slide thành các nội dung 2–4 phút + chọn LO chính
  - ``video_script_generation`` / ``video_script_verification`` / ``video_script_regeneration``:
    sinh kịch bản từng cảnh bám slide theo chính sách (b) (§8.6) và ngôn ngữ (§8.1)
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from worker.prompts._common import Prompt, json_system


def outline_suggestion(
    *,
    course_title: str,
    document_title: str,
    chapter_code: str | None,
    total_pages: int,
    page_summaries: Sequence[dict[str, Any]],
    candidate_los: Sequence[dict[str, Any]],
) -> Prompt:
    lo_lines: list[str] = []
    for lo in candidate_los:
        preferred_tag = " [ƯU TIÊN - LO của chương này]" if lo.get("preferred") else ""
        en_part = f" ({lo['statement_en']})" if lo.get("statement_en") else ""
        lo_lines.append(
            f"- lo_id={lo['lo_id']} | {lo['code']}{preferred_tag}: {lo['statement_vi']}{en_part}"
        )
    los_block = "\n".join(lo_lines) if lo_lines else "(học phần chưa có LO lá nào)"

    page_lines: list[str] = []
    for page in page_summaries:
        pno = page["page_number"]
        heading = page.get("heading") or ""
        snippet = page.get("text_snippet") or "[slide ít chữ / chỉ hình ảnh]"
        chars = page.get("char_count", 0)
        head_str = f" | heading: {heading}" if heading else ""
        page_lines.append(f"[Trang {pno}/{total_pages}{head_str} | {chars} ký tự]\n{snippet}")
    pages_block = "\n\n".join(page_lines)

    chapter_note = f"Chương: {chapter_code}" if chapter_code else "Chương: (chưa gắn)"
    user = (
        f"Học phần: {course_title}\n"
        f"Tài liệu slide: {document_title} ({chapter_note}, tổng {total_pages} trang)\n\n"
        "Mục tiêu: Tách toàn bộ bộ slide dưới đây thành các nội dung học độc lập (micro-content), "
        "mỗi nội dung tương ứng một video ngắn 2–4 phút cho sinh viên.\n\n"
        "DANH SÁCH LO LÁ CỦA HỌC PHẦN (chọn ĐÚNG MỘT lo_id từ danh sách nếu phù hợp ngữ nghĩa):\n"
        f"{los_block}\n\n"
        "NỘI DUNG TỪNG TRANG SLIDE (1..N):\n"
        f"{pages_block}\n\n"
        "YÊU CẦU BẮT BUỘC:\n"
        "1. Chia thành 1 đến tối đa 20 nội dung liên tiếp nhau, KHÔNG gối trang, KHÔNG lộn ngược thứ tự "
        f"(page_from <= page_to, trong khoảng 1..{total_pages}). Mỗi nội dung nên dài 2–15 trang "
        "(chỉ dùng 1 trang nếu không thể ghép hợp lý hoặc cả phần chỉ còn 1 trang kiến thức).\n"
        "2. Bỏ qua các trang bìa/mục lục thuần túy hoặc gộp trang chuyển tiếp vào đúng nội dung liền kề; "
        "KHÔNG bỏ sót trang kiến thức chính.\n"
        "3. Mỗi nội dung cần:\n"
        "   - `title`: tiêu đề tiếng Việt ngắn gọn, rõ nghĩa khi đứng một mình (<= 100 ký tự).\n"
        "   - `summary`: đúng 1 câu tiếng Việt tóm tắt người học sẽ nắm được gì.\n"
        "   - `page_from`, `page_to`: số nguyên trang bắt đầu/kết thúc.\n"
        "   - `suggested_lo_id`: UUID `lo_id` khớp nhất theo phát biểu LO (ưu tiên LO gắn `[ƯU TIÊN]` nếu phù hợp, "
        "hoặc chọn LO lá gần nhất trong học phần; chỉ trả `null` nếu không có LO nào phù hợp).\n"
        "   - `lo_rationale`: 1 câu ngắn giải thích lý do chọn LO đó (hoặc `null` nếu `suggested_lo_id` là `null`).\n"
        "   - `low_text`: `true` nếu dải trang đó chủ yếu là hình ảnh/sơ đồ/slide ít chữ.\n\n"
        "Trả về ĐÚNG JSON theo mẫu:\n"
        '{"sections":[{"title":"...","summary":"...","page_from":1,"page_to":5,'
        '"suggested_lo_id":"...","lo_rationale":"...","low_text":false}]}'
    )
    return Prompt(
        name="micro_outline_suggestion",
        system=json_system(
            "You are a curriculum designer splitting lecture slides into coherent micro-learning video modules."
        ),
        user=user,
    )


def _format_slide_pages(pages: Sequence[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for p in pages:
        pno = p["page_number"]
        cids = p.get("chunk_ids") or []
        cid_str = f" | chunk_ids={','.join(cids)}" if cids else ""
        content = (p.get("content") or "").strip() or "[Trang này chủ yếu là hình ảnh/sơ đồ hoặc không trích được văn bản]"
        blocks.append(f"=== TRANG {pno}{cid_str} ===\n{content}")
    return "\n\n".join(blocks)


def video_script_generation(
    *,
    course_title: str,
    topic_title: str,
    topic_summary: str | None,
    lo_line_text: str | None,
    page_from: int,
    page_to: int,
    pages: Sequence[dict[str, Any]],
    neighbor_topics: Sequence[str],
    low_text: bool = False,
) -> Prompt:
    neighbor_block = (
        "\n".join(f"- {t}" for t in neighbor_topics)
        if neighbor_topics
        else "(không có)"
    )
    sparse_note = (
        "\nLưu ý: Dải trang này có nhiều slide ít chữ/hình ảnh; bạn có thể viết lời dẫn giải sư phạm "
        "ngắn gọn bằng tiếng Việt để kết nối ý mạch lạc, nhưng TUYỆT ĐỐI KHÔNG bịa kiến thức mới.\n"
        if low_text
        else ""
    )
    lo_block = f"LO mục tiêu: {lo_line_text}\n" if lo_line_text else ""
    summary_block = f"Tóm tắt nội dung: {topic_summary}\n" if topic_summary else ""

    user = (
        f"Học phần: {course_title}\n"
        f"Chủ đề video: {topic_title}\n"
        f"{summary_block}{lo_block}"
        f"Dải trang slide nguồn: trang {page_from} đến trang {page_to}\n"
        f"Các nội dung khác cùng bộ slide (KHÔNG được nhắc 'bài trước'/'phần sau' sang các mục này):\n{neighbor_block}\n"
        f"{sparse_note}\n"
        "NỘI DUNG CÁC TRANG SLIDE NGUỒN:\n"
        f"{_format_slide_pages(pages)}\n\n"
        "YÊU CẦU KỊCH BẢN (thời lượng mục tiêu 2–4 phút):\n"
        "1. Tạo từ **4 đến 8 cảnh** (`scenes`).\n"
        "2. Mỗi cảnh gồm:\n"
        f"   - `slide_page`: số nguyên trang slide mà cảnh giảng (BẮT BUỘC trong [{page_from}, {page_to}]).\n"
        "   - `title`: tiêu đề ngắn cho màn hình cảnh (<= 80 ký tự).\n"
        "   - `bullets`: từ 1 đến 4 gạch đầu dòng súc tích hiển thị trên slide (giữ thuật ngữ chuyên ngành/tên riêng).\n"
        "   - `narration`: lời giảng đọc tự nhiên bằng tiếng Việt, độ dài **BẮT BUỘC 40 đến 110 từ mỗi cảnh**, "
        "không chứa Markdown (`**`, `` ` ``, `#`), không emoji, không URL.\n"
        "3. QUY TẮC NGÔN NGỮ:\n"
        "   - Giảng hoàn toàn bằng tiếng Việt tự nhiên.\n"
        "   - Giữ nguyên tên riêng và thuật ngữ kỹ thuật phổ biến (ví dụ: Spark, RDD, SparkContext, JVM, HDFS, API, SQL).\n"
        "   - Dịch sang tiếng Việt các từ tiếng Anh thông thường (ví dụ: abstraction → lớp trừu tượng, collection → tập hợp).\n"
        "4. QUY TẮC BÁM NGUỒN (CHÍNH SÁCH B):\n"
        "   - ĐƯỢC PHÉP diễn giải sư phạm ngắn gọn, đặt câu chuyển ý mượt khi slide ít chữ.\n"
        "   - CẤM TUYỆT ĐỐI tự thêm: (a) số liệu, mốc thời gian, phần trăm, phiên bản; "
        "(b) tên riêng, thư viện, API, lệnh mới không có trong slide; "
        "(c) khẳng định sự thật/kiến thức mới ngoài slide; "
        "(d) cụm từ tham chiếu ngoài bài như 'trong bài trước', 'ở phần sau', 'video tiếp theo'.\n\n"
        "Trả về ĐÚNG JSON:\n"
        '{"scenes":[{"title":"...","slide_page":1,"bullets":["..."],"narration":"..."}]}'
    )
    return Prompt(
        name="micro_video_script_generation",
        system=json_system(
            "You are a Vietnamese university lecturer creating a concise, strictly grounded micro-video script from slide pages."
        ),
        user=user,
    )


def video_script_verification(
    *,
    page_from: int,
    page_to: int,
    pages: Sequence[dict[str, Any]],
    scenes: Sequence[Any],
) -> Prompt:
    rendered_scenes: list[str] = []
    for idx, scene in enumerate(scenes):
        words = len((scene.narration or "").split())
        rendered_scenes.append(
            f"[Cảnh index={idx}] slide_page={scene.slide_page} | words={words}\n"
            f"Title: {scene.title}\n"
            f"Bullets: {json.dumps(list(scene.bullets), ensure_ascii=False)}\n"
            f"Narration: {scene.narration}"
        )
    scenes_block = "\n\n".join(rendered_scenes)

    user = (
        f"Dải trang nguồn hợp lệ: trang {page_from}..{page_to}.\n\n"
        "NỘI DUNG SLIDE NGUỒN:\n"
        f"{_format_slide_pages(pages)}\n\n"
        "CÁC CẢNH CẦN KIỂM CHỨNG:\n"
        f"{scenes_block}\n\n"
        "TIÊU CHÍ CHẤM TỪNG CẢNH:\n"
        "- Được phép: diễn giải sư phạm ngắn gọn và dịch thuật từ tiếng Anh thông thường sang tiếng Việt nếu KHÔNG thêm thông tin mới.\n"
        "- Phải đánh `contradicted` hoặc `not_enough_evidence` (kèm `reason` tiếng Việt cụ thể) nếu cảnh vi phạm BẤT KỲ điểm nào:\n"
        "  1. Thêm số liệu, tỷ lệ, hiệu năng, phiên bản hoặc ngày tháng không có trong slide.\n"
        "  2. Thêm tên riêng, API, công cụ hoặc khái niệm kỹ thuật mới không có trong slide.\n"
        "  3. Đưa ra khẳng định kiến thức hoặc ví dụ thực tế cụ thể không được slide hậu thuẫn.\n"
        "  4. Gắn sai `slide_page` hoặc nhắc 'bài trước'/'bài sau'/'chương trước'.\n"
        "  5. Độ dài narration < 40 từ hoặc > 110 từ.\n\n"
        "Trả về ĐÚNG JSON cho TẤT CẢ các cảnh theo đúng index:\n"
        '{"verdicts":[{"scene_index":0,"verdict":"supported|contradicted|not_enough_evidence","reason":"..."}]}'
    )
    return Prompt(
        name="micro_video_script_verification",
        system=json_system(
            "You are a strict fact-checking verifier checking Vietnamese lecture video scenes against source slides."
        ),
        user=user,
    )


def video_script_regeneration(
    *,
    topic_title: str,
    page_from: int,
    page_to: int,
    pages: Sequence[dict[str, Any]],
    scene: Any,
    reason: str,
) -> Prompt:
    user = (
        f"Chủ đề: {topic_title} (trang {page_from}..{page_to})\n\n"
        "NỘI DUNG SLIDE NGUỒN:\n"
        f"{_format_slide_pages(pages)}\n\n"
        f"CẢNH CẦN VIẾT LẠI:\n{json.dumps(scene.model_dump(), ensure_ascii=False)}\n\n"
        f"LÝ DO BỊ GẮN CỜ:\n{reason}\n\n"
        "YÊU CẦU VIẾT LẠI CẢNH NÀY:\n"
        f"- Giữ `slide_page` trong [{page_from}, {page_to}].\n"
        "- Tối đa 4 gạch đầu dòng trong `bullets`.\n"
        "- `narration` tiếng Việt tự nhiên từ **40 đến 110 từ**, không Markdown, không URL.\n"
        "- Loại bỏ toàn bộ số liệu/tên riêng/khẳng định ngoài slide và không nhắc 'bài trước'/'bài sau'.\n\n"
        "Trả về ĐÚNG JSON cho 1 cảnh:\n"
        '{"title":"...","slide_page":1,"bullets":["..."],"narration":"..."}'
    )
    return Prompt(
        name="micro_video_script_regeneration",
        system=json_system(
            "You are revising a single lecture video scene to make it strictly faithful to the source slides."
        ),
        user=user,
    )

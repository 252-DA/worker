"""
Prompt cho luồng micro-content video & quiz từ slide chương:
  - ``outline_suggestion``: tách bộ slide thành các nội dung 2–4 phút + chọn LO chính
  - ``video_script_generation`` / ``video_script_verification`` / ``video_script_regeneration``:
    sinh kịch bản từng cảnh bám slide theo chính sách (b) (§8.6) và ngôn ngữ (§8.1)
  - ``micro_quiz_generation`` / ``micro_quiz_verification`` / ``micro_quiz_regeneration``:
    sinh câu hỏi củng cố bám đúng phạm vi trang slide & kịch bản của bài micro-content
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from worker.prompts._common import SOURCE_LANGUAGE, Prompt, excerpts, json_system
from worker.prompts.curriculum_quiz import _KINDS


def outline_suggestion(
    *,
    course_title: str,
    document_title: str,
    chapter_code: str | None,
    total_pages: int,
    page_summaries: Sequence[dict[str, Any]],
    candidate_los: Sequence[dict[str, Any]],
    max_sections: int = 20,
) -> Prompt:
    lo_lines: list[str] = []
    for lo in candidate_los:
        preferred_tag = " [ƯU TIÊN - LO của chương này]" if lo.get("preferred") else ""
        en_part = f" ({lo['statement_en']})" if lo.get("statement_en") else ""
        lo_lines.append(
            f"- lo_id={lo['lo_id']} | {lo['code']}{preferred_tag}: {lo['statement_vi']}{en_part}"
        )
    los_block = "\n".join(lo_lines) if lo_lines else "(học phần chưa có LO lá nào)"

    total_chars = sum(int(p.get("char_count", 0)) for p in page_summaries)
    page_lines: list[str] = []
    for page in page_summaries:
        pno = page["page_number"]
        heading = page.get("heading") or ""
        snippet = page.get("text_snippet") or "[slide ít chữ / chỉ hình ảnh]"
        chars = page.get("char_count", 0)
        head_str = f" | heading: {heading}" if heading else ""
        role_note = " | GỢI Ý: có thể là trang bìa/kết thúc" if page.get("is_boilerplate") else ""
        page_lines.append(
            f"[Trang {pno}/{total_pages}{head_str} | {chars} ký tự{role_note}]\n{snippet}"
        )
    pages_block = "\n\n".join(page_lines)

    # Bộ slide dài: 1.000–2.500 ký tự/nội dung sẽ vượt trần max_sections, model
    # phải chọn một trong hai và hay chọn vượt trần. Nói rõ trần thắng.
    min_chars_per_section = -(-total_chars // max_sections) if total_chars else 0
    if min_chars_per_section > 2500:
        size_rule = (
            f"Bộ slide này dài (~{total_chars:,} ký tự) nên mỗi nội dung phải gộp nhiều hơn, "
            f"khoảng **{min_chars_per_section:,} ký tự chữ trở lên**, để tổng số nội dung KHÔNG vượt {max_sections}"
        )
    else:
        size_rule = (
            "~1.000–2.500 ký tự chữ cho mỗi nội dung, thường gộp **3–10 trang** liên tiếp nếu slide thưa chữ "
            "hoặc nhiều hình ảnh"
        )

    chapter_note = f"Chương: {chapter_code}" if chapter_code else "Chương: (chưa gắn)"
    user = (
        f"Học phần: {course_title}\n"
        f"Tài liệu slide: {document_title} ({chapter_note}, tổng {total_pages} trang, ~{total_chars:,} ký tự chữ)\n\n"
        "Mục tiêu: Tách bộ slide dưới đây thành các nội dung học độc lập (micro-content), "
        "mỗi nội dung tương ứng một video ngắn 1,5–4 phút đủ dày kiến thức cho sinh viên.\n\n"
        "DANH SÁCH LO LÁ CỦA HỌC PHẦN (chọn ĐÚNG MỘT lo_id từ danh sách nếu phù hợp ngữ nghĩa):\n"
        f"{los_block}\n\n"
        "NỘI DUNG TỪNG TRANG SLIDE (1..N):\n"
        f"{pages_block}\n\n"
        "YÊU CẦU BẮT BUỘC:\n"
        f"1. Chia theo **dung lượng kiến thức thực tế** ({size_rule}). Tổng cộng **TỐI ĐA {max_sections} nội dung** "
        f"(giới hạn cứng, ưu tiên hơn mức ký tự ở trên) và KHÔNG gối trang, "
        f"KHÔNG lộn ngược thứ tự (page_from <= page_to, trong khoảng 1..{total_pages}). "
        "TUYỆT ĐỐI không cắt vụn thành nhiều phần nhỏ dưới ~450 ký tự chỉ để chia đều số trang.\n"
        "2. **Bỏ qua hoàn toàn trang bìa** (trang mở đầu chỉ có tên môn, tên giảng viên, phòng lab, ngày tháng), "
        "trang mục lục/agenda thuần túy, và trang cảm ơn / hỏi đáp (Thank you / Q&A) cuối bài — không đưa các trang này "
        "vào `page_from..page_to`. Ngược lại, KHÔNG bỏ sót trang kiến thức chính.\n"
        "3. Mỗi nội dung cần:\n"
        "   - `title`: tiêu đề tiếng Việt ngắn gọn, rõ nghĩa khi đứng một mình (<= 100 ký tự).\n"
        "   - `summary`: đúng 1 câu tiếng Việt tóm tắt người học sẽ nắm được gì.\n"
        "   - `page_from`, `page_to`: số nguyên trang bắt đầu/kết thúc.\n"
        "   - `suggested_lo_id`: UUID `lo_id` khớp nhất theo phát biểu LO (ưu tiên LO gắn `[ƯU TIÊN]` nếu phù hợp, "
        "hoặc chọn LO lá gần nhất trong học phần; chỉ trả `null` nếu không có LO nào phù hợp).\n"
        "   - `lo_rationale`: 1 câu ngắn giải thích lý do chọn LO đó (hoặc `null` nếu `suggested_lo_id` là `null`).\n"
        "   - `low_text`: `true` nếu dải trang đó chủ yếu là hình ảnh/sơ đồ/slide ít chữ.\n\n"
        "Trả về ĐÚNG JSON theo mẫu:\n"
        '{"sections":[{"title":"...","summary":"...","page_from":2,"page_to":8,'
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
    instructor_prompt: str | None = None,
    min_scenes: int = 4,
    max_scenes: int = 8,
    min_words: int = 40,
    max_words: int = 110,
) -> Prompt:
    neighbor_block = (
        "\n".join(f"- {t}" for t in neighbor_topics)
        if neighbor_topics
        else "(không có)"
    )
    sparse_note = (
        "\nLưu ý: Dải trang này có nhiều slide ít chữ/hình ảnh; hãy giữ số cảnh gọn, bám sát đúng chữ có trên slide, "
        "tuyệt đối KHÔNG độn thêm cảnh lặp ý hay bịa kiến thức mới.\n"
        if low_text
        else ""
    )
    lo_block = f"LO mục tiêu: {lo_line_text}\n" if lo_line_text else ""
    summary_block = f"Tóm tắt nội dung: {topic_summary}\n" if topic_summary else ""
    instructor_block = (
        f"YÊU CẦU BỔ SUNG CỦA GIẢNG VIÊN (ưu tiên đáp ứng trong phạm vi slide nguồn): {instructor_prompt.strip()}\n"
        if instructor_prompt and instructor_prompt.strip()
        else ""
    )

    user = (
        f"Học phần: {course_title}\n"
        f"Chủ đề video: {topic_title}\n"
        f"{summary_block}{lo_block}{instructor_block}"
        f"Dải trang slide nguồn: trang {page_from} đến trang {page_to}\n"
        f"Các nội dung khác cùng bộ slide (KHÔNG được nhắc 'bài trước'/'phần sau' sang các mục này):\n{neighbor_block}\n"
        f"{sparse_note}\n"
        "NỘI DUNG CÁC TRANG SLIDE NGUỒN:\n"
        f"{_format_slide_pages(pages)}\n\n"
        "YÊU CẦU KỊCH BẢN (thời lượng mục tiêu 1,5–4 phút):\n"
        f"1. Tạo từ **{min_scenes} đến {max_scenes} cảnh** (`scenes`), co gọn tự nhiên theo đúng lượng kiến thức "
        "có trên slide (KHÔNG cố kéo dài nếu ít ý và KHÔNG lặp lại nội dung cảnh trước).\n"
        "2. CẤU TRÚC MẠCH DẪN CÂU HỎI SƯ PHẠM:\n"
        "   - Cảnh mở đầu đặt 1–2 câu hỏi trọng tâm dẫn dắt vấn đề bài học (bỏ qua thông tin hành chính ở trang bìa như tên giảng viên/lab nếu có).\n"
        "   - Các cảnh tiếp theo lần lượt giải đáp từng phần theo mạch logic rõ ràng.\n"
        "   - Cảnh cuối đúc kết câu trả lời trọng tâm và kết nối tự nhiên sang phần kiểm tra hiểu bài.\n"
        "3. Mỗi cảnh gồm:\n"
        f"   - `slide_page`: số nguyên trang slide mà cảnh giảng (BẮT BUỘC trong [{page_from}, {page_to}]).\n"
        "   - `title`: tiêu đề ngắn cho màn hình cảnh (<= 80 ký tự).\n"
        "   - `bullets`: từ 1 đến 4 gạch đầu dòng súc tích hiển thị trên slide (giữ thuật ngữ chuyên ngành/tên riêng).\n"
        f"   - `narration`: lời giảng đọc tự nhiên bằng tiếng Việt, độ dài **BẮT BUỘC {min_words} đến {max_words} từ mỗi cảnh**, "
        "không chứa Markdown (`**`, `` ` ``, `#`), không emoji, không URL.\n"
        "4. QUY TẮC NGÔN NGỮ:\n"
        "   - Giảng hoàn toàn bằng tiếng Việt tự nhiên.\n"
        "   - Giữ nguyên tên riêng và thuật ngữ kỹ thuật phổ biến (ví dụ: Spark, RDD, SparkContext, JVM, HDFS, API, SQL).\n"
        "   - Dịch sang tiếng Việt các từ tiếng Anh thông thường (ví dụ: abstraction → lớp trừu tượng, collection → tập hợp).\n"
        "5. QUY TẮC BÁM NGUỒN (CHÍNH SÁCH B):\n"
        "   - ĐƯỢC PHÉP diễn giải sư phạm ngắn gọn, đặt câu hỏi dẫn dắt và câu chuyển ý mượt.\n"
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
        rendered_scenes.append(
            f"[Cảnh index={idx}] slide_page={scene.slide_page}\n"
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
        "TIÊU CHÍ CHẤM ĐỘ BÁM NGUỒN TỪNG CẢNH:\n"
        "- Được phép: câu hỏi dẫn nhập, diễn giải sư phạm ngắn gọn và dịch thuật từ tiếng Anh thông thường sang tiếng Việt nếu KHÔNG thêm kiến thức mới.\n"
        "- CHỈ đánh `contradicted` hoặc `not_enough_evidence` (kèm `reason` tiếng Việt cụ thể) nếu cảnh vi phạm tính xác thực nguồn:\n"
        "  1. Thêm số liệu, tỷ lệ, hiệu năng, phiên bản hoặc ngày tháng không có trong slide.\n"
        "  2. Thêm tên riêng, API, công cụ hoặc khái niệm kỹ thuật mới không có trong slide.\n"
        "  3. Đưa ra khẳng định kiến thức hoặc ví dụ thực tế cụ thể không được slide hậu thuẫn.\n"
        "  4. Gắn sai `slide_page` hoặc nhắc 'bài trước'/'bài sau'/'chương trước'.\n"
        "- Lưu ý: KHÔNG đánh rớt chỉ vì số từ dài hay ngắn nếu toàn bộ nội dung đều bám slide.\n\n"
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
    min_words: int = 30,
    max_words: int = 110,
) -> Prompt:
    user = (
        f"Chủ đề: {topic_title} (trang {page_from}..{page_to})\n\n"
        "NỘI DUNG SLIDE NGUỒN:\n"
        f"{_format_slide_pages(pages)}\n\n"
        f"CẢNH CẦN VIẾT LẠI:\n{json.dumps(scene.model_dump(), ensure_ascii=False)}\n\n"
        f"LÝ DO CẦN SỬA:\n{reason}\n\n"
        "YÊU CẦU VIẾT LẠI CẢNH NÀY:\n"
        f"- Giữ `slide_page` trong [{page_from}, {page_to}].\n"
        "- Tối đa 4 gạch đầu dòng trong `bullets`.\n"
        f"- `narration` tiếng Việt tự nhiên từ **{min_words} đến {max_words} từ**, không Markdown, không URL.\n"
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


def micro_quiz_generation(
    *,
    course_title: str,
    topic_title: str,
    topic_summary: str | None,
    page_from: int,
    page_to: int,
    lo_code: str,
    lo_vi: str,
    lo_en: str | None = None,
    count: int = 3,
    context_parts: Sequence[str],
    question_type: str = "MCQ_SINGLE",
) -> Prompt:
    kind = _KINDS[question_type]
    lo_context = (
        f"LO tham khảo ngữ cảnh chung ({lo_code}): {lo_vi}"
        + (f" ({lo_en})" if lo_en else "")
        if lo_vi
        else ""
    )
    summary_line = f"Tóm tắt bài học: {topic_summary}\n" if topic_summary else ""
    lo_line_note = f"{lo_context}\n" if lo_context else ""

    user = (
        f"Học phần: {course_title}\n"
        f"Bài micro-content: {topic_title} (trang slide {page_from}–{page_to})\n"
        f"{summary_line}"
        f"{lo_line_note}\n"
        "QUY TẮC PHẠM VI BẮT BUỘC:\n"
        f"- Tạo ĐÚNG {count} câu hỏi ({kind.noun}) để củng cố kiến thức ĐÃ DẠY trong kịch bản video "
        f"và các trang slide {page_from}–{page_to} dưới đây.\n"
        "- TUYỆT ĐỐI KHÔNG đặt câu hỏi về khái niệm, đặc trưng hay ví dụ chưa được dạy trong kịch bản/"
        "slide nguồn bên dưới, kể cả khi khái niệm đó có xuất hiện trong phát biểu LO chung của học phần.\n"
        "- Mỗi câu hỏi cần rõ ràng, không mơ hồ, có đúng một ý kiểm tra trọng tâm của bài.\n"
        f"{kind.rules}"
        f"{SOURCE_LANGUAGE}"
        "- QUY TẮC LỜI GIẢI THÍCH (`explanation`): Viết giải thích sư phạm bằng tiếng Việt tự nhiên cho người học. "
        "CẤM TUYỆT ĐỐI nhắc mã chunk (ví dụ: `chunk ...`, UUID), ký hiệu `[1]`, tên trường nội bộ hay tên mức Bloom "
        "(`understand`, `remember`, `apply`, `mức độ hiểu`, ...) trong `explanation`; "
        "có thể nhắc tự nhiên như 'Theo nội dung bài giảng...' hoặc 'Theo slide trang ...'.\n"
        "- Trường `lo_alignment_rationale` ghi ngắn 1 câu câu hỏi củng cố ý nào trong bài học; "
        "`source_chunk_ids` liệt kê các chunk ID trích dẫn thực tế để hệ thống lưu vết.\n\n"
        f"Nguồn nội dung bài học (kịch bản & slide trang {page_from}–{page_to}):\n"
        f"{excerpts(context_parts)}\n\n"
        'Return strict JSON:\n{"questions":[' + kind.item_json + "]}"
    )
    return Prompt(
        name="micro_quiz_generation",
        system=json_system(
            "You are a Vietnamese university assessment specialist writing concise micro-learning quiz questions strictly scoped to the provided lesson content."
        ),
        user=user,
    )


def micro_quiz_verification(
    *,
    page_from: int,
    page_to: int,
    context_parts: Sequence[str],
    questions: Sequence[Any],
    question_type: str = "MCQ_SINGLE",
) -> Prompt:
    kind = _KINDS[question_type]
    rendered = "\n\n".join(kind.render(i, q) for i, q in enumerate(questions))
    return Prompt(
        name="micro_quiz_verification",
        system=json_system("You are a strict fact-checking verifier checking micro-learning quiz scope and grounding."),
        user=(
            f"Bạn kiểm tra câu hỏi củng cố cho bài micro-content thuộc dải trang {page_from}–{page_to}.\n\n"
            f"Nguồn nội dung bài học:\n{excerpts(context_parts)}\n\n"
            "Với mỗi câu hỏi, chấm:\n"
            "- `supported`: CẢ câu hỏi, đáp án đúng VÀ phần giải thích đều được dạy rõ ràng trong nguồn trên, và đáp án nhiễu không đúng.\n"
            "- `contradicted`: sai kiến thức so với nguồn hoặc đáp án/nhiễu bị ngược.\n"
            "- `not_enough_evidence`: câu hỏi lấn sang khái niệm KHÔNG có trong kịch bản/dải trang slide này (ví dụ mượn ý từ LO chung chưa dạy) hoặc nguồn không đủ chứng minh đáp án.\n\n"
            f"{kind.check_subject}:\n{rendered}\n\n"
            "Return strict JSON evaluating EVERY question index:\n"
            f'{{"verdicts":[{kind.verdicts}]}}'
        ),
    )


def micro_quiz_regeneration(
    *,
    page_from: int,
    page_to: int,
    context_parts: Sequence[str],
    original: Any,
    reason: str,
    question_type: str = "MCQ_SINGLE",
) -> Prompt:
    kind = _KINDS[question_type]
    return Prompt(
        name="micro_quiz_regeneration",
        system=json_system("You are an assessment writer fixing one rejected micro-quiz question."),
        user=(
            f"Câu hỏi ({kind.regen_subject}) dưới đây bị loại vì chưa bám đúng nội dung bài học trang {page_from}–{page_to}. "
            f"Hãy viết lại sao cho bám sát hoàn toàn nguồn dưới đây; {kind.regen_freedom}.\n"
            "Lưu ý `explanation` viết tự nhiên cho người học, TUYỆT ĐỐI không nhắc chunk ID, `[1]` hay mức Bloom.\n\n"
            f"Nguồn nội dung bài học:\n{excerpts(context_parts)}\n\n"
            f"{kind.render_rejected(original)}"
            f"Lý do bị loại: {reason}\n\n"
            "Return strict JSON for ONE corrected question:\n"
            f"{kind.item_json}"
        ),
    )

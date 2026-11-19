from __future__ import annotations

import base64
import math
import re
from typing import Any

try:
    import pymupdf as fitz  # PyMuPDF modern import
except ImportError:
    import fitz

from document_chunk.shared.logger import get_logger

logger = get_logger(__name__)


def normalize_for_tts(text: str) -> str:
    """Chuyển các từ viết tắt và ký hiệu sang văn nói tiếng Việt tự nhiên để TTS đọc mượt."""
    if not text:
        return ""

    replacements = [
        # Viết tắt sư phạm & học thuật
        (r"\bVD\s*:", "Ví dụ:"),
        (r"\bvd\b", "ví dụ"),
        (r"\bT/c\b", "Tính chất"),
        (r"\bt/c\b", "tính chất"),
        (r"\bĐ/n\b", "Định nghĩa"),
        (r"\bđ/n\b", "định nghĩa"),
        (r"\bv\.v\.(\b|$)", "vân vân "),
        (r"\bv\.v\b", "vân vân"),
        (r"\bth\b", "trường hợp"),
        (r"\bTH\b", "Trường hợp"),
        (r"\bĐK\b", "Điều kiện"),
        (r"\bđk\b", "điều kiện"),
        (r"\bCSDL\b", "Cơ sở dữ liệu"),
        (r"\bcsdl\b", "cơ sở dữ liệu"),
        (r"\bCNTT\b", "Công nghệ thông tin"),
        (r"\bcntt\b", "công nghệ thông tin"),
        (r"\bAI\b", "Trí tuệ nhân tạo"),
        (r"\bML\b", "Học máy"),
        (r"\bSV\b", "Sinh viên"),
        (r"\bGV\b", "Giảng viên"),
        (r"\bpt\b", "phương trình"),
        (r"\bhpt\b", "hệ phương trình"),
        (r"\bbpt\b", "bất phương trình"),

        # Toán học & Logic & Ký hiệu
        (r"->|→", " dẫn đến "),
        (r"<-|←", " bắt nguồn từ "),
        (r"=>|⇒", " suy ra "),
        (r"<=>|⇔", " tương đương với "),
        (r">=", " lớn hơn hoặc bằng "),
        (r"<=", " nhỏ hơn hoặc bằng "),
        (r"!=", " khác "),
        (r"≠", " khác "),
        (r"==", " bằng "),
        (r"=", " bằng "),
        (r"\+", " cộng "),
        (r"%", " phần trăm "),
        (r"\bmax\b", "lớn nhất"),
        (r"\bmin\b", "nhỏ nhất"),

        # Độ phức tạp thuật toán O(...)
        (r"\bO\s*\(\s*1\s*\)", "độ phức tạp ô một"),
        (r"\bO\s*\(\s*N\s+log\s+N\s*\)", "độ phức tạp ô N lô-ga-rít N"),
        (r"\bO\s*\(\s*N\s*\^\s*2\s*\)", "độ phức tạp ô N bình phương"),
        (r"\bO\s*\(\s*N\s*2\s*\)", "độ phức tạp ô N bình phương"),
        (r"\bO\s*\(\s*N\s*\)", "độ phức tạp ô N"),
        (r"\bO\s*\(\s*log\s+N\s*\)", "độ phức tạp ô lô-ga-rít N"),
        (r"\bO\s*\(([^)]+)\)", r"độ phức tạp ô \1"),
        (r"\blog\s*N\b", "lô-ga-rít N"),
        (r"\blog2\s*N\b", "lô-ga-rít cơ số 2 của N"),
    ]

    res = text
    for pattern, repl in replacements:
        res = re.sub(pattern, repl, res, flags=re.IGNORECASE)

    # Dọn dẹp khoảng trắng dư thừa
    res = re.sub(r"\s+", " ", res).strip()
    return res


def extract_structured_slides(
    pdf_bytes: bytes,
    start_page: int = 1,
    end_page: int = 20,
) -> tuple[list[dict[str, Any]], int]:
    """Bóc tách cấu trúc slide từ PDF stream trong RAM theo font size và margin cropping.

    Returns:
        (slides_data, total_pages)
    """
    if not pdf_bytes:
        raise ValueError("Dữ liệu file PDF trống.")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Không thể mở file PDF: {exc}") from exc

    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        raise ValueError("File PDF không chứa trang nào.")

    start_idx = max(0, start_page - 1)
    end_idx = min(total_pages, max(start_idx + 1, end_page))

    if start_idx >= total_pages:
        doc.close()
        raise ValueError(
            f"Trang bắt đầu ({start_page}) vượt quá tổng số trang của file ({total_pages} trang)."
        )

    # 1. Kiểm tra PDF scan (nếu tổng ký tự trung bình < 15 ký tự/trang)
    total_chars = 0
    selected_page_count = max(1, end_idx - start_idx)
    for p_no in range(start_idx, end_idx):
        page_raw = doc[p_no].get_text("text") or ""
        total_chars += len(page_raw.strip())

    avg_chars = total_chars / selected_page_count
    if avg_chars < 15:
        doc.close()
        raise ValueError(
            "File PDF có vẻ là dạng ảnh scan (không có lớp chữ có thể chọn). "
            "Vui lòng sử dụng file PDF chứa văn bản (selectable text) để hệ thống trích xuất nội dung."
        )

    slides_data: list[dict[str, Any]] = []

    # 2. Duyệt từng trang và lọc rác
    for page_num in range(start_idx, end_idx):
        page = doc[page_num]
        page_height = page.rect.height
        page_dict = page.get_text("dict")

        lines_with_size: list[tuple[float, str]] = []

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # Chỉ lấy text block (bỏ ảnh, đồ họa)
                continue

            bbox = block.get("bbox", [0, 0, 0, 0])
            # Margin Cropping: Bỏ qua Header (7% mép trên) và Footer/Số trang (8% mép dưới)
            if bbox[1] < page_height * 0.07 or bbox[3] > page_height * 0.92:
                continue

            for line in block.get("lines", []):
                spans = line.get("spans", [])
                line_text = "".join(s.get("text", "") for s in spans).strip()

                if not line_text:
                    continue

                # Bỏ qua các dòng chỉ chứa số trang (e.g. "12", "12/45", "Trang 5", "Page 5 of 20")
                if re.match(r"^(\d+|trang\s*\d+|page\s*\d+)(\s*/\s*\d+)?$", line_text, re.IGNORECASE):
                    continue

                max_size = max((s.get("size", 12) for s in spans), default=12)
                lines_with_size.append((float(max_size), line_text))

        if not lines_with_size:
            continue

        # Dòng có font size lớn nhất làm Tiêu đề slide
        max_font = max(item[0] for item in lines_with_size)
        title_parts: list[str] = []
        body_lines: list[str] = []

        for size, text in lines_with_size:
            if size >= max_font - 0.8 and len(title_parts) < 2:
                title_parts.append(text)
            else:
                body_lines.append(text)

        title = " ".join(title_parts).strip()

        # Bỏ qua các slide kết thúc như Thank You / Q&A / Hỏi đáp
        if re.search(r"^(thank you|thanks|cảm ơn|q\s*&?\s*a|hỏi\s*đáp|hỏi & đáp)$", title, re.IGNORECASE):
            continue

        # Chuẩn hóa bullets (giữ nguyên công thức, code, ký hiệu)
        clean_bullets: list[str] = []
        for bl in body_lines:
            # Bỏ ký tự đầu dòng kiểu •, -, *, 1., a)
            cleaned = re.sub(r"^[\s•\-*+▪▫–—\d\.\)\:]+\s*", "", bl).strip()
            if cleaned and len(cleaned) >= 2:
                # Nếu bullet quá dài (> 20 từ), ngắt bớt cho gọn slide
                words = cleaned.split()
                if len(words) > 18:
                    cleaned = " ".join(words[:18]) + "..."
                clean_bullets.append(cleaned)

        slides_data.append({
            "page": page_num + 1,
            "title": title or f"Nội dung trang {page_num + 1}",
            "bullets": clean_bullets,
        })

    doc.close()
    return slides_data, total_pages


def cluster_and_format_scenes(
    slides_data: list[dict[str, Any]],
    target_scenes: int = 5,
    topic: str | None = None,
) -> list[dict[str, Any]]:
    """Gộp / phân cụm các slide thành target_scenes cảnh sư phạm hoàn chỉnh với narration."""
    if not slides_data:
        default_topic = topic or "Tổng quan bài giảng"
        return [
            {
                "title": default_topic,
                "bullets": ["Giới thiệu nội dung trọng tâm", "Mục tiêu bài học"],
                "narration": normalize_for_tts(
                    f"Chào mừng các bạn đến với bài học {default_topic}. "
                    f"Hôm nay chúng ta sẽ cùng tìm hiểu các nội dung trọng tâm và mục tiêu cần đạt được."
                ),
            }
        ]

    target_count = max(1, min(target_scenes, 10))
    total_slides = len(slides_data)

    scenes: list[dict[str, Any]] = []

    if total_slides <= target_count:
        for idx, slide in enumerate(slides_data):
            title = slide["title"]
            bullets = slide.get("bullets", [])
            if not bullets:
                bullets = ["Nội dung trọng tâm bài học"]

            # Giữ tối đa 4 bullets cho slide
            display_bullets = bullets[:4]

            # Tạo narration tự nhiên từ bullets
            explanation_parts = []
            for b in display_bullets:
                clean_b = b.rstrip(".")
                explanation_parts.append(f"{clean_b}.")

            narration_raw = (
                f"Ở phần {title}, "
                + " ".join(explanation_parts)
                + " Các bạn lưu ý ghi nhớ điểm mấu chốt này nhé."
            )

            scenes.append({
                "title": title,
                "bullets": display_bullets,
                "narration": normalize_for_tts(narration_raw),
            })
    else:
        # Gom nhóm slides thành target_count phần
        step = total_slides / target_count
        for i in range(target_count):
            start_i = int(i * step)
            end_i = int((i + 1) * step) if i < target_count - 1 else total_slides
            group = slides_data[start_i:end_i]
            if not group:
                continue

            first_slide = group[0]
            title = first_slide["title"]

            # Gộp bullets từ các slide trong group
            merged_bullets: list[str] = []
            for s in group:
                for b in s.get("bullets", []):
                    if b not in merged_bullets and len(merged_bullets) < 4:
                        merged_bullets.append(b)

            if not merged_bullets:
                merged_bullets = ["Khái niệm và nguyên lý hoạt động", "Ứng dụng thực tế"]

            display_bullets = merged_bullets[:4]

            bullet_sentences = ". ".join(b.rstrip(".") for b in display_bullets)
            narration_raw = (
                f"Tiếp theo là phần {title}. "
                f"Các điểm cốt lõi bao gồm: {bullet_sentences}. "
                f"Đây là kiến thức quan trọng phục vụ cho phần thực hành."
            )

            scenes.append({
                "title": title,
                "bullets": display_bullets,
                "narration": normalize_for_tts(narration_raw),
            })

    return scenes


def extract_script_from_pdf(
    pdf_bytes: bytes,
    start_page: int = 1,
    end_page: int = 20,
    target_scenes: int = 5,
    topic: str | None = None,
) -> dict[str, Any]:
    """Hàm điều phối chính: Nhận PDF stream, bóc tách slide và trả về scenes JSON."""
    slides_data, total_pages = extract_structured_slides(
        pdf_bytes=pdf_bytes,
        start_page=start_page,
        end_page=end_page,
    )

    scenes = cluster_and_format_scenes(
        slides_data=slides_data,
        target_scenes=target_scenes,
        topic=topic,
    )

    return {
        "status": "ok",
        "total_pages": total_pages,
        "extracted_slides": len(slides_data),
        "scenes": scenes,
    }


from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from document_chunk.shared.logger import get_logger

logger = get_logger(__name__)

# Common font candidates across Linux / Windows
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "DejaVuSans.ttf",
    "arial.ttf",
]


def _get_font(size: int, bold: bool = False) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    if bold:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
        ] + FONT_CANDIDATES
    else:
        candidates = FONT_CANDIDATES
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue

    try:
        return ImageFont.load_default()
    except Exception:
        return None  # type: ignore


def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    lines: list[str] = []
    words = text.split()
    if not words:
        return [""]

    current_line = words[0]
    for word in words[1:]:
        test_line = f"{current_line} {word}"
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]
        if line_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines


class SlideGenerator:
    """Tạo slide 1080p và các trạng thái hiện dần bullet cho từng cảnh."""

    def __init__(self, width: int = 1920, height: int = 1080) -> None:
        self.width = width
        self.height = height

    def generate_slide(
        self,
        title: str,
        bullets: Sequence[str],
        output_path: str,
        scene_index: int = 1,
        total_scenes: int = 1,
    ) -> str:
        return self._generate_slide(
            title=title,
            bullets=bullets,
            output_path=output_path,
            scene_index=scene_index,
            total_scenes=total_scenes,
            visible_bullet_count=None,
        )

    def generate_slide_states(
        self,
        title: str,
        bullets: Sequence[str],
        output_prefix: str,
        scene_index: int = 1,
        total_scenes: int = 1,
    ) -> list[str]:
        bullet_list = list(bullets)[:5]
        paths: list[str] = []
        for visible_count in range(len(bullet_list) + 1):
            output_path = f"{output_prefix}_{visible_count:02d}.png"
            paths.append(
                self._generate_slide(
                    title=title,
                    bullets=bullet_list,
                    output_path=output_path,
                    scene_index=scene_index,
                    total_scenes=total_scenes,
                    visible_bullet_count=visible_count,
                )
            )
        return paths

    def _generate_slide(
        self,
        title: str,
        bullets: Sequence[str],
        output_path: str,
        scene_index: int,
        total_scenes: int,
        visible_bullet_count: int | None,
    ) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        scale = self.width / 1280

        def s(value: int) -> int:
            return round(value * scale)

        img = Image.new("RGB", (self.width, self.height), color=(15, 23, 42))
        draw = ImageDraw.Draw(img)

        badge_font = _get_font(s(18), bold=True)
        title_font = _get_font(s(38), bold=True)
        bullet_font = _get_font(s(24), bold=False)

        card_x0, card_y0 = s(80), s(60)
        card_x1, card_y1 = self.width - s(80), self.height - s(60)
        draw.rounded_rectangle(
            [card_x0, card_y0, card_x1, card_y1],
            radius=s(20),
            fill=(30, 41, 59),
            outline=(51, 65, 85),
            width=s(2),
        )
        draw.rounded_rectangle(
            [card_x0 + s(20), card_y0 + s(2), card_x1 - s(20), card_y0 + s(8)],
            radius=s(3),
            fill=(59, 130, 246),
        )

        badge_text = f"CẢNH {scene_index} / {total_scenes}"
        draw.text(
            (card_x0 + s(50), card_y0 + s(35)),
            badge_text,
            fill=(96, 165, 250),
            font=badge_font,
        )

        max_title_w = (card_x1 - card_x0) - s(100)
        title_lines = _wrap_text(title, title_font, max_title_w, draw)
        curr_y = card_y0 + s(75)
        for line in title_lines[:2]:
            draw.text((card_x0 + s(50), curr_y), line, fill=(248, 250, 252), font=title_font)
            curr_y += s(50)

        curr_y += s(15)
        draw.line(
            [(card_x0 + s(50), curr_y), (card_x1 - s(50), curr_y)],
            fill=(71, 85, 105),
            width=s(1),
        )
        curr_y += s(35)

        max_bullet_w = (card_x1 - card_x0) - s(140)
        visible_bullets = bullets if visible_bullet_count is None else bullets[:visible_bullet_count]
        for bullet in visible_bullets[:5]:
            lines = _wrap_text(bullet, bullet_font, max_bullet_w, draw)
            draw.ellipse(
                [card_x0 + s(50), curr_y + s(8), card_x0 + s(60), curr_y + s(18)],
                fill=(59, 130, 246),
            )
            for line in lines:
                draw.text(
                    (card_x0 + s(75), curr_y),
                    line,
                    fill=(226, 232, 240),
                    font=bullet_font,
                )
                curr_y += s(36)
            curr_y += s(12)

        img.save(output_path, "PNG")
        logger.info("slide_generator.saved", path=output_path, scene=scene_index)
        return output_path


def render_pdf_slide_page(
    pdf_path: str,
    page_number: int,
    output_path: str,
    *,
    width: int = 1920,
    height: int = 1080,
) -> str:
    """Kết xuất nguyên trang PDF thành ảnh 16:9, giữ nguyên hình và biểu đồ nguồn."""
    import pymupdf as fitz

    with fitz.open(pdf_path) as doc:
        if page_number < 1 or page_number > len(doc):
            raise ValueError(f"Slide page {page_number} out of bounds (total={len(doc)})")
        page = doc[page_number - 1]
        rect = page.rect
        zoom = min(width / max(1.0, rect.width), height / max(1.0, rect.height))
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        rendered = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    canvas = Image.new("RGB", (width, height), (248, 250, 252))
    offset_x = max(0, (width - rendered.width) // 2)
    offset_y = max(0, (height - rendered.height) // 2)
    canvas.paste(rendered, (offset_x, offset_y))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "PNG")
    logger.info("slide_generator.pdf_page_saved", path=output_path, page_number=page_number)
    return output_path

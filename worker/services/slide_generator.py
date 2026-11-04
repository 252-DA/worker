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
    """Tạo slide ảnh 1280x720 cho các cảnh trong Micro Content bằng Pillow."""

    def __init__(self, width: int = 1280, height: int = 720) -> None:
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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Background canvas (Slate-950)
        img = Image.new("RGB", (self.width, self.height), color=(15, 23, 42))
        draw = ImageDraw.Draw(img)

        # Header bar / badge
        badge_font = _get_font(18, bold=True)
        title_font = _get_font(38, bold=True)
        bullet_font = _get_font(24, bold=False)

        # Inner Card container: x: 80, y: 60, w: 1120, h: 600
        card_x0, card_y0 = 80, 60
        card_x1, card_y1 = self.width - 80, self.height - 60
        card_radius = 20

        # Draw card background (Slate-900)
        draw.rounded_rectangle(
            [card_x0, card_y0, card_x1, card_y1],
            radius=card_radius,
            fill=(30, 41, 59),
            outline=(51, 65, 85),
            width=2,
        )

        # Top Accent gradient-like line (Blue-500)
        draw.rounded_rectangle(
            [card_x0 + 20, card_y0 + 2, card_x1 - 20, card_y0 + 8],
            radius=3,
            fill=(59, 130, 246),
        )

        # Scene counter badge (e.g. "CẢNH 1 / 3")
        badge_text = f"CẢNH {scene_index} / {total_scenes}"
        draw.text(
            (card_x0 + 50, card_y0 + 35),
            badge_text,
            fill=(96, 165, 250),
            font=badge_font,
        )

        # Scene title (wrapped if necessary)
        max_title_w = (card_x1 - card_x0) - 100
        title_lines = _wrap_text(title, title_font, max_title_w, draw)
        curr_y = card_y0 + 75
        for line in title_lines[:2]:  # max 2 lines for title
            draw.text((card_x0 + 50, curr_y), line, fill=(248, 250, 252), font=title_font)
            curr_y += 50

        # Divider line
        curr_y += 15
        draw.line(
            [(card_x0 + 50, curr_y), (card_x1 - 50, curr_y)],
            fill=(71, 85, 105),
            width=1,
        )
        curr_y += 35

        # Bullets
        max_bullet_w = (card_x1 - card_x0) - 140
        for bullet in bullets[:5]:  # max 5 bullets per slide to avoid clutter
            lines = _wrap_text(bullet, bullet_font, max_bullet_w, draw)
            # Draw bullet icon
            draw.ellipse(
                [card_x0 + 50, curr_y + 8, card_x0 + 60, curr_y + 18],
                fill=(59, 130, 246),
            )
            for line_idx, line in enumerate(lines):
                draw.text(
                    (card_x0 + 75, curr_y),
                    line,
                    fill=(226, 232, 240),
                    font=bullet_font,
                )
                curr_y += 36
            curr_y += 12

        # Save image
        img.save(output_path, "PNG")
        logger.info("slide_generator.saved", path=output_path, scene=scene_index)
        return output_path

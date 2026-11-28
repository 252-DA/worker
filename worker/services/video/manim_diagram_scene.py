from __future__ import annotations

import json
import os
import textwrap

from manim import (
    DOWN,
    RIGHT,
    UP,
    Arrow,
    Create,
    FadeIn,
    ManimColor,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
)

BACKGROUND = ManimColor("#0f172a")
CARD = ManimColor("#1e293b")
BORDER = ManimColor("#3b82f6")
TEXT = ManimColor("#e2e8f0")
ACCENT = ManimColor("#60a5fa")


def _label(
    value: str,
    *,
    font_size: float,
    wrap_at: int,
    max_width: float,
    max_height: float,
    color: ManimColor = TEXT,
) -> Text:
    normalized = " ".join(value.split())
    wrapped = "\n".join(textwrap.wrap(normalized, width=wrap_at, break_on_hyphens=False))
    label = Text(
        wrapped,
        font="DejaVu Sans",
        font_size=font_size,
        color=color,
        line_spacing=0.9,
    )
    if label.width > max_width:
        label.scale_to_fit_width(max_width)
    if label.height > max_height:
        label.scale_to_fit_height(max_height)
    return label


def _node(value: str, *, width: float, height: float, wrap_at: int) -> VGroup:
    box = RoundedRectangle(
        corner_radius=0.12,
        width=width,
        height=height,
        fill_color=CARD,
        fill_opacity=1,
        stroke_color=BORDER,
        stroke_width=2.5,
    )
    label = _label(
        value,
        font_size=26,
        wrap_at=wrap_at,
        max_width=width - 0.35,
        max_height=height - 0.16,
    )
    label.move_to(box.get_center())
    return VGroup(box, label)


class MicroContentDiagram(Scene):
    def construct(self) -> None:
        spec = json.loads(os.environ["DA_MANIM_SPEC"])
        items = list(spec["items"])
        duration = max(2.0, float(spec["duration_seconds"]))
        self.camera.background_color = BACKGROUND

        title = _label(
            spec["title"],
            font_size=42,
            wrap_at=64,
            max_width=12.5,
            max_height=0.9,
        )
        title.move_to((0, 3.15, 0))
        intro = min(0.8, duration * 0.1)
        self.play(FadeIn(title, shift=UP * 0.2), run_time=intro)
        elapsed = intro

        diagram = spec["diagram"]
        if diagram == "flow":
            elapsed += self._flow(items, duration)
        elif diagram == "array":
            elapsed += self._array(items, duration)
        elif diagram == "stack":
            elapsed += self._stack(items, duration)
        else:
            raise ValueError(f"Unsupported diagram: {diagram}")
        self.wait(max(0.1, duration - elapsed))

    def _flow(self, items: list[str], duration: float) -> float:
        count = len(items)
        height = min(0.85, (5.2 - (count - 1) * 0.25) / count)
        nodes = VGroup(*[_node(item, width=10.5, height=height, wrap_at=64) for item in items])
        nodes.arrange(DOWN, buff=0.25).move_to((0, -0.35, 0))
        arrows = [
            Arrow(
                nodes[index].get_bottom(),
                nodes[index + 1].get_top(),
                buff=0.03,
                stroke_width=3,
                color=ACCENT,
                max_tip_length_to_length_ratio=0.5,
            )
            for index in range(count - 1)
        ]
        unit = duration * 0.78 / (count + (count - 1) * 0.5)
        elapsed = 0.0
        for index, node in enumerate(nodes):
            self.play(FadeIn(node, shift=UP * 0.15), run_time=unit)
            elapsed += unit
            if index < len(arrows):
                arrow_time = unit * 0.5
                self.play(Create(arrows[index]), run_time=arrow_time)
                elapsed += arrow_time
        return elapsed

    def _array(self, items: list[str], duration: float) -> float:
        count = len(items)
        width = min(1.9, 11.8 / count - 0.18)
        cells = VGroup()
        for index, item in enumerate(items):
            node = _node(item, width=width, height=1.65, wrap_at=16)
            index_label = _label(
                str(index),
                font_size=20,
                wrap_at=4,
                max_width=width,
                max_height=0.35,
                color=ACCENT,
            )
            index_label.next_to(node, UP, buff=0.14)
            cells.add(VGroup(node, index_label))
        cells.arrange(RIGHT, buff=0.18).move_to((0, -0.35, 0))
        unit = duration * 0.78 / count
        for cell in cells:
            self.play(FadeIn(cell, shift=UP * 0.25), run_time=unit)
        return duration * 0.78

    def _stack(self, items: list[str], duration: float) -> float:
        count = len(items)
        step = min(0.95, 4.8 / count)
        height = step * 0.78
        start_y = -((count - 1) * step) / 2 - 0.35
        nodes = [
            _node(item, width=7.5, height=height, wrap_at=52).move_to((0, start_y + index * step, 0))
            for index, item in enumerate(items)
        ]
        unit = duration * 0.78 / count
        for node in nodes:
            self.play(FadeIn(node, shift=UP * 0.35), run_time=unit)
        return duration * 0.78

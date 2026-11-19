from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, cast

from document_chunk.shared.logger import get_logger

logger = get_logger(__name__)

DiagramType = Literal["flow", "array", "stack"]
DIAGRAM_TYPES = ("flow", "array", "stack")


@dataclass(frozen=True)
class ManimVisual:
    diagram: DiagramType
    items: tuple[str, ...]


def parse_manim_visual(visual: Mapping[str, Any]) -> ManimVisual | None:
    if visual.get("type") != "manim":
        return None
    diagram = visual.get("diagram")
    if diagram not in DIAGRAM_TYPES:
        raise ValueError("visual.diagram phải là flow, array hoặc stack")
    raw_items = visual.get("items")
    if not isinstance(raw_items, list) or not all(isinstance(item, str) for item in raw_items):
        raise ValueError("visual.items của cảnh Manim phải là danh sách chuỗi")
    items = tuple(" ".join(item.split()) for item in raw_items)
    if not 2 <= len(items) <= 6 or any(not item for item in items):
        raise ValueError("Cảnh Manim cần từ 2 đến 6 nhãn không rỗng")
    if any(len(item) > 80 for item in items):
        raise ValueError("Mỗi nhãn Manim không được dài quá 80 ký tự")
    return ManimVisual(diagram=cast(DiagramType, diagram), items=items)


class ManimGenerator:
    async def generate(
        self,
        *,
        visual: ManimVisual,
        title: str,
        duration_seconds: float,
        output_path: str,
    ) -> str:
        if importlib.util.find_spec("manim") is None:
            raise RuntimeError("Thiếu Manim; cài worker với extra 'video'")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        media_dir = output.parent / f"{output.stem}_media"
        media_dir.mkdir(parents=True, exist_ok=True)
        scene_path = Path(__file__).with_name("manim_diagram_scene.py")
        spec = {
            "title": title,
            "diagram": visual.diagram,
            "items": list(visual.items),
            "duration_seconds": max(2.0, float(duration_seconds)),
        }
        env = os.environ.copy()
        env["DA_MANIM_SPEC"] = json.dumps(spec, ensure_ascii=False)
        cmd = [
            sys.executable,
            "-m",
            "manim",
            "render",
            "--renderer",
            "cairo",
            "--format",
            "mp4",
            "--resolution",
            "1920,1080",
            "--fps",
            "25",
            "--disable_caching",
            "--silent",
            "--progress_bar",
            "none",
            "--media_dir",
            str(media_dir),
            "--output_file",
            output.stem,
            str(scene_path),
            "MicroContentDiagram",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            details = stderr.decode("utf-8", errors="replace")
            if not details.strip():
                details = stdout.decode("utf-8", errors="replace")
            logger.error("manim_generator.render_failed", error=details[-2000:])
            raise RuntimeError(f"Manim render failed: {details[-1000:]}")

        candidates = sorted(media_dir.rglob(f"{output.stem}.mp4"))
        if not candidates:
            raise RuntimeError("Manim kết thúc nhưng không tạo được file MP4")
        shutil.copyfile(candidates[-1], output)
        logger.info(
            "manim_generator.saved",
            output_path=output_path,
            diagram=visual.diagram,
            duration=duration_seconds,
        )
        return output_path

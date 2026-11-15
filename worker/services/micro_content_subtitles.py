"""Tiện ích dựng phụ đề WebVTT và timeline chuyển cảnh cho video micro-content."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class SubtitleCue:
    start_ms: int
    end_ms: int
    text: str


def format_vtt_timestamp(ms: int) -> str:
    ms = max(0, int(ms))
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def normalize_for_tts(text: str) -> str:
    """Chuẩn hoá văn bản đọc cho TTS: bỏ Markdown/URL và mở rộng ký hiệu viết tắt phổ biến."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"[*_~#>]+", " ", cleaned)
    cleaned = re.sub(r"\[\d+(?:,\s*\d+)*\]", " ", cleaned)
    cleaned = re.sub(r"\b(?:VD|vd|Vd)\s*[:.]", "Ví dụ:", cleaned)
    cleaned = re.sub(r"\bv\.v\.", "vân vân", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(\d+)\s*%", r"\1 phần trăm", cleaned)
    cleaned = cleaned.replace("&", " và ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[.!?…])\s+", cleaned) if p.strip()]
    return parts or [cleaned]


def build_proportional_cues(
    narration: str,
    speech_duration_ms: int,
) -> list[SubtitleCue]:
    """Chia lời đọc thành các câu có thời lượng tỷ lệ số ký tự (dự phòng khi TTS không trả boundary)."""
    sentences = _split_sentences(narration)
    if not sentences:
        return []
    total_ms = max(400, int(speech_duration_ms))
    weights = [max(1, len(sentence)) for sentence in sentences]
    total_weight = sum(weights)
    cues: list[SubtitleCue] = []
    cursor = 0
    for idx, (sentence, weight) in enumerate(zip(sentences, weights)):
        if idx == len(sentences) - 1:
            end_ms = total_ms
        else:
            share = max(200, round(total_ms * weight / total_weight))
            end_ms = min(total_ms, cursor + share)
        if end_ms <= cursor:
            end_ms = min(total_ms, cursor + 200)
        cues.append(SubtitleCue(start_ms=cursor, end_ms=end_ms, text=sentence))
        cursor = end_ms
        if cursor >= total_ms and idx < len(sentences) - 1:
            break
    return cues


def build_webvtt_and_timeline(
    scenes: Sequence[dict[str, Any]],
    scene_durations_ms: Sequence[int],
    scene_cues: Sequence[Sequence[SubtitleCue] | None] | None = None,
    speech_durations_ms: Sequence[int] | None = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    """
    Tổng hợp WebVTT, transcript đầy đủ và mảng ``scene_timeline`` theo mốc ffprobe thực tế.
    """
    timeline: list[dict[str, Any]] = []
    vtt_blocks: list[str] = ["WEBVTT", ""]
    transcript_lines: list[str] = []
    cursor_ms = 0
    last_cue_end_ms = 0

    for idx, (scene, raw_dur_ms) in enumerate(zip(scenes, scene_durations_ms)):
        clip_ms = max(400, int(raw_dur_ms))
        start_ms = cursor_ms
        end_ms = start_ms + clip_ms
        cursor_ms = end_ms

        title = str(scene.get("title") or f"Cảnh {idx + 1}").strip()
        narration = str(scene.get("narration") or "").strip()
        raw_page = scene.get("slide_page")
        slide_page = int(raw_page) if isinstance(raw_page, int) and raw_page >= 1 else None

        entry: dict[str, Any] = {
            "scene_index": idx,
            "title": title,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": clip_ms,
        }
        if slide_page is not None:
            entry["slide_page"] = slide_page
        timeline.append(entry)

        if narration:
            prefix = f"[Cảnh {idx + 1}] {title}: " if title else f"[Cảnh {idx + 1}] "
            transcript_lines.append(f"{prefix}{narration}")
        elif title:
            transcript_lines.append(f"[Cảnh {idx + 1}] {title}")

        local_cues = list(scene_cues[idx]) if scene_cues and idx < len(scene_cues) and scene_cues[idx] else []
        if not local_cues and narration:
            speech_ms = (
                int(speech_durations_ms[idx])
                if speech_durations_ms and idx < len(speech_durations_ms)
                else max(400, clip_ms - 400)
            )
            local_cues = build_proportional_cues(narration, min(clip_ms, max(400, speech_ms)))

        for cue in local_cues:
            cue_text = cue.text.strip()
            if not cue_text:
                continue
            raw_start = start_ms + max(0, min(cue.start_ms, clip_ms - 50))
            abs_start = min(max(last_cue_end_ms, raw_start), max(start_ms, end_ms - 50))
            abs_end = min(end_ms, max(abs_start + 50, start_ms + min(cue.end_ms, clip_ms)))
            last_cue_end_ms = abs_end
            vtt_blocks.append(
                f"{format_vtt_timestamp(abs_start)} --> {format_vtt_timestamp(abs_end)}\n{cue_text}\n"
            )

    vtt_text = "\n".join(vtt_blocks).rstrip() + "\n"
    transcript_text = "\n\n".join(transcript_lines)
    return vtt_text, transcript_text, timeline

from __future__ import annotations

import math
import os
import struct
import wave
from abc import ABC, abstractmethod

from document_chunk.shared.logger import get_logger

logger = get_logger(__name__)


class TtsClient(ABC):
    @abstractmethod
    async def synthesize(self, text: str, output_path: str, voice: str = "vi-VN-HoaiMyNeural") -> float:
        """Sinh file âm thanh từ văn bản và trả về thời lượng (giây)."""
        pass


class MockTtsClient(TtsClient):
    """
    Sinh file WAV giả lập (tạo sóng âm thanh nhẹ nhàng) để chạy offline hoặc test mà không cần mạng.
    Tốc độ đọc trung bình tiếng Việt khoảng ~3 từ/giây (hoặc 12 ký tự/giây).
    """

    async def synthesize(self, text: str, output_path: str, voice: str = "vi-VN-HoaiMyNeural") -> float:
        words = text.strip().split()
        duration_seconds = max(2.0, len(words) * 0.38)
        sample_rate = 24000
        num_samples = int(duration_seconds * sample_rate)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)

            # Tạo âm thanh nền nhẹ (440Hz sine wave biên độ nhỏ)
            samples = bytearray()
            for i in range(num_samples):
                val = int(800 * math.sin(2 * math.pi * 440 * (i / sample_rate)))
                samples.extend(struct.pack("<h", val))
            wav_file.writeframes(samples)

        logger.info(
            "mock_tts.synthesized",
            path=output_path,
            duration=duration_seconds,
            words=len(words),
        )
        return duration_seconds


class EdgeTtsClient(TtsClient):
    """Sử dụng edge-tts miễn phí với các giọng đọc tiếng Việt chất lượng cao (Hoài My / Nam Minh)."""

    def __init__(self, fallback: TtsClient | None = None) -> None:
        self._fallback = fallback or MockTtsClient()

    async def synthesize(self, text: str, output_path: str, voice: str = "vi-VN-HoaiMyNeural") -> float:
        try:
            import edge_tts  # type: ignore

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)

            # Ước lượng hoặc đo thời lượng file
            words = text.strip().split()
            estimated_duration = max(2.0, len(words) * 0.38)
            logger.info("edge_tts.synthesized", path=output_path, voice=voice)
            return estimated_duration
        except Exception as exc:
            logger.warning("edge_tts.failed_fallback_to_mock", error=str(exc))
            return await self._fallback.synthesize(text, output_path, voice)


def get_default_tts_client() -> TtsClient:
    return EdgeTtsClient()

"""
Transcript Exporter supporting .txt, .srt, .vtt, and .json formats with timecodes.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Union
import json


@dataclass
class TranscriptSegment:
    start_time: float  # in seconds
    end_time: float    # in seconds
    text: str


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds into SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds into WebVTT timestamp: HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class TranscriptExporter:
    """
    Exports transcription results and timestamped segments to multiple formats.
    """

    @staticmethod
    def export_txt(full_text: str, file_path: Union[str, Path]) -> None:
        """Export plain text file."""
        Path(file_path).write_text(full_text.strip(), encoding="utf-8")

    @staticmethod
    def export_srt(segments: List[TranscriptSegment], file_path: Union[str, Path]) -> None:
        """Export SubRip (.srt) subtitle file."""
        lines = []
        for idx, seg in enumerate(segments, start=1):
            if not seg.text.strip():
                continue
            start_str = format_timestamp_srt(seg.start_time)
            end_str = format_timestamp_srt(seg.end_time)
            lines.append(f"{idx}\n{start_str} --> {end_str}\n{seg.text.strip()}\n")

        Path(file_path).write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def export_vtt(segments: List[TranscriptSegment], file_path: Union[str, Path]) -> None:
        """Export WebVTT (.vtt) subtitle file."""
        lines = ["WEBVTT\n"]
        for idx, seg in enumerate(segments, start=1):
            if not seg.text.strip():
                continue
            start_str = format_timestamp_vtt(seg.start_time)
            end_str = format_timestamp_vtt(seg.end_time)
            lines.append(f"{idx}\n{start_str} --> {end_str}\n{seg.text.strip()}\n")

        Path(file_path).write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def export_json(segments: List[TranscriptSegment], file_path: Union[str, Path]) -> None:
        """Export structured JSON file with segments and timestamps."""
        data = {
            "segments": [asdict(seg) for seg in segments],
            "full_text": " ".join(seg.text for seg in segments).strip(),
        }
        Path(file_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

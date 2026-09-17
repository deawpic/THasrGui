"""
Verification Tests for Text Stitcher, Exporters, and Batch File Transcription.
"""

import tempfile
from pathlib import Path
import json
import numpy as np
import soundfile as sf
import pytest

from typhoon_transcriber.config import SAMPLE_RATE
from typhoon_transcriber.transcriber.text_stitcher import TextStitcher
from typhoon_transcriber.transcriber.exporter import (
    TranscriptExporter,
    TranscriptSegment,
    format_timestamp_srt,
    format_timestamp_vtt,
)
from typhoon_transcriber.transcriber.batch_worker import load_audio_file, BatchFileTranscriber
from typhoon_transcriber.models.onnx_engine import TyphoonONNXEngine


def test_text_stitcher_overlap_merging():
    """Verify stitcher finds suffix-prefix overlaps and merges cleanly."""
    stitcher = TextStitcher()

    # Chunk 1
    t1 = stitcher.append_chunk("สวัสดีครับคุณ")
    assert t1 == "สวัสดีครับคุณ"

    # Chunk 2 (overlaps "ครับคุณ")
    t2 = stitcher.append_chunk("ครับคุณผู้ฟังทุกท่าน")
    assert t2 == "สวัสดีครับคุณผู้ฟังทุกท่าน"

    # Chunk 3 (no overlap, space attached)
    t3 = stitcher.append_chunk("ยินดีต้อนรับ")
    assert "ยินดีต้อนรับ" in t3


def test_timestamp_formatting():
    """Verify SRT and WebVTT timestamp formatting."""
    sec = 3723.456  # 1 hour, 2 min, 3.456 sec
    assert format_timestamp_srt(sec) == "01:02:03,456"
    assert format_timestamp_vtt(sec) == "01:02:03.456"


def test_transcript_exporters():
    """Verify export to .txt, .srt, .vtt, and .json formats."""
    segments = [
        TranscriptSegment(start_time=0.0, end_time=2.5, text="สวัสดีครับ"),
        TranscriptSegment(start_time=2.5, end_time=5.0, text="นี่คือการทดสอบ"),
    ]
    full_text = "สวัสดีครับ นี่คือการทดสอบ"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # TXT
        txt_file = tmp_path / "out.txt"
        TranscriptExporter.export_txt(full_text, txt_file)
        assert txt_file.read_text(encoding="utf-8") == full_text

        # SRT
        srt_file = tmp_path / "out.srt"
        TranscriptExporter.export_srt(segments, srt_file)
        srt_content = srt_file.read_text(encoding="utf-8")
        assert "00:00:00,000 --> 00:00:02,500" in srt_content
        assert "สวัสดีครับ" in srt_content

        # VTT
        vtt_file = tmp_path / "out.vtt"
        TranscriptExporter.export_vtt(segments, vtt_file)
        vtt_content = vtt_file.read_text(encoding="utf-8")
        assert "WEBVTT" in vtt_content
        assert "00:00:00.000 --> 00:00:02.500" in vtt_content

        # JSON
        json_file = tmp_path / "out.json"
        TranscriptExporter.export_json(segments, json_file)
        json_data = json.loads(json_file.read_text(encoding="utf-8"))
        assert len(json_data["segments"]) == 2
        assert json_data["segments"][0]["text"] == "สวัสดีครับ"


def test_batch_file_transcription_pipeline():
    """Verify synthetic WAV file generation and batch transcription."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_file = Path(tmpdir) / "synthetic_speech.wav"

        # Generate 3 seconds synthetic audio
        duration = 3.0
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        audio_data = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sf.write(str(wav_file), audio_data, SAMPLE_RATE)

        # Load file
        loaded_audio = load_audio_file(wav_file, target_sr=SAMPLE_RATE)
        assert len(loaded_audio) == int(SAMPLE_RATE * duration)

        # Run Batch Transcriber with Mock Engine
        engine = TyphoonONNXEngine(use_mock=True)
        worker = BatchFileTranscriber(
            file_path=wav_file,
            engine=engine,
            segment_duration_sec=1.5,
        )

        # Execute synchronously for deterministic testing
        worker.run()

        assert len(worker.segments) > 0
        assert worker.segments[0].text != ""

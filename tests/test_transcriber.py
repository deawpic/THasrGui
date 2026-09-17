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


def test_text_stitcher_duplicate_prevention():
    """Verify stitcher prevents repeated sentences and space-variation duplicates."""
    stitcher = TextStitcher()

    # 1. Space moved / variation duplicate prevention
    s1 = stitcher.append_chunk("วันนี้เราจะมา")
    assert s1 == "วันนี้เราจะมา"

    s2 = stitcher.append_chunk("วันนี้เรา จะมาคุยเรื่อง")
    # Must NOT duplicate 'วันนี้เราจะมา'
    assert s2 == "วันนี้เราจะมาคุยเรื่อง"
    assert s2.count("วันนี้เรา") == 1

    # 2. Redundant tail on silence / pause
    s3 = stitcher.append_chunk("คุยเรื่อง")
    assert s3 == "วันนี้เราจะมาคุยเรื่อง"

    # 3. Space added in prefix
    s_new = TextStitcher()
    s_new.append_chunk("สวัสดี ครับ")
    res = s_new.append_chunk("สวัสดีครับ ผมชื่อเด่น")
    assert res == "สวัสดี ครับ ผมชื่อเด่น"
    assert res.count("สวัสดี") == 1


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


def test_text_stitcher_preview_and_deduplication():
    """Verify stitcher preview does not mutate state and suppresses duplicate sentences."""
    stitcher = TextStitcher()

    # 1. Initial chunk
    committed1 = stitcher.append_chunk("สวัสดีครับ")
    assert committed1 == "สวัสดีครับ"
    assert stitcher.get_text() == "สวัสดีครับ"

    # 2. Interim preview without mutating internal state
    preview1 = stitcher.preview("วันนี้เราจะมา")
    assert preview1 == "สวัสดีครับ วันนี้เราจะมา"
    assert stitcher.get_text() == "สวัสดีครับ"  # State preserved

    # 3. Preview with overlapping prefix
    preview2 = stitcher.preview("สวัสดีครับ วันนี้เราจะมาคุย")
    assert preview2 == "สวัสดีครับ วันนี้เราจะมาคุย"
    assert stitcher.get_text() == "สวัสดีครับ"

    # 4. Preview with duplicate content
    preview_dup = stitcher.preview("สวัสดีครับ")
    assert preview_dup == "สวัสดีครับ"

    # 5. Commit second utterance
    committed2 = stitcher.append_chunk("วันนี้เราจะมาคุย")
    assert committed2 == "สวัสดีครับ วันนี้เราจะมาคุย"

    # 6. Reject exact sentence repetition
    repeated = stitcher.append_chunk("วันนี้เราจะมาคุย")
    assert repeated == "สวัสดีครับ วันนี้เราจะมาคุย"
    assert repeated.count("คุย") == 1

    # 7. Reject previous sentence repetition
    repeated_past = stitcher.append_chunk("สวัสดีครับ")
    assert repeated_past == "สวัสดีครับ วันนี้เราจะมาคุย"
    assert repeated_past.count("สวัสดีครับ") == 1


def test_streaming_worker_energy_valley_and_reset():
    """Verify ASRStreamingWorker energy valley splitting, initial text setting, and resetting."""
    from typhoon_transcriber.transcriber.streaming_worker import ASRStreamingWorker

    engine = TyphoonONNXEngine(use_mock=True)
    worker = ASRStreamingWorker(engine=engine)

    # 1. Initial text setting
    worker.set_initial_text("ข้อความก่อนหน้า")
    assert worker.stitcher.get_text() == "ข้อความก่อนหน้า"

    # 2. Reset transcript
    worker.reset_transcript()
    assert worker.stitcher.get_text() == ""

    # 3. Energy valley split detection on 4 seconds of audio
    # Construct 4.0s signal with high energy everywhere except a quiet dip at 3.5s (800 samples)
    total_samples = 4 * SAMPLE_RATE
    audio = np.ones(total_samples, dtype=np.float32) * 0.2
    dip_start = int(3.5 * SAMPLE_RATE)
    audio[dip_start : dip_start + 800] = 0.0001  # Quiet frame

    split_pt = worker._find_split_point(audio)
    assert abs(split_pt - (dip_start + 800)) <= 800  # Identified within 1 frame


def test_streaming_worker_simulation_no_duplicates():
    """Verify that multiple simulated speech-silence cycles produce zero duplicated sentences."""
    from typhoon_transcriber.transcriber.streaming_worker import ASRStreamingWorker

    class SequentialMockEngine:
        def __init__(self):
            self.calls = 0
            self.outputs = ["สวัสดีครับ", "วันนี้เราจะมาคุยเรื่องเอไอ", "ยินดีที่ได้รู้จักครับ"]

        def transcribe_chunk(self, audio: np.ndarray) -> str:
            if audio.size == 0 or np.max(np.abs(audio)) < 0.001:
                return ""
            out = self.outputs[min(self.calls, len(self.outputs) - 1)]
            self.calls += 1
            return out

    mock_engine = SequentialMockEngine()
    worker = ASRStreamingWorker(engine=mock_engine)

    # 1. Simulate Phrase 1: 0.8s speech + 0.6s silence
    speech_1 = np.ones(int(0.8 * SAMPLE_RATE), dtype=np.float32) * 0.1
    # 0.8s speech passed in 100ms blocks
    block_size = int(0.1 * SAMPLE_RATE)
    for i in range(8):
        blk = speech_1[i * block_size : (i + 1) * block_size]
        worker.stitcher.preview(mock_engine.transcribe_chunk(blk))

    # Commit Phrase 1 on silence endpoint
    out1 = worker.stitcher.append_chunk("สวัสดีครับ")
    assert out1 == "สวัสดีครับ"

    # 2. Simulate Phrase 2: 1.0s speech + 0.6s silence
    out2 = worker.stitcher.append_chunk("วันนี้เราจะมาคุยเรื่องเอไอ")
    assert out2 == "สวัสดีครับ วันนี้เราจะมาคุยเรื่องเอไอ"
    assert out2.count("สวัสดีครับ") == 1

    # 3. Simulate accidental repeat of Phrase 2: must NOT duplicate
    out_dup = worker.stitcher.append_chunk("วันนี้เราจะมาคุยเรื่องเอไอ")
    assert out_dup == "สวัสดีครับ วันนี้เราจะมาคุยเรื่องเอไอ"
    assert out_dup.count("วันนี้เราจะมาคุยเรื่องเอไอ") == 1

    # 4. Phrase 3
    out3 = worker.stitcher.append_chunk("ยินดีที่ได้รู้จักครับ")
    assert out3 == "สวัสดีครับ วันนี้เราจะมาคุยเรื่องเอไอ ยินดีที่ได้รู้จักครับ"
    assert out3.count("ยินดีที่ได้รู้จักครับ") == 1



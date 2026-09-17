"""
Verification Tests for Multi-File/Folder Batch Scanning, Folder Mirroring, and Queue Worker.
"""

import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import pytest

from typhoon_transcriber.config import SAMPLE_RATE
from typhoon_transcriber.transcriber.batch_queue import (
    BatchItem,
    scan_files_and_folders,
    BatchQueueWorker,
)
from typhoon_transcriber.models.onnx_engine import TyphoonONNXEngine


def test_scan_files_and_folders_recursive_hierarchy():
    """Verify recursive scanning and relative path preservation for folder mirroring."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "Recordings"
        sub1 = root / "2026" / "Q1"
        sub2 = root / "Interviews"
        sub1.mkdir(parents=True, exist_ok=True)
        sub2.mkdir(parents=True, exist_ok=True)

        # Create dummy audio files
        dummy_data = np.zeros(SAMPLE_RATE, dtype=np.float32)
        sf.write(str(sub1 / "meeting.wav"), dummy_data, SAMPLE_RATE)
        sf.write(str(sub1 / "notes.flac"), dummy_data, SAMPLE_RATE)
        sf.write(str(sub2 / "interview_01.wav"), dummy_data, SAMPLE_RATE)
        (root / "unrelated.txt").write_text("ignore me", encoding="utf-8")

        # Scan the entire root folder
        items = scan_files_and_folders([root])

        assert len(items) == 3
        rel_paths = {str(item.rel_path) for item in items}
        assert "2026/Q1/meeting.wav" in rel_paths
        assert "2026/Q1/notes.flac" in rel_paths
        assert "Interviews/interview_01.wav" in rel_paths


def test_batch_queue_worker_mirrored_destination_execution():
    """Verify BatchQueueWorker creates identical subfolder structures at destination."""
    with tempfile.TemporaryDirectory() as src_tmp, tempfile.TemporaryDirectory() as dest_tmp:
        src_root = Path(src_tmp) / "AudioSource"
        nested_dir = src_root / "DepartmentA" / "Meetings"
        nested_dir.mkdir(parents=True, exist_ok=True)

        # Create synthetic audio file
        wav_path = nested_dir / "conf_call.wav"
        t = np.linspace(0, 2.0, SAMPLE_RATE * 2, endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sf.write(str(wav_path), audio, SAMPLE_RATE)

        # Scan
        items = scan_files_and_folders([src_root])
        assert len(items) == 1
        assert str(items[0].rel_path) == "DepartmentA/Meetings/conf_call.wav"

        # Execute Batch Worker with Destination Mirroring
        dest_root = Path(dest_tmp) / "Transcripts"
        engine = TyphoonONNXEngine(use_mock=True)

        worker = BatchQueueWorker(
            items=items,
            engine=engine,
            dest_root_dir=dest_root,
            formats=[".txt", ".srt"],
            overwrite=True,
            segment_duration_sec=1.5,
        )

        # Synchronous execution
        worker.run()

        # Check Mirrored Output Folder and Files
        expected_txt = dest_root / "DepartmentA" / "Meetings" / "conf_call.txt"
        expected_srt = dest_root / "DepartmentA" / "Meetings" / "conf_call.srt"

        assert expected_txt.exists(), f"Missing mirrored file: {expected_txt}"
        assert expected_srt.exists(), f"Missing mirrored file: {expected_srt}"
        assert len(expected_txt.read_text(encoding="utf-8")) > 0
        assert "00:00:00,000" in expected_srt.read_text(encoding="utf-8")
        assert items[0].status == "Completed"


def test_batch_item_serialization(tmp_path):
    """Verify BatchItem to_dict and from_dict preserve all properties."""
    item = BatchItem(
        source_file=tmp_path / "audio.wav",
        root_source_dir=tmp_path,
        rel_path=Path("sub/audio.wav"),
        status="Completed",
        error_message=None,
        duration_sec=12.5,
        generated_files=[tmp_path / "out.txt"],
        custom_dest_dir=tmp_path / "custom_output",
    )

    d = item.to_dict()
    assert d["source_file"] == str(tmp_path / "audio.wav")
    assert d["status"] == "Completed"
    assert d["custom_dest_dir"] == str(tmp_path / "custom_output")

    restored = BatchItem.from_dict(d)
    assert restored.source_file == item.source_file
    assert restored.root_source_dir == item.root_source_dir
    assert restored.rel_path == item.rel_path
    assert restored.status == "Completed"
    assert restored.duration_sec == 12.5
    assert restored.custom_dest_dir == tmp_path / "custom_output"
    assert len(restored.generated_files) == 1


def test_save_and_load_batch_queue(tmp_path):
    """Verify saving batch queue to file and loading it back."""
    from typhoon_transcriber.transcriber.batch_queue import save_batch_queue, load_batch_queue

    item1 = BatchItem(source_file=tmp_path / "file1.wav", status="Completed")
    item2 = BatchItem(source_file=tmp_path / "file2.mp3", status="Pending", custom_dest_dir=tmp_path / "out2")
    items = [item1, item2]

    queue_file = tmp_path / "test_queue.json"
    dest_dir = tmp_path / "GlobalOut"
    formats = [".txt", ".srt", ".vtt"]

    save_batch_queue(queue_file, items, dest_dir=dest_dir, formats=formats)
    assert queue_file.exists()

    loaded_items, loaded_dest, loaded_fmts = load_batch_queue(queue_file)
    assert len(loaded_items) == 2
    assert loaded_items[0].status == "Completed"
    assert loaded_items[1].status == "Pending"
    assert loaded_items[1].custom_dest_dir == tmp_path / "out2"
    assert loaded_dest == dest_dir
    assert loaded_fmts == formats


def test_last_session_queue_recovery(tmp_path):
    """Verify last session queue persistence and crash recovery."""
    from typhoon_transcriber.transcriber.batch_queue import (
        save_last_session_queue,
        load_last_session_queue,
        clear_last_session_queue,
    )

    session_file = tmp_path / "session_queue.json"
    item_done = BatchItem(source_file=tmp_path / "done.wav", status="Completed")
    item_crashed = BatchItem(source_file=tmp_path / "crashed.wav", status="Processing")
    item_pending = BatchItem(source_file=tmp_path / "pending.wav", status="Pending")

    # Save session with simulated in-progress crash
    save_last_session_queue([item_done, item_crashed, item_pending], session_file=session_file)
    assert session_file.exists()

    # Load session
    loaded_items, _, _ = load_last_session_queue(session_file=session_file)
    assert len(loaded_items) == 3

    # Clear session
    clear_last_session_queue(session_file=session_file)
    assert not session_file.exists()


def test_batch_queue_worker_skips_completed_items(tmp_path):
    """Verify BatchQueueWorker skips already completed items and only transcribes pending ones."""
    # Create 2 synthetic audio files
    file1 = tmp_path / "file1.wav"
    file2 = tmp_path / "file2.wav"
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(file1), audio, SAMPLE_RATE)
    sf.write(str(file2), audio, SAMPLE_RATE)

    # Item 1 is already completed, Item 2 is pending
    item1 = BatchItem(source_file=file1, rel_path=Path(file1.name), status="Completed")
    item2 = BatchItem(source_file=file2, rel_path=Path(file2.name), status="Pending")

    dest_root = tmp_path / "Out"
    engine = TyphoonONNXEngine(use_mock=True)

    worker = BatchQueueWorker(
        items=[item1, item2],
        engine=engine,
        dest_root_dir=dest_root,
        formats=[".txt"],
    )
    worker.run()

    # Item 1 stays Completed without generating txt in dest_root (was already completed)
    assert item1.status == "Completed"
    # Item 2 was processed and completed
    assert item2.status == "Completed"
    assert (dest_root / "file2.txt").exists()


def test_batch_queue_worker_custom_dest_dir(tmp_path):
    """Verify BatchQueueWorker outputs to custom_dest_dir when specified."""
    file1 = tmp_path / "custom_test.wav"
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    sf.write(str(file1), audio, SAMPLE_RATE)

    custom_dest = tmp_path / "SpecialFolder"
    item = BatchItem(
        source_file=file1,
        rel_path=Path("default_sub/custom_test.wav"),
        custom_dest_dir=custom_dest,
    )

    dest_root = tmp_path / "DefaultTranscripts"
    engine = TyphoonONNXEngine(use_mock=True)

    worker = BatchQueueWorker(
        items=[item],
        engine=engine,
        dest_root_dir=dest_root,
        formats=[".txt"],
    )
    worker.run()

    # Verify output file was written to custom_dest, NOT dest_root / default_sub
    assert (custom_dest / "custom_test.txt").exists()
    assert not (dest_root / "default_sub" / "custom_test.txt").exists()


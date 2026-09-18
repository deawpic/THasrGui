"""
Audio device discovery, query, and classification (Mic vs Loopback/Monitor).
"""

from dataclasses import dataclass
from typing import List, Optional
import sys
import logging
import sounddevice as sd

logger = logging.getLogger(__name__)

WASAPI_LOOPBACK_OFFSET = 10000


@dataclass
class AudioDevice:
    index: int
    name: str
    hostapi: int
    max_input_channels: int
    default_samplerate: float
    is_loopback: bool
    is_default: bool
    is_wasapi_loopback: bool = False
    raw_device_index: Optional[int] = None


class AudioDeviceManager:
    """
    Manages dynamic discovery and categorization of audio input hardware.
    Supports standard microphone inputs, stereo mix, monitor sources,
    and Windows Native WASAPI Loopback endpoints.
    Never assumes hardcoded device IDs.
    """

    def __init__(self):
        self._devices: List[AudioDevice] = []

    def refresh_devices(self) -> List[AudioDevice]:
        """
        Query system host APIs and audio endpoints dynamically.
        1. Queries standard hardware inputs via sounddevice.
        2. On Windows, queries native WASAPI loopback playback devices via pyaudiowpatch.
        """
        self._devices.clear()
        default_input_index = None
        try:
            device_list = sd.query_devices()
            default_input_index = sd.default.device[0]
        except Exception as exc:
            logger.error("Failed to query audio devices via sounddevice: %s", exc)
            device_list = []

        for idx, dev in enumerate(device_list):
            max_in = dev.get("max_input_channels", 0)
            if max_in <= 0:
                continue  # Skip output-only devices

            name = dev.get("name", f"Device {idx}")
            name_lower = name.lower()

            # Detect loopback / monitor endpoints
            is_loopback = False
            if "monitor" in name_lower or "stereo mix" in name_lower or "loopback" in name_lower:
                is_loopback = True

            audio_dev = AudioDevice(
                index=idx,
                name=name,
                hostapi=dev.get("hostapi", 0),
                max_input_channels=max_in,
                default_samplerate=dev.get("default_samplerate", 16000.0),
                is_loopback=is_loopback,
                is_default=(idx == default_input_index),
                is_wasapi_loopback=False,
                raw_device_index=idx,
            )
            self._devices.append(audio_dev)

        # On Windows, discover Native WASAPI Loopback endpoints
        if sys.platform == "win32":
            try:
                import pyaudiowpatch as pyaudio

                p = pyaudio.PyAudio()
                try:
                    default_loopback_idx = None
                    try:
                        def_loop = p.get_default_wasapi_loopback()
                        if def_loop:
                            default_loopback_idx = def_loop.get("index")
                    except Exception as loop_exc:
                        logger.debug("No default WASAPI loopback device found: %s", loop_exc)

                    for loop_dev in p.get_loopback_device_info_generator():
                        raw_idx = loop_dev.get("index", 0)
                        raw_name = loop_dev.get("name", f"Output Loopback {raw_idx}")
                        clean_name = raw_name.replace(" [Loopback]", "").replace("[Loopback]", "").strip()

                        loopback_audio_dev = AudioDevice(
                            index=WASAPI_LOOPBACK_OFFSET + raw_idx,
                            name=clean_name,
                            hostapi=loop_dev.get("hostApi", 0),
                            max_input_channels=loop_dev.get("maxInputChannels", 2),
                            default_samplerate=loop_dev.get("defaultSampleRate", 48000.0),
                            is_loopback=True,
                            is_default=False,  # Keep microphone as default unless chosen
                            is_wasapi_loopback=True,
                            raw_device_index=raw_idx,
                        )
                        self._devices.append(loopback_audio_dev)
                finally:
                    p.terminate()
            except Exception as exc:
                logger.debug("WASAPI loopback discovery not available or failed: %s", exc)

        logger.info("Discovered %d audio devices.", len(self._devices))
        return self._devices

    def get_devices(self) -> List[AudioDevice]:
        """Return cached or refreshed device list."""
        if not self._devices:
            return self.refresh_devices()
        return self._devices

    def get_default_device(self) -> Optional[AudioDevice]:
        """Get the system default input device."""
        devices = self.get_devices()
        for dev in devices:
            if dev.is_default:
                return dev
        for dev in devices:
            if not dev.is_loopback:
                return dev
        return devices[0] if devices else None

    def get_device_by_index(self, index: Optional[int]) -> Optional[AudioDevice]:
        """Find device by its unique index."""
        if index is None:
            return None
        for dev in self.get_devices():
            if dev.index == index:
                return dev
        return None

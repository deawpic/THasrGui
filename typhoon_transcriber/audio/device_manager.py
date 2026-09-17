"""
Audio device discovery, query, and classification (Mic vs Loopback/Monitor).
"""

from dataclasses import dataclass
from typing import List, Optional
import sounddevice as sd
import logging

logger = logging.getLogger(__name__)


@dataclass
class AudioDevice:
    index: int
    name: str
    hostapi: int
    max_input_channels: int
    default_samplerate: float
    is_loopback: bool
    is_default: bool


class AudioDeviceManager:
    """
    Manages dynamic discovery and categorization of audio input hardware.
    Never assumes hardcoded device IDs.
    """

    def __init__(self):
        self._devices: List[AudioDevice] = []

    def refresh_devices(self) -> List[AudioDevice]:
        """
        Query system host APIs and audio endpoints dynamically using sounddevice.
        Categorizes devices into regular Microphone vs Desktop Loopback/Monitor.
        """
        self._devices.clear()
        try:
            device_list = sd.query_devices()
            default_input_index = sd.default.device[0]
        except Exception as exc:
            logger.error("Failed to query audio devices via sounddevice: %s", exc)
            return []

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
            )
            self._devices.append(audio_dev)

        logger.info("Discovered %d audio input devices.", len(self._devices))
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
        return devices[0] if devices else None

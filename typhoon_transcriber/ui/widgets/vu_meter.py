"""
Custom PySide6 Audio VU Level Meter Widget.
"""

from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QLinearGradient
from PySide6.QtCore import Qt, Slot


class VUMeterWidget(QWidget):
    """
    Smooth visual level meter indicating real-time microphone / internal audio input energy.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._level = 0.0  # 0.0 to 1.0
        self.setMinimumWidth(80)
        self.setMaximumWidth(120)
        self.setFixedHeight(18)
        self.setToolTip("Audio Input Level (VU Meter)")

    @Slot(float)
    def set_level(self, level: float) -> None:
        """Update audio level (0.0 to 1.0) and trigger repaint."""
        clamped = max(0.0, min(float(level), 1.0))
        # Smooth decay
        if clamped >= self._level:
            self._level = clamped
        else:
            self._level = self._level * 0.7 + clamped * 0.3
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        w = rect.width()
        h = rect.height()

        # Draw dark background track
        painter.setBrush(QColor("#181825"))
        painter.setPen(QColor("#313244"))
        painter.drawRoundedRect(0, 0, w, h, 4, 4)

        # Draw level fill
        fill_width = int(w * self._level)
        if fill_width > 2:
            gradient = QLinearGradient(0, 0, w, 0)
            gradient.setColorAt(0.0, QColor("#a6e3a1"))  # Green
            gradient.setColorAt(0.7, QColor("#f9e2af"))  # Yellow
            gradient.setColorAt(1.0, QColor("#f38ba8"))  # Red

            painter.setBrush(gradient)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(1, 1, fill_width - 2, h - 2, 3, 3)

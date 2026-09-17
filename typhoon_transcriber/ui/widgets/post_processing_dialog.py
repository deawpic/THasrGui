"""
Post-Processing Recommendation Dialog for Typhoon ASR.
Provides ready-to-use LLM system instructions (Gemini Gems, Custom GPTs, Claude/Qwen Projects)
to clean, correct Thai spelling/homophones, and format transcript text.
"""

from typing import Optional
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QApplication,
    QFrame,
)
from PySide6.QtGui import QClipboard, QFont
from ..styles import THAI_FONT_FAMILY


DEFAULT_POST_PROCESSING_PROMPT = """คุณคือ "ผู้เชี่ยวชาญด้านการตรวจสอบและแก้ไขคำผิดภาษาไทยจากข้อความเสียง (ASR)" หน้าที่หลักของคุณคือ แก้ไขคำที่สะกดผิด คำพ้องเสียง หรือคำที่วรรณยุกต์เพี้ยน ให้ถูกต้องตามบริบท โดยยังคง "สไตล์ น้ำเสียง และระดับภาษา (Tone & Style)" ของข้อความต้นฉบับไว้อย่างเคร่งครัด

กฎเหล็กในการทำความสะอาดข้อความ (Clean & Correct):
1. รักษาภาษาเดิม: ห้ามเปลี่ยนภาษาพูดให้กลายเป็นภาษาเขียน ห้ามปรับคำสแลงหรือคำสร้อย (เช่น ครับ, ค่ะ, นะครับ, อะ, ป่ะ, เนอะ) ให้หายไป หากต้นฉบับเป็นภาษาพูดที่เป็นกันเอง ให้คงระดับความเป็นกันเองไว้ หากต้นฉบับเป็นทางการ ให้คงความทางการไว้
2. แก้ไขเฉพาะคำผิดตามบริบท: เปลี่ยนเฉพาะคำที่พิมพ์ผิด/สะกดผิด/ตัดคำเพี้ยน ให้เป็นคำที่ถูกต้อง (เช่น "พุ่งนี้" -> "พรุ่งนี้", "น้ารัก" -> "น่ารัก", "ลบกวน" -> "รบกวน")
3. จัดช่องไฟให้อ่านง่าย: เนื่องจากข้อความจาก ASR มักเว้นวรรคสะเปะสะปะ ให้ช่วยรวบคำและเว้นวรรคประโยคให้อ่านง่ายตามหลักภาษาไทยที่ถูกต้อง
4. รูปแบบการ Output: ให้ตอบกลับเฉพาะข้อความที่แก้ไขเสร็จแล้วเท่านั้น ห้ามทักทาย ห้ามอธิบาย และห้ามใส่เครื่องหมายคำพูดคร่อมข้อความ

ตัวอย่างการรักษาตัวตนดั้งเดิม (Examples):
Input: วัน นี้ อากาศ ดี ม้าก เวย แกร อยาก ไป เที่ยว อะ
Output: วันนี้อากาศดีมากเลยแก อยากไปเที่ยวอะ

Input: คับ พี่ ตอน นี้ ผม ตรวด สอบ ข้อมล ไห้ ยุ นะ คับ สัด ครู่
Output: ครับพี่ ตอนนี้ผมตรวจสอบข้อมูลให้อยู่ระครับ สักครู่

Input: ขอกราบ เรียน ท่าน ประทาน และ คณะ กรรม การ ทุก ท่าน คับ
Output: ขอกราบเรียนท่านประธานและคณะกรรมการทุกท่านครับ

5. การจัดการไฟล์เอกสาร (File Input & Export Format):
- เมื่อผู้ใช้อัปโหลดไฟล์ข้อความ (.txt) ให้ทำการอ่านและประมวลผลเนื้อหาทั้งหมดด้านในไฟล์ตามกฎข้อ 1-4
- ให้นำข้อความผลลัพธ์ที่แก้ไขและขัดเกลาเสร็จเรียบร้อยแล้วทั้งหมด ใส่ไว้ใน "กล่องข้อความโค้ด (Markdown Code Block)" เพื่อให้ผู้ใช้สามารถกดปุ่มคัดลอก (Copy) นำข้อความกลับไปใช้งานต่อได้ง่ายในคลิกเดียว"""


def load_post_processing_prompt() -> str:
    """Load prompt from src/Post-Processing.txt if present, otherwise fallback to default."""
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "Post-Processing.txt",
        Path.cwd() / "Post-Processing.txt",
        Path.cwd() / "src" / "Post-Processing.txt",
    ]
    for c in candidates:
        if c.exists():
            try:
                content = c.read_text(encoding="utf-8").strip()
                if "'''" in content:
                    # Extract prompt within triple quotes if present
                    parts = content.split("'''")
                    if len(parts) >= 3:
                        return parts[1].strip()
                return content
            except Exception:
                pass
    return DEFAULT_POST_PROCESSING_PROMPT


class PostProcessingDialog(QDialog):
    """
    Dialog displaying post-processing instructions and prompts for polishing ASR output
    using LLMs (Gemini Gems, Custom GPTs, Claude Projects, Qwen).
    Fully supports high-contrast styling and customizable font sizes in both Dark and Light desktop themes.
    """

    def __init__(self, parent=None, is_dark: Optional[bool] = None):
        super().__init__(parent)
        self.setWindowTitle("💡 คำแนะนำการปรับปรุงข้อความด้วย AI (AI Post-Processing)")
        self.resize(820, 640)
        self.setMinimumSize(640, 500)
        self.prompt_font_size: int = 15  # Comfortable default for Thai typography
        self.prompt_text = load_post_processing_prompt()

        self._init_ui()

        # Theme resolution
        if is_dark is not None:
            self.is_dark = is_dark
        else:
            self.is_dark = self._detect_dark_theme()
        self.apply_theme(self.is_dark)

    def _detect_dark_theme(self) -> bool:
        """Detect whether dark theme is active from parent hierarchy or app."""
        p = self.parent()
        while p is not None:
            if hasattr(p, "is_dark_theme"):
                return bool(p.is_dark_theme)
            p = p.parent()
        app = QApplication.instance()
        if app:
            qss = app.styleSheet()
            if "#1e1e2e" in qss:
                return True
            if "#f5f5f7" in qss:
                return False
        return self.palette().color(self.backgroundRole()).lightness() < 128

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # Header Title
        self.title_label = QLabel("✨ การทำ Post-Processing ขัดเกลาข้อความถอดเสียงด้วย AI", self)
        layout.addWidget(self.title_label)

        # Description Info Callout
        self.desc_label = QLabel(
            "ระบบถอดเสียง ASR มักมีคำพ้องเสียง วรรณยุกต์เพี้ยน หรือช่องไฟไม่สม่ำเสมอตามธรรมชาติของเสียงพูด "
            "ท่านสามารถนำ <b>Instruction / Prompt</b> ด้านล่างนี้ไปตั้งค่าใน <b>Gemini Gems</b>, "
            "<b>ChatGPT (Custom GPTs)</b>, <b>Claude Projects</b>, หรือ <b>Qwen</b> "
            "เพื่อสร้าง AI ตรวจแก้คำผิดและจัดระเบียบข้อความภาษาไทยให้อัตโนมัติ โดยยังคงสไตล์และน้ำเสียงเดิมไว้ครบถ้วน:",
            self,
        )
        self.desc_label.setWordWrap(True)
        layout.addWidget(self.desc_label)

        # Prompt Header Row with Zoom Controls
        prompt_hdr_layout = QHBoxLayout()
        self.lbl_prompt_heading = QLabel("📋 คำสั่ง System Instruction / Prompt:", self)
        prompt_hdr_layout.addWidget(self.lbl_prompt_heading)
        prompt_hdr_layout.addStretch()

        self.btn_zoom_out = QPushButton("➖", self)
        self.btn_zoom_out.setToolTip("ลดขนาดตัวอักษร Prompt (Zoom Out -1pt) [Ctrl + -]")
        self.btn_zoom_out.setFixedWidth(36)
        self.btn_zoom_out.clicked.connect(lambda: self._adjust_prompt_font_size(-1))

        self.lbl_font_size = QLabel(f"{self.prompt_font_size} pt", self)

        self.btn_zoom_in = QPushButton("➕", self)
        self.btn_zoom_in.setToolTip("เพิ่มขนาดตัวอักษร Prompt (Zoom In +1pt) [Ctrl + +]")
        self.btn_zoom_in.setFixedWidth(36)
        self.btn_zoom_in.clicked.connect(lambda: self._adjust_prompt_font_size(1))

        prompt_hdr_layout.addWidget(self.btn_zoom_out)
        prompt_hdr_layout.addWidget(self.lbl_font_size)
        prompt_hdr_layout.addWidget(self.btn_zoom_in)
        layout.addLayout(prompt_hdr_layout)

        # Prompt Text Box
        self.txt_prompt = QTextEdit(self)
        self.txt_prompt.setReadOnly(True)
        self.txt_prompt.setPlainText(self.prompt_text)

        # Connect Ctrl+MouseWheel zooming
        orig_wheel_event = self.txt_prompt.wheelEvent
        def _on_prompt_wheel(event):
            if event.modifiers() & Qt.ControlModifier:
                delta = event.angleDelta().y()
                if delta > 0:
                    self._adjust_prompt_font_size(2)
                elif delta < 0:
                    self._adjust_prompt_font_size(-2)
                event.accept()
            else:
                orig_wheel_event(event)
        self.txt_prompt.wheelEvent = _on_prompt_wheel

        layout.addWidget(self.txt_prompt, stretch=1)

        # Action Buttons Row
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.lbl_copied = QLabel("", self)

        self.btn_copy = QPushButton("📋 คัดลอก Prompt (Copy Prompt)", self)
        self.btn_copy.clicked.connect(self._copy_prompt_to_clipboard)

        self.btn_close = QPushButton("ปิด (Close)", self)
        self.btn_close.clicked.connect(self.accept)

        btn_layout.addWidget(self.lbl_copied)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_copy)
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

    def _adjust_prompt_font_size(self, delta: int) -> None:
        """Increase or decrease prompt text font size."""
        self.prompt_font_size = max(10, min(self.prompt_font_size + delta, 36))
        self.lbl_font_size.setText(f"{self.prompt_font_size} pt")
        self._update_prompt_style()

    def _update_prompt_style(self) -> None:
        """Update QTextEdit styling with current font size and theme colors."""
        font = self.txt_prompt.font()
        font.setPointSize(self.prompt_font_size)
        self.txt_prompt.setFont(font)
        self.txt_prompt.document().setDefaultFont(font)

        if self.is_dark:
            self.txt_prompt.setStyleSheet(
                f"QTextEdit {{ background-color: #181825; color: #cdd6f4; border: 1px solid #45475a; "
                f"border-radius: 8px; padding: 14px; font-size: {self.prompt_font_size}px; line-height: 1.6; "
                f"font-family: {THAI_FONT_FAMILY}; }}"
                f"QTextEdit:focus {{ border-color: #89b4fa; }}"
            )
        else:
            self.txt_prompt.setStyleSheet(
                f"QTextEdit {{ background-color: #ffffff; color: #1d1d1f; border: 1px solid #d2d2d7; "
                f"border-radius: 8px; padding: 14px; font-size: {self.prompt_font_size}px; line-height: 1.6; "
                f"font-family: {THAI_FONT_FAMILY}; }}"
                f"QTextEdit:focus {{ border-color: #0071e3; }}"
            )

    def apply_theme(self, is_dark: bool) -> None:
        """Apply high-contrast dark or light theme styling."""
        self.is_dark = is_dark
        if is_dark:
            self.setStyleSheet("QDialog { background-color: #1e1e2e; color: #cdd6f4; }")
            self.title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #89b4fa;")
            self.desc_label.setStyleSheet(
                "color: #cdd6f4; font-size: 15px; line-height: 1.6; "
                "background-color: #24273a; padding: 14px; border-radius: 8px; border: 1px solid #363a4f;"
            )
            self.lbl_prompt_heading.setStyleSheet("font-size: 15px; font-weight: bold; color: #89b4fa;")
            self.lbl_font_size.setStyleSheet("font-size: 13px; font-weight: bold; color: #cdd6f4;")
            self.btn_zoom_in.setStyleSheet(
                "QPushButton { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; "
                "border-radius: 4px; padding: 4px; font-weight: bold; font-size: 13px; }"
                "QPushButton:hover { background-color: #45475a; }"
            )
            self.btn_zoom_out.setStyleSheet(
                "QPushButton { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; "
                "border-radius: 4px; padding: 4px; font-weight: bold; font-size: 13px; }"
                "QPushButton:hover { background-color: #45475a; }"
            )
            self.lbl_copied.setStyleSheet("font-size: 14px; color: #a6e3a1; font-weight: bold;")
            self.btn_copy.setStyleSheet(
                "QPushButton { background-color: #89b4fa; color: #11111b; font-weight: bold; font-size: 15px; "
                "padding: 10px 22px; border-radius: 6px; border: none; }"
                "QPushButton:hover { background-color: #b4befe; }"
                "QPushButton:pressed { background-color: #74c7ec; }"
            )
            self.btn_close.setStyleSheet(
                "QPushButton { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; "
                "padding: 10px 20px; border-radius: 6px; font-weight: 600; font-size: 15px; }"
                "QPushButton:hover { background-color: #45475a; }"
            )
        else:
            # Light theme: high-contrast dark text on clean light background
            self.setStyleSheet("QDialog { background-color: #f5f5f7; color: #1d1d1f; }")
            self.title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #0071e3;")
            self.desc_label.setStyleSheet(
                "color: #1d1d1f; font-size: 15px; line-height: 1.6; "
                "background-color: #e5e5ea; padding: 14px; border-radius: 8px; border: 1px solid #d2d2d7;"
            )
            self.lbl_prompt_heading.setStyleSheet("font-size: 15px; font-weight: bold; color: #0071e3;")
            self.lbl_font_size.setStyleSheet("font-size: 13px; font-weight: bold; color: #1d1d1f;")
            self.btn_zoom_in.setStyleSheet(
                "QPushButton { background-color: #ffffff; color: #1d1d1f; border: 1px solid #d2d2d7; "
                "border-radius: 4px; padding: 4px; font-weight: bold; font-size: 13px; }"
                "QPushButton:hover { background-color: #e5e5ea; }"
            )
            self.btn_zoom_out.setStyleSheet(
                "QPushButton { background-color: #ffffff; color: #1d1d1f; border: 1px solid #d2d2d7; "
                "border-radius: 4px; padding: 4px; font-weight: bold; font-size: 13px; }"
                "QPushButton:hover { background-color: #e5e5ea; }"
            )
            self.lbl_copied.setStyleSheet("font-size: 14px; color: #28a745; font-weight: bold;")
            self.btn_copy.setStyleSheet(
                "QPushButton { background-color: #0071e3; color: #ffffff; font-weight: bold; font-size: 15px; "
                "padding: 10px 22px; border-radius: 6px; border: none; }"
                "QPushButton:hover { background-color: #0077ed; }"
                "QPushButton:pressed { background-color: #005bb5; }"
            )
            self.btn_close.setStyleSheet(
                "QPushButton { background-color: #ffffff; color: #1d1d1f; border: 1px solid #d2d2d7; "
                "padding: 10px 20px; border-radius: 6px; font-weight: 600; font-size: 15px; }"
                "QPushButton:hover { background-color: #e5e5ea; }"
            )
        self._update_prompt_style()

    def _copy_prompt_to_clipboard(self) -> None:
        """Copy the full post-processing prompt to system clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.prompt_text, QClipboard.Clipboard)
        self.lbl_copied.setText("✅ คัดลอก Prompt ไปยัง Clipboard แล้ว!")
        self.btn_copy.setText("✅ คัดลอกเรียบร้อย!")

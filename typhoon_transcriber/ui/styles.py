"""
Modern Dark & Light QSS stylesheets with Thai typography metric standards.
Prevents clipping of upper/lower Thai vowels and tone marks.
"""

# Thai Font Family Stack
THAI_FONT_FAMILY = (
    "'Sarabun', 'Noto Sans Thai', 'Leelawadee UI', 'Segoe UI', 'Ubuntu', 'Cantarell', sans-serif"
)

DARK_THEME_QSS = f"""
QMainWindow, QDialog {{
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: {THAI_FONT_FAMILY};
    font-size: 14px;
}}

QWidget {{
    font-family: {THAI_FONT_FAMILY};
}}

QToolBar {{
    background-color: #181825;
    border-bottom: 1px solid #313244;
    padding: 6px;
    spacing: 8px;
}}

QPushButton {{
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
    min-height: 20px;
}}

QPushButton:hover {{
    background-color: #45475a;
    border-color: #89b4fa;
}}

QPushButton:pressed {{
    background-color: #585b70;
}}

QPushButton#start_btn {{
    background-color: #a6e3a1;
    color: #11111b;
    border: 1px solid #94e2d5;
}}

QPushButton#start_btn:hover {{
    background-color: #94e2d5;
}}

QPushButton#pause_btn {{
    background-color: #fab387;
    color: #11111b;
    border: 1px solid #f9e2af;
}}

QPushButton#pause_btn:hover {{
    background-color: #f9e2af;
}}

QPushButton#stop_btn {{
    background-color: #f38ba8;
    color: #11111b;
    border: 1px solid #eba0ac;
}}

QPushButton#stop_btn:hover {{
    background-color: #eba0ac;
}}

QComboBox {{
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 12px;
    min-width: 220px;
    min-height: 22px;
}}

QComboBox:hover {{
    border-color: #89b4fa;
}}

QComboBox::drop-down {{
    border: none;
}}

QComboBox QAbstractItemView {{
    background-color: #1e1e2e;
    color: #cdd6f4;
    selection-background-color: #45475a;
    border: 1px solid #45475a;
    padding: 4px;
}}

QTextEdit {{
    background-color: #11111b;
    color: #cdd6f4;
    border: 1px solid #313244;
    border-radius: 8px;
    padding: 12px;
    line-height: 1.5;
    selection-background-color: #585b70;
}}

QProgressBar {{
    border: 1px solid #313244;
    border-radius: 6px;
    text-align: center;
    color: #cdd6f4;
    background-color: #181825;
    height: 18px;
}}

QProgressBar::chunk {{
    background-color: #89b4fa;
    border-radius: 5px;
}}

QStatusBar {{
    background-color: #181825;
    color: #a6adc8;
    border-top: 1px solid #313244;
    font-size: 12px;
    padding: 4px 8px;
}}

QLabel {{
    color: #cdd6f4;
}}

QTabWidget::pane {{
    border: 1px solid #313244;
    border-radius: 8px;
    background-color: #181825;
    padding: 8px;
}}

QTabBar::tab {{
    background-color: #313244;
    color: #a6adc8;
    border: 1px solid #45475a;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 18px;
    font-weight: 600;
    margin-right: 3px;
}}

QTabBar::tab:selected {{
    background-color: #181825;
    color: #89b4fa;
    border-bottom: 2px solid #89b4fa;
}}

QTabBar::tab:hover:!selected {{
    background-color: #45475a;
    color: #cdd6f4;
}}

QTableWidget {{
    background-color: #11111b;
    color: #cdd6f4;
    border: 1px solid #313244;
    border-radius: 6px;
    gridline-color: #313244;
    selection-background-color: #45475a;
    selection-color: #cdd6f4;
}}

QHeaderView::section {{
    background-color: #181825;
    color: #89b4fa;
    padding: 6px;
    border: 1px solid #313244;
    font-weight: bold;
}}

QLineEdit {{
    background-color: #11111b;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 10px;
}}

QLineEdit:focus {{
    border-color: #89b4fa;
}}

QGroupBox {{
    border: 1px solid #313244;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 12px;
    font-weight: bold;
    color: #89b4fa;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}}
"""

LIGHT_THEME_QSS = f"""
QMainWindow, QDialog {{
    background-color: #f5f5f7;
    color: #1d1d1f;
    font-family: {THAI_FONT_FAMILY};
    font-size: 14px;
}}

QWidget {{
    font-family: {THAI_FONT_FAMILY};
}}

QToolBar {{
    background-color: #ffffff;
    border-bottom: 1px solid #d2d2d7;
    padding: 6px;
    spacing: 8px;
}}

QPushButton {{
    background-color: #ffffff;
    color: #1d1d1f;
    border: 1px solid #d2d2d7;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
    min-height: 20px;
}}

QPushButton:hover {{
    background-color: #f0f0f2;
    border-color: #0071e3;
}}

QPushButton#start_btn {{
    background-color: #34c759;
    color: #ffffff;
    border: none;
}}

QPushButton#start_btn:hover {{
    background-color: #28a745;
}}

QPushButton#pause_btn {{
    background-color: #ff9500;
    color: #ffffff;
    border: none;
}}

QPushButton#pause_btn:hover {{
    background-color: #e08500;
}}

QPushButton#stop_btn {{
    background-color: #ff3b30;
    color: #ffffff;
    border: none;
}}

QPushButton#stop_btn:hover {{
    background-color: #d70015;
}}

QComboBox {{
    background-color: #ffffff;
    color: #1d1d1f;
    border: 1px solid #d2d2d7;
    border-radius: 6px;
    padding: 6px 12px;
    min-width: 220px;
    min-height: 22px;
}}

QTextEdit {{
    background-color: #ffffff;
    color: #1d1d1f;
    border: 1px solid #d2d2d7;
    border-radius: 8px;
    padding: 12px;
    line-height: 1.5;
    selection-background-color: #b3d7ff;
}}

QProgressBar {{
    border: 1px solid #d2d2d7;
    border-radius: 6px;
    text-align: center;
    color: #1d1d1f;
    background-color: #e5e5ea;
    height: 18px;
}}

QProgressBar::chunk {{
    background-color: #0071e3;
    border-radius: 5px;
}}

QStatusBar {{
    background-color: #ffffff;
    color: #6e6e73;
    border-top: 1px solid #d2d2d7;
    font-size: 12px;
    padding: 4px 8px;
}}

QLabel {{
    color: #1d1d1f;
}}

QTabWidget::pane {{
    border: 1px solid #d2d2d7;
    border-radius: 8px;
    background-color: #ffffff;
    padding: 8px;
}}

QTabBar::tab {{
    background-color: #e5e5ea;
    color: #6e6e73;
    border: 1px solid #d2d2d7;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 18px;
    font-weight: 600;
    margin-right: 3px;
}}

QTabBar::tab:selected {{
    background-color: #ffffff;
    color: #0071e3;
    border-bottom: 2px solid #0071e3;
}}

QTabBar::tab:hover:!selected {{
    background-color: #d2d2d7;
    color: #1d1d1f;
}}

QTableWidget {{
    background-color: #ffffff;
    color: #1d1d1f;
    border: 1px solid #d2d2d7;
    border-radius: 6px;
    gridline-color: #e5e5ea;
    selection-background-color: #b3d7ff;
    selection-color: #1d1d1f;
}}

QHeaderView::section {{
    background-color: #f5f5f7;
    color: #0071e3;
    padding: 6px;
    border: 1px solid #d2d2d7;
    font-weight: bold;
}}

QLineEdit {{
    background-color: #ffffff;
    color: #1d1d1f;
    border: 1px solid #d2d2d7;
    border-radius: 6px;
    padding: 6px 10px;
}}

QLineEdit:focus {{
    border-color: #0071e3;
}}

QGroupBox {{
    border: 1px solid #d2d2d7;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 12px;
    font-weight: bold;
    color: #0071e3;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}}
"""

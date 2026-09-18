"""
Typhoon ASR Desktop Transcriber package.
"""

def main():
    """Application entrypoint with lazy GUI import to keep package import lightweight."""
    from .main import main as _main
    return _main()


__all__ = ["main"]

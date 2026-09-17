"""
CLI entry point for running typhoon_transcriber module directly:
    python -m typhoon_transcriber
"""

import sys
from typhoon_transcriber.main import main

if __name__ == "__main__":
    sys.exit(main())

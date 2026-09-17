"""
Thai Tokenizer wrapper supporting SentencePiece (.model) and vocabulary list (.json).
Includes Thai text post-processing and normalization.
"""

from pathlib import Path
from typing import List, Optional, Union, Dict
import json
import re
import logging
import sentencepiece as spm

logger = logging.getLogger(__name__)


def normalize_thai_text(text: str) -> str:
    """
    Apply Thai text normalization rules:
    - Remove redundant repeated Thai tone marks and upper/lower vowels
    - Normalize zero-width spaces and irregular whitespaces
    - Fix common punctuation spacing
    """
    if not text:
        return ""

    # Remove zero-width spaces / joiners
    text = text.replace("\u200b", "").replace("\ufeff", "")

    # Clean SentencePiece / BPE underscore boundary markers (\u2581)
    text = text.replace("\u2581", " ")

    # Collapse multiple spaces into single space
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize repeated Thai vowels / tone marks (e.g. ้้ -> ้)
    text = re.sub(r"([\u0e31\u0e34-\u0e39\u0e47-\u0e4e])\1+", r"\1", text)

    return text.strip()


class ThaiTokenizer:
    """
    Tokenizer wrapper supporting both SentencePiece (.model) and JSON vocabulary files (.json).
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.sp = spm.SentencePieceProcessor()
        self.vocab_list: Optional[List[str]] = None
        self.vocab_map: Dict[str, int] = {}
        self.is_loaded = False

        if model_path:
            self.load(model_path)

    def load(self, model_path: Union[str, Path]) -> None:
        """Load SentencePiece model (.model) or vocabulary file (.json) from disk."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer / vocab file not found at {path}")

        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.vocab_list = data
                elif isinstance(data, dict):
                    # Invert if it's token -> id or id -> token
                    if all(isinstance(k, str) and isinstance(v, int) for k, v in data.items()):
                        self.vocab_list = [None] * len(data)  # type: ignore
                        for tok, tid in data.items():
                            if tid < len(self.vocab_list):
                                self.vocab_list[tid] = tok
                    else:
                        self.vocab_list = list(data.values())
            self.vocab_map = {tok: idx for idx, tok in enumerate(self.vocab_list) if tok}
            self.is_loaded = True
            logger.info("Loaded JSON vocabulary from %s (size: %d)", path, len(self.vocab_list))
        else:
            self.sp.load(str(path))
            self.is_loaded = True
            logger.info("SentencePiece tokenizer loaded from %s (size: %d)", path, self.sp.get_piece_size())

    @property
    def vocab_size(self) -> int:
        if not self.is_loaded:
            return 0
        if self.vocab_list is not None:
            return len(self.vocab_list)
        return self.sp.get_piece_size()

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if not self.is_loaded:
            raise RuntimeError("Tokenizer is not loaded.")
        if self.vocab_list is not None:
            # Fallback simple character/subword lookup
            return [self.vocab_map.get(ch, 0) for ch in text if ch in self.vocab_map]
        return self.sp.encode_as_ids(text)

    def decode(self, token_ids: List[int], normalize: bool = True) -> str:
        """
        Decode token IDs to string with optional Thai normalization.
        Filters out blank / special padding token IDs.
        """
        if not self.is_loaded:
            raise RuntimeError("Tokenizer is not loaded.")

        if self.vocab_list is not None:
            # Decode using vocab list
            special_tokens = {"<unk>", "<s>", "</s>", "<pad>", "<bos>", "<eos>"}
            tokens = []
            for tid in token_ids:
                if 0 <= tid < len(self.vocab_list):
                    tok = self.vocab_list[tid]
                    if tok and tok not in special_tokens:
                        tokens.append(tok)
            raw_text = "".join(tokens).replace("\u2581", " ")
        else:
            valid_ids = [int(i) for i in token_ids if 0 <= i < self.vocab_size]
            raw_text = self.sp.decode_ids(valid_ids)

        if normalize:
            return normalize_thai_text(raw_text)
        return raw_text

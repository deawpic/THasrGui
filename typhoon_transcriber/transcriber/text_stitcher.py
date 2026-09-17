"""
Overlap-and-merge text stitching algorithm for sliding window streaming ASR.
Specialized for Thai language to prevent duplicated phrases, repeated sentences,
and spacing variations between overlapping speech frames.
"""

from typing import List


class TextStitcher:
    """
    Stitches consecutive transcriptions produced by streaming ASR.
    Detects suffix-to-prefix overlaps, suppresses redundant substrings, provides
    real-time live preview without state mutation, and prevents duplicate sentences.
    """

    def __init__(self, min_overlap_chars: int = 2):
        self.min_overlap_chars = min_overlap_chars
        self.accumulated_text = ""

    def get_text(self) -> str:
        """Return the current accumulated text."""
        return self.accumulated_text

    def set_text(self, text: str) -> None:
        """Set or initialize the accumulated text."""
        self.accumulated_text = text.strip()

    def preview(self, interim_text: str) -> str:
        """
        Generate preview transcript combining accumulated text and live interim text
        without committing interim text to the permanent transcript.
        """
        interim_text = interim_text.strip()
        if not interim_text:
            return self.accumulated_text
        if not self.accumulated_text:
            return interim_text

        clean_accum = "".join(self.accumulated_text.split())
        clean_interim = "".join(interim_text.split())

        # If interim is already contained in the recent accumulated text tail, don't show duplicate
        tail = clean_accum[-max(len(clean_interim) * 2, 80):]
        if clean_interim in tail or tail.endswith(clean_interim):
            return self.accumulated_text

        # Suffix-prefix overlap matching for preview
        max_k = min(len(clean_accum), len(clean_interim), 80)
        best_overlap = 0
        for k in range(max_k, self.min_overlap_chars - 1, -1):
            if clean_accum.endswith(clean_interim[:k]):
                best_overlap = k
                break

        if best_overlap > 0:
            count = 0
            cut_idx = len(interim_text)
            for i, ch in enumerate(interim_text):
                if not ch.isspace():
                    count += 1
                if count == best_overlap:
                    cut_idx = i + 1
                    break
            new_part = interim_text[cut_idx:]
            if not new_part or not new_part.strip():
                return self.accumulated_text
            clean_new = "".join(new_part.split())
            if clean_new and clean_accum.endswith(clean_new):
                return self.accumulated_text
            if new_part.startswith(" ") and not self.accumulated_text.endswith(" "):
                return f"{self.accumulated_text} {new_part.lstrip()}"
            return f"{self.accumulated_text}{new_part}"

        # No overlap between accumulated text and interim text
        if self.accumulated_text.endswith(" ") or interim_text.startswith(" "):
            return f"{self.accumulated_text}{interim_text}"
        return f"{self.accumulated_text} {interim_text}"

    def append_chunk(self, chunk_text: str) -> str:
        """
        Merge a finalized chunk transcription into accumulated text,
        filtering out duplicate or overlapping phrases.
        """
        chunk_text = chunk_text.strip()
        if not chunk_text:
            return self.accumulated_text

        if not self.accumulated_text:
            self.accumulated_text = chunk_text
            return self.accumulated_text

        # 1. Non-space normalized representations for space-agnostic comparison
        clean_accum = "".join(self.accumulated_text.split())
        clean_chunk = "".join(chunk_text.split())

        # 2. Complete redundancy check against recent history
        # Check against a generous tail window to catch re-decoded tails or repeated phrases
        tail_window = clean_accum[-max(len(clean_chunk) * 2, 120):]
        if len(clean_chunk) >= self.min_overlap_chars and (
            clean_chunk in tail_window or tail_window.endswith(clean_chunk)
        ):
            return self.accumulated_text

        # 3. Check phrase-level duplicate against recent tokens/phrases
        recent_tokens = [t for t in self.accumulated_text.replace("\n", " ").split(" ") if t]
        if recent_tokens:
            last_token_clean = "".join(recent_tokens[-1].split())
            if last_token_clean == clean_chunk:
                return self.accumulated_text

        # 4. Find longest common overlap: suffix of clean_accum matching prefix of clean_chunk
        max_k = min(len(clean_accum), len(clean_chunk), 80)
        best_clean_overlap = 0

        for k in range(max_k, self.min_overlap_chars - 1, -1):
            prefix = clean_chunk[:k]
            if clean_accum.endswith(prefix):
                best_clean_overlap = k
                break

        # Suffix search in recent history: handle slight trailing noise or token variations
        if best_clean_overlap == 0:
            tail_check_len = min(len(clean_accum), len(clean_chunk) + 20)
            recent_accum = clean_accum[-tail_check_len:]
            for k in range(min(len(clean_chunk), 40), self.min_overlap_chars + 1, -1):
                prefix = clean_chunk[:k]
                pos = recent_accum.rfind(prefix)
                if pos != -1:
                    overlap_from_end = len(recent_accum) - pos
                    if overlap_from_end <= k + 10:
                        best_clean_overlap = k
                        break

        if best_clean_overlap > 0:
            # Map best_clean_overlap non-space characters back to index in original chunk_text
            count = 0
            cut_idx = len(chunk_text)
            for i, ch in enumerate(chunk_text):
                if not ch.isspace():
                    count += 1
                if count == best_clean_overlap:
                    cut_idx = i + 1
                    break
            new_part = chunk_text[cut_idx:]
            if not new_part or not new_part.strip():
                return self.accumulated_text

            clean_new = "".join(new_part.split())
            if clean_new and clean_accum.endswith(clean_new):
                return self.accumulated_text
            if new_part.startswith(" ") and not self.accumulated_text.endswith(" "):
                self.accumulated_text += " " + new_part.lstrip()
            else:
                self.accumulated_text += new_part
        else:
            new_part = chunk_text
            clean_new = "".join(new_part.split())
            if clean_new and clean_accum.endswith(clean_new):
                return self.accumulated_text
            if self.accumulated_text.endswith(" ") or new_part.startswith(" "):
                self.accumulated_text += new_part
            else:
                self.accumulated_text += " " + new_part

        return self.accumulated_text

    def flush_boundary(self) -> None:
        """Mark end of utterance at silence boundary."""
        if self.accumulated_text and not self.accumulated_text.endswith(" "):
            self.accumulated_text += " "

    def reset(self) -> None:
        """Reset the stitcher accumulator."""
        self.accumulated_text = ""


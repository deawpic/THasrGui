"""
Overlap-and-merge text stitching algorithm for sliding window streaming ASR.
"""

from typing import List


class TextStitcher:
    """
    Stitches consecutive transcriptions produced by overlapping audio sliding windows.
    Finds maximum overlapping suffix/prefix between previous and current text chunks.
    """

    def __init__(self, min_overlap_chars: int = 2):
        self.min_overlap_chars = min_overlap_chars
        self.accumulated_text = ""

    def append_chunk(self, chunk_text: str) -> str:
        """
        Merge a new chunk transcription into the accumulated text.

        Returns
        -------
        updated_full_text : str
            The complete updated transcript.
        """
        chunk_text = chunk_text.strip()
        if not chunk_text:
            return self.accumulated_text

        if not self.accumulated_text:
            self.accumulated_text = chunk_text
            return self.accumulated_text

        # Find longest common overlap between end of accumulated_text and start of chunk_text
        max_search = min(len(self.accumulated_text), len(chunk_text), 50)
        best_overlap = 0

        for k in range(max_search, self.min_overlap_chars - 1, -1):
            if self.accumulated_text.endswith(chunk_text[:k]):
                best_overlap = k
                break

        if best_overlap > 0:
            # Append only non-overlapping remainder
            new_part = chunk_text[best_overlap:]
            self.accumulated_text += new_part
        else:
            # If no overlap, append with space (or direct attach for Thai)
            if self.accumulated_text.endswith(" ") or chunk_text.startswith(" "):
                self.accumulated_text += chunk_text
            else:
                self.accumulated_text += " " + chunk_text

        return self.accumulated_text

    def reset(self) -> None:
        """Reset the stitcher accumulator."""
        self.accumulated_text = ""

"""
Marathi text normalization for ASR fine-tuning transcripts.

Devanagari can be represented with multiple codepoint sequences that render
identically, silently fragmenting the tokenizer if left unnormalized. NFC
normalization runs first to collapse these to one canonical form.
"""

import re
import sys
import unicodedata

DEVANAGARI_DANDA = "।"
DEVANAGARI_DOUBLE_DANDA = "॥"

_QUOTE_CHARS = (
    "\"'`"
    "‘’‚‛“”„‟"
    "‹›«»"
)

_NOISE_PUNCT = "*_~^|\\<>{}[]#@$%&=+…"

_QUOTE_RE = re.compile(f"[{re.escape(_QUOTE_CHARS)}]")
_NOISE_RE = re.compile(f"[{re.escape(_NOISE_PUNCT)}]")

# Collapse repeated Latin punctuation ("..", "!!"); danda/double-danda are
# excluded so an intentional "।।" isn't merged into one danda.
_REPEATED_PUNCT_RE = re.compile(r"([.,!?;:])\1+")

# A standalone "." or "," is almost always a transcription-tool artifact,
# not deliberate -- Marathi prose doesn't end clauses on a bare Latin period.
_STANDALONE_PUNCT_RE = re.compile(r"(?<!\S)[.,](?!\S)")

_SPACE_BEFORE_PUNCT_RE = re.compile(
    rf"\s+([.,!?;:{DEVANAGARI_DANDA}{DEVANAGARI_DOUBLE_DANDA}])"
)

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_marathi(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = _QUOTE_RE.sub("", text)
    text = _NOISE_RE.sub("", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)
    text = _STANDALONE_PUNCT_RE.sub("", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _main() -> None:
    raw = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else sys.stdin.read()
    print(normalize_marathi(raw))


if __name__ == "__main__":
    _main()

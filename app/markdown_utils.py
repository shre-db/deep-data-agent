"""Markdown preprocessing for agent-rendered text.

Streamlit renders markdown with KaTeX math enabled, so `$...$` pairs are
parsed as LaTeX delimiters. Currency amounts break that: the first `$` of
`$3,643,063.54` opens a math span that closes at the next `$`, mangling the
text between in italic serif math. `protect_currency` escapes the `$` that
starts a currency amount while leaving genuine LaTeX delimiters untouched.
"""

from __future__ import annotations

import re
from pathlib import Path

_MATH_HINT_CHARS = set("\\^_{}")

# Fenced code blocks (``` ... ```) and inline code spans (`...`): their
# content must pass through byte-identical, never currency-escaped.
_CODE_RE = re.compile(r"(`{3,})[\s\S]*?\1|`[^`\n]*`")


def _escape_dollars(text: str) -> str:
    """Escape currency '$' in one text segment (no code blocks inside)."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch != "$" or (i > 0 and text[i - 1] == "\\"):
            out.append(ch)
            i += 1
            continue
        # Display math: $$ opener at start/after whitespace with a closing $$.
        if text.startswith("$$", i) and (i == 0 or text[i - 1].isspace()):
            close = text.find("$$", i + 2)
            if close != -1:
                out.append(text[i : close + 2])
                i = close + 2
                continue
        # Find the next unescaped $ on the same line — the markdown math
        # engine never pairs delimiters across lines, and neither do we.
        close = -1
        j = i + 1
        while j < n and text[j] != "\n":
            if text[j] == "$" and text[j - 1] != "\\":
                close = j
                break
            j += 1
        if close == -1:
            out.append("\\$")  # unmatched: render it literally
            i += 1
            continue
        span = text[i + 1 : close]
        is_math = bool(_MATH_HINT_CHARS & set(span)) or (
            bool(span.strip()) and not span.strip()[0].isdigit()
        )
        if is_math:
            # LaTeX: keep both delimiters, continue after the closer.
            out.append(text[i : close + 1])
            i = close + 1
        else:
            # Currency: escape the opener only; the closer is classified on
            # the next iteration (it may open its own span).
            out.append("\\$")
            i += 1
    return "".join(out)


def protect_currency(text: str) -> str:
    """Escape currency '$' so Streamlit does not parse it as LaTeX.

    Currency forms like `$5,824`, `$98–106`, `$3,643,063.54` stay literal;
    `$x^2 + y^2 = z^2$` and `$$...$$` display math pass through unchanged,
    as does anything inside code spans or fenced code blocks.
    """
    parts: list[str] = []
    pos = 0
    for match in _CODE_RE.finditer(text):
        parts.append(_escape_dollars(text[pos : match.start()]))
        parts.append(match.group(0))
        pos = match.end()
    parts.append(_escape_dollars(text[pos:]))
    return "".join(parts)


# Optional list-marker prefix so references work inside bullet lists too.
# The `!` is optional: models often write plain links for tables.
_REF_LINE_RE = re.compile(r"^(\s*(?:[-*+]\s+)?)(!?)\[([^\]]*)\]\(([^)\s]+)\)\s*$")

_ANSWER_HEADING_RE = re.compile(r"^\s*#{1,6}\s*Answer\s*$", re.IGNORECASE)


def strip_leading_answer_heading(content: str) -> str:
    """Drop an 'Answer' heading when it opens the answer.

    In a chat bubble everything is the answer, so an '## Answer' heading at
    the top is noise. Only the first content line is considered; headings
    later in the text are left alone.
    """
    lines = content.splitlines()
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    if start < len(lines) and _ANSWER_HEADING_RE.match(lines[start]):
        del lines[start]
    return "\n".join(lines)


def _match_artifact(target: str, artifacts: list[Path]) -> Path | None:
    """Match a reference target against the turn's artifacts.

    Tolerant of the leading-'/' virtual-root form file tools report
    (`/artifacts/<thread>/x.json`) and of bare filenames.
    """
    norm = Path(target.strip().lstrip("/"))
    for path in artifacts:
        if path.name == norm.name and (
            str(path).endswith(str(norm)) or len(norm.parts) == 1
        ):
            return path
    return None


def iter_answer_segments(
    content: str, artifacts: list[Path]
) -> "Iterator[tuple[str, object]]":
    """Split an answer into ("markdown", text) and ("figure", payload) parts.

    A line of the form `![caption](artifact/path.json)` or a plain
    `[caption](artifact/path.csv)` whose target matches an artifact becomes
    a figure segment (`{"path": Path, "caption": str}`); everything else
    stays markdown. An unmatched image reference renders as its caption
    text (no broken image); an unmatched plain link passes through
    unchanged (it may be a genuine external link).
    """
    buffer: list[str] = []
    for line in content.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        match = _REF_LINE_RE.match(stripped)
        path = _match_artifact(match.group(4), artifacts) if match else None
        if match and path:
            if buffer:
                yield ("markdown", "".join(buffer))
                buffer = []
            yield ("figure", {"path": path, "caption": match.group(3)})
        elif match and match.group(2):
            buffer.append(match.group(1) + match.group(3) + "\n")
        else:
            buffer.append(line)
    if buffer:
        yield ("markdown", "".join(buffer))

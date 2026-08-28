"""Tests for markdown preprocessing (currency vs LaTeX, figure references)."""

from pathlib import Path

from app.markdown_utils import iter_answer_segments, protect_currency


# --- protect_currency: currency -------------------------------------------


def test_currency_amounts_escaped():
    assert protect_currency("Revenue was $3,643,063.54 total.") == (
        r"Revenue was \$3,643,063.54 total."
    )
    assert protect_currency("cost $5,824 today") == r"cost \$5,824 today"
    assert protect_currency("range $98–106 per unit") == r"range \$98–106 per unit"


def test_two_currency_amounts_in_one_sentence():
    out = protect_currency("was $3,643,063.54 and $724,522.57")
    assert out == r"was \$3,643,063.54 and \$724,522.57"


def test_unmatched_currency_dollar_escaped():
    assert protect_currency("costs $5") == r"costs \$5"


# --- protect_currency: LaTeX ----------------------------------------------


def test_inline_latex_untouched():
    text = r"$x^2 + y^2 = z^2$"
    assert protect_currency(text) == text


def test_display_latex_untouched():
    assert protect_currency(r"$$E = mc^2$$") == r"$$E = mc^2$$"
    assert protect_currency("$$\nE = mc^2\n$$") == "$$\nE = mc^2\n$$"


def test_math_starting_with_digit_kept():
    # Contains a backslash: math, not currency.
    assert protect_currency(r"$5 \times 3 = 15$") == r"$5 \times 3 = 15$"


# --- protect_currency: mixed content --------------------------------------


def test_mixed_currency_and_math():
    assert protect_currency("Cost $5,824 while $x^2$ grows") == (
        r"Cost \$5,824 while $x^2$ grows"
    )
    assert protect_currency("range $98–106 and $y = mx + b$") == (
        r"range \$98–106 and $y = mx + b$"
    )


def test_currency_never_pairs_across_lines():
    """The math engine pairs $ only within a line; a currency $ whose only
    possible closer is on a later line must not open a math span."""
    text = "Organic $1,387,240.56 and $724,522.57.\n\nThe model $y = mx + b$ fits."
    out = protect_currency(text)
    assert out == (
        r"Organic \$1,387,240.56 and \$724,522.57." + "\n\nThe model $y = mx + b$ fits."
    )


def test_code_spans_and_fences_untouched():
    fenced = '```python\nprint(f"${x}")\n```'
    assert protect_currency(fenced) == fenced
    assert protect_currency("run `echo $HOME` now") == "run `echo $HOME` now"


def test_already_escaped_dollar_not_double_escaped():
    assert protect_currency(r"\$5") == r"\$5"


# --- iter_answer_segments --------------------------------------------------


def _arts(*names: str) -> list[Path]:
    return [Path("/proj/artifacts/session-x") / n for n in names]


def test_figure_reference_splits_answer():
    segs = list(
        iter_answer_segments(
            "Intro text.\n![Rev chart](artifacts/session-x/rev.json)\nMore text.\n",
            _arts("rev.json"),
        )
    )
    assert [s[0] for s in segs] == ["markdown", "figure", "markdown"]
    assert segs[1][1]["path"].name == "rev.json"
    assert segs[1][1]["caption"] == "Rev chart"
    assert segs[0][1] == "Intro text.\n"
    assert segs[2][1] == "More text.\n"


def test_leading_slash_virtual_path_matches():
    segs = list(
        iter_answer_segments("![Rev](/artifacts/session-x/rev.json)\n", _arts("rev.json"))
    )
    assert [s[0] for s in segs] == ["figure"]


def test_bare_filename_matches():
    segs = list(iter_answer_segments("![Rev](rev.json)\n", _arts("rev.json")))
    assert [s[0] for s in segs] == ["figure"]


def test_list_item_reference_matches():
    segs = list(
        iter_answer_segments("- ![Rev](artifacts/session-x/rev.json)\n", _arts("rev.json"))
    )
    assert [s[0] for s in segs] == ["figure"]


def test_unmatched_reference_renders_caption_text():
    segs = list(iter_answer_segments("![Rev](other.json)\n", _arts("rev.json")))
    assert segs == [("markdown", "Rev\n")]


def test_plain_markdown_unchanged():
    text = "## Answer\n\nSome **bold** text.\n"
    assert list(iter_answer_segments(text, [])) == [("markdown", text)]

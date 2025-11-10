import os
import sys
import pytest

# Ensure project root is on import path (for `instructions` package).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.mark.parametrize(
    "prefix",
    [
        "「",  # Japanese opening quote
        "『",  # Japanese opening book quote
        "　",  # full-width space
        "> ",  # blockquote
        "# ",  # heading level 1
        "### ",  # heading level 3
        "- ",  # unordered list dash
        "* ",  # unordered list asterisk
        "+ ",  # unordered list plus
        "・",  # Japanese bullet
        "1. ",  # numbered list dot
        "2) ",  # numbered list paren
        "**",  # bold marker (leading)
        "*",   # italic marker (leading)
        "__",  # bold underscore
        "_",   # italic underscore
        "`",   # code span
    ],
)
def test_paragraph_first_word_normalization_needed(prefix):
    """
    The nth paragraph should be allowed to start with common formatting markers,
    and still be recognized as starting with the specified first word after
    normalization. We expect this test to fail before normalization is implemented.
    """
    from instructions.ja_instructions import ParagraphFirstWordCheck

    checker = ParagraphFirstWordCheck("ja:length_constraints:nth_paragraph_first_word")
    checker.build_description(num_paragraphs=2, nth_paragraph=2, first_word="創造性")

    response = f"序文\n\n{prefix}創造性 についての説明です。"

    assert checker.check_following(response) is True


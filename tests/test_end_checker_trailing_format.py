import os
import sys
import pytest

# Ensure project root is on import path (for `instructions` package).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_end_checker_passes_with_trailing_bold_markers():
    """
    Confirms EndChecker accepts when the required ending sentence
    is followed by formatting-only markers like trailing '**'.
    """
    from instructions.ja_instructions import EndChecker

    end_phrase = "「そばの香りが漂う中、物語は終わりを迎えた。」"
    checker = EndChecker("ja:startend:end_checker")
    checker.build_description(end_phrase=end_phrase)

    # Response ends with the correct sentence, but wrapped in bold markers at the end.
    response = (
        "本文...\n\n"
        "**「そばの香りが漂う中、物語は終わりを迎えた。」**"
    )

    assert checker.check_following(response) is True


def test_end_checker_passes_without_trailing_markers():
    """
    Control: with the exact ending (no trailing markers), it passes.
    """
    from instructions.ja_instructions import EndChecker

    end_phrase = "「そばの香りが漂う中、物語は終わりを迎えた。」"
    checker = EndChecker("ja:startend:end_checker")
    checker.build_description(end_phrase=end_phrase)

    response = (
        "本文...\n\n"
        "「そばの香りが漂う中、物語は終わりを迎えた。」"
    )

    assert checker.check_following(response) is True

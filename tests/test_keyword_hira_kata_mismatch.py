import os
import sys

# Ensure project root is on import path (for `instructions` package).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_keyword_existence_hiragana_katakana_mismatch_fails():
    """
    Mirrors row 172: prompt mentions たんぱく質 (hiragana) while kwargs passes
    タンパク質 (katakana). Our substring-based KeywordChecker does not normalize
    script differences, so this should return False when the response uses
    たんぱく質 but keyword expects タンパク質.
    """
    from instructions.ja_instructions import KeywordChecker

    checker = KeywordChecker("ja:keywords:existence")
    checker.build_description(keywords=["タンパク質", "炭水化物", "脂質"])

    response = "これは《たんぱく質》・《炭水化物》・《脂質》についての説明です。"

    assert checker.check_following(response) is False


def test_keyword_existence_katakana_present_passes():
    """
    Control: using タンパク質 (katakana) in the response satisfies the checker.
    """
    from instructions.ja_instructions import KeywordChecker

    checker = KeywordChecker("ja:keywords:existence")
    checker.build_description(keywords=["タンパク質", "炭水化物", "脂質"])

    response = "これは《タンパク質》・《炭水化物》・《脂質》についての説明です。"

    assert checker.check_following(response) is True


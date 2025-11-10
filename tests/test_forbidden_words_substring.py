import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_forbidden_words_catches_compound_by_substring():
    """Forbid '大聖堂' and ensure it is caught via substring matching."""
    from instructions.ja_instructions import ForbiddenWords

    checker = ForbiddenWords("ja:keywords:forbidden_words")
    checker.build_description(forbidden_words=["大聖堂"])

    response = "この大聖堂は世界的に有名です。"
    assert checker.check_following(response) is False


def test_forbidden_words_absent_passes():
    from instructions.ja_instructions import ForbiddenWords

    checker = ForbiddenWords("ja:keywords:forbidden_words")
    checker.build_description(forbidden_words=["禁止語"])

    response = "これは許可された文章です。"
    assert checker.check_following(response) is True


def test_forbidden_words_script_mismatch_not_flagged():
    """No normalization between scripts: forbidding タンパク質 should not flag たんぱく質."""
    from instructions.ja_instructions import ForbiddenWords

    checker = ForbiddenWords("ja:keywords:forbidden_words")
    checker.build_description(forbidden_words=["タンパク質"])  # katakana

    response = "これは たんぱく質 についての説明です。"  # hiragana
    assert checker.check_following(response) is True


import os
import sys

# Ensure project root is on import path (for `instructions` package).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_keyword_frequency_substring_counts_three_or_more():
    from instructions.ja_instructions import KeywordFrequencyChecker

    checker = KeywordFrequencyChecker("ja:keywords:frequency")
    checker.build_description(keyword="大聖堂", frequency=3, relation="以上")

    # Contains the substring "大聖堂" exactly three times.
    response = (
        "この大聖堂は有名です。\n"
        "内部の大聖堂は美しい。\n"
        "また、大聖堂には観光客が多い。"
    )

    assert checker.check_following(response) is True


def test_keyword_frequency_substring_less_than_required_fails():
    from instructions.ja_instructions import KeywordFrequencyChecker

    checker = KeywordFrequencyChecker("ja:keywords:frequency")
    checker.build_description(keyword="大聖堂", frequency=2, relation="以上")

    response = "この建築は壮大です。大聖堂のような雰囲気。"

    # Only one occurrence → should fail for relation "以上" 2
    assert checker.check_following(response) is False


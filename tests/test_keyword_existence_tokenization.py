import os
import sys
import pytest

# Ensure project root is on import path (for `instructions` package).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_keyword_checker_substring_matching_succeeds():
    """
    Verifies the updated ja:keywords:existence uses substring matching and
    succeeds when the raw text contains the keyword, regardless of tokenizer
    behavior.
    """

    from instructions.ja_instructions import KeywordChecker

    checker = KeywordChecker("ja:keywords:existence")
    checker.build_description(keywords=["創造性"])  # sets internal _keywords

    # A response that visibly contains the substring "創造性".
    response = "現代アートは、その創造性を通じて社会問題への意識を高める。"

    # Should pass with substring-based matching.
    assert checker.check_following(response) is True

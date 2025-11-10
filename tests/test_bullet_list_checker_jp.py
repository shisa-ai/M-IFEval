import os
import sys
import pytest

# Ensure project root is on import path (for `instructions` package).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from instructions.ja_instructions import BulletListChecker


def make_checker(n):
    chk = BulletListChecker("ja:detectable_format:number_bullet_lists")
    chk.build_description(num_bullets=n)
    return chk


def test_accepts_japanese_and_markdown_markers():
    chk = make_checker(3)
    value = (
        "以下に列挙します:\n\n"
        "・一つめの内容\n"
        "* second item\n"
        "- third item\n\n"
        "**強調** はカウントしない\n"
        "---\n"
        "終わり\n"
    )
    assert chk.check_following(value) is True


def test_does_not_count_bold_or_horizontal_rule():
    chk = make_checker(1)
    value = (
        "**太字の見出し**\n"
        "---\n"
        "・だけカウント\n"
    )
    assert chk.check_following(value) is True


def test_mismatch_returns_false():
    chk = make_checker(2)
    value = (
        "説明:\n"
        "- only-one-list-item\n"
        "---\n"
        "本文\n"
    )
    assert chk.check_following(value) is False

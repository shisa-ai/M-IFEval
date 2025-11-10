import os
import sys

import pytest

# Ensure project root is on import path (for `instructions` package).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_postscript_marker_pps_dot_with_spaced_prefix_passes():
    """
    Verifies the current inconsistency for row 38-style settings:
    - Dataset passes postscript_marker='P.P.S.' (with trailing dot)
    - Prompt suggests using spaced prefix 'P. P. S.'

    After fix: when marker includes the trailing dot ('P.P.S.'), the checker
    accepts optional spaces and enforces the trailing dot.
    """
    from instructions.ja_instructions import PostscriptChecker

    checker = PostscriptChecker('ja:detectable_content:postscript')
    checker.build_description(postscript_marker='P.P.S.')

    response = '本文\nP. P. S. 追加のメモです。'
    assert checker.check_following(response) is True


def test_postscript_marker_pps_without_dot_spaced_prefix_passes():
    """
    When postscript_marker is 'P.P.S' (no trailing dot), the implementation
    uses a flexible regex allowing optional spaces: '\s*p\.\s?p\.\s?s.*$'.
    Therefore, 'P. P. S.' is accepted and returns True.
    """
    from instructions.ja_instructions import PostscriptChecker

    checker = PostscriptChecker('ja:detectable_content:postscript')
    checker.build_description(postscript_marker='P.P.S')

    response = '本文\nP. P. S. 追加のメモです。'
    assert checker.check_following(response) is True


def test_postscript_marker_ps_with_optional_space_passes():
    """
    For completeness, confirm 'P.S.' branch already allows optional spacing
    between 'P.' and 'S.'
    """
    from instructions.ja_instructions import PostscriptChecker

    checker = PostscriptChecker('ja:detectable_content:postscript')
    checker.build_description(postscript_marker='P.S.')

    response_spaced = '本文\nP. S. 追伸: テストです。'
    response_unspaced = '本文\nP.S. 追伸: テストです。'

    assert checker.check_following(response_spaced) is True
    assert checker.check_following(response_unspaced) is True

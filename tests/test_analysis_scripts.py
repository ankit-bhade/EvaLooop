"""Unit tests for the helpers behind the report numbers in scripts/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_multilingual import asl_pow, renamed_functions, sign_test, summary_language  # noqa: E402

from conftest import MBPP_STYLE_PROMPT  # noqa: E402


def test_asl_pow_matches_leaderboard_formula():
    assert asl_pow([10, 10]) == 10.0
    assert asl_pow([0, 1, 10]) == (0 + 1 + 100) / (3 * 10)


def test_summary_language_ignores_the_dictated_opening():
    assert summary_language("escribe una función de Python que returns the sum of the list") == "english"
    assert summary_language("escribe una función de Python que devuelve la suma de los números") == "spanish"
    assert summary_language("write a python function to return the elements that are in both lists") == "english"


def test_renamed_functions():
    passing = "def similar_elements(a, b):\n    return set(a) & set(b)"
    assert renamed_functions(MBPP_STYLE_PROMPT, passing, passing) == []
    renamed = "def shared_elements(a, b):\n    return set(a) & set(b)"
    assert renamed_functions(MBPP_STYLE_PROMPT, passing, renamed) == ["similar_elements"]


def test_sign_test():
    assert sign_test(0, 0) == 1.0
    assert sign_test(5, 5) == 1.0
    assert round(sign_test(10, 0), 4) == round(2 / 2 ** 10, 4)

"""
Invariant tests for the minbpe regex tokenizer.
Each test pins down a guarantee that must hold for the tokenizer to be correct.
Run: python -m pytest tests/test_tokenizer.py -v
"""
import sys, os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from minbpe.base import get_stats
from minbpe.tokenizer import RegexTokenizer


TRAIN_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "Sphinx of black quartz, judge my vow. "
) * 8


def _trained(vocab_size=300):
    t = RegexTokenizer()
    t.train(TRAIN_TEXT, vocab_size=vocab_size)
    return t


def test_roundtrip_ascii():
    """decode(encode(s)) == s for plain ASCII."""
    t = _trained()
    s = "Hello world! GPT-2 tokenizer test."
    assert t.decode(t.encode(s)) == s


def test_roundtrip_unicode():
    """Round-trip survives emoji, diacritics, and whitespace control chars."""
    t = _trained()
    s = "café 🚀 naïve \nLine\t— résumé"
    assert t.decode(t.encode(s)) == s


def test_train_min_vocab():
    """train() rejects vocab sizes below the 256-byte floor."""
    t = RegexTokenizer()
    with pytest.raises(AssertionError):
        t.train("anything", vocab_size=200)


def test_save_load_roundtrip(tmp_path):
    """save → load reproduces identical encode output.
    Exercises merge() end-to-end; would have caught the i++ SyntaxError."""
    t = _trained()
    s = "the quick brown fox"
    before = t.encode(s)
    prefix = str(tmp_path / "tok")
    t.save(prefix)
    t2 = RegexTokenizer()
    t2.load(prefix + ".model")
    assert t2.encode(s) == before


def test_special_token_encode():
    """Registered specials encode to assigned ids; none_raise rejects stray specials."""
    t = _trained()
    t.register_special_tokens({"<|endoftext|>": 50256})
    ids = t.encode("hello <|endoftext|> world", allowed_special="all")
    assert 50256 in ids
    with pytest.raises(AssertionError):
        t.encode("contains <|endoftext|>", allowed_special="none_raise")


def test_get_stats_accumulates():
    """get_stats(ids, counts) adds into counts instead of replacing it.
    Pins base.py line 4 — broken accumulator made train() count only the last chunk."""
    counts = {}
    get_stats([1, 2, 1, 2], counts)
    get_stats([1, 2], counts)
    assert counts[(1, 2)] == 3

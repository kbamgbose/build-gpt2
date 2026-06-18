# Tokenization Edge Cases

Notes on the minbpe regex tokenizer (`minbpe/base.py`, `minbpe/tokenizer.py`).
Four real bugs hit during the build, plus the edge cases the tokenizer is and is not designed to handle.

The tokenizer follows Karpathy's minbpe: byte-level BPE with a GPT-2 regex pre-tokenizer. It is **not** wired into the training loop (`train.py` still uses `tiktoken.get_encoding("gpt2")`); it exists as a teaching artifact and is exercised only by `tests/test_tokenizer.py`.

---

## Bugs hit

### 1. `i++` in `merge()`

**Symptom:** `SyntaxError` on import. The module never loads, so `train()` is never reached.

**Root cause:** `base.py` had `i++` (C/Java post-increment) instead of `i += 1` in the `else` branch of `merge()`. Python has no `++` operator; the parser fails at module-load time.

**Why it survived:** No test imported `minbpe.base` before the test suite was added. CI only ran the transformer tests.

**Fix:**
```python
else:
    newIds.append(ids[i])
    i += 1
```

**Detection:** Any unit test that imports the module catches this. `tests/test_tokenizer.py::test_roundtrip_ascii` is sufficient.

---

### 2. Pair comparison used `pair[0]` for both positions

**Symptom:** Silent. Training "works" but the learned merges are nonsense. `merge()` only collapses runs of the same byte (`AA`), never an actual pair like `AB`.

**Measurement:** With the bug in place and training on the Programmer's Intro to Unicode text, the top merge was `(101, 101)` (`ee`) with ~3x the count it should have had, while the true GPT-2-style first merge `( , t)` was never collected.

**Root cause:**
```python
if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[0]:
```
The second comparison should be `pair[1]`. As written, `merge((1,2,1,2), (1,2), 99)` returns `[1,2,1,2]` instead of `[99,99]`.

**Fix:** `ids[i+1] == pair[1]`.

**Detection:** `test_save_load_roundtrip` exercises `merge()` end-to-end through training, save, load, and encode. Round-trip equality fails when `merge()` is a no-op for non-repeated pairs.

---

### 3. `get_stats` clobbered its accumulator

**Symptom:** Silent. Wrong merges learned. Hardest of the four to spot.

**Measurement:** `tokenizer.py:train()` regex-splits the corpus into N chunks, then calls `get_stats(chunk_ids, stats)` once per chunk, expecting `stats` to accumulate. With the bug, only the *last chunk's* pair counts survive into `max(stats, key=stats.get)`. On a 10k-character corpus split into ~2000 chunks, that means merges are picked based on counts from a single ~5-character chunk.

**Root cause:**
```python
def get_stats(ids, counts=None):
    counts = {}  # unconditional reset
    ...
```
The signature accepts an accumulator, then throws it away.

**Fix:**
```python
counts = {} if counts is None else counts
```

**Detection:** `test_get_stats_accumulates` directly pins the invariant:
```python
counts = {}
get_stats([1, 2, 1, 2], counts)
get_stats([1, 2], counts)
assert counts[(1, 2)] == 3
```

This bug class (function with optional accumulator that secretly ignores it) is worth a unit test on its own. No integration test would obviously surface it. Training "succeeds," loss drops, downstream perplexity is just quietly worse than it should be.

---

### 4. Default split pattern was GPT-4, not GPT-2

**Symptom:** Silent parity bug. Tokenization differs from `tiktoken.get_encoding("gpt2")` in several systematic ways.

**Root cause:** `RegexTokenizer.__init__` defaulted to `GPT4_SPLIT_PATTERN`. The two patterns differ:

| | GPT-2 | GPT-4 |
|---|---|---|
| Contractions | case-sensitive `'(?:[sdmt]\|ll\|ve\|re)` | case-insensitive `'(?i:[sdmt]\|ll\|ve\|re)` |
| Number runs | unbounded `\p{N}+` | capped at 3 digits `\p{N}{1,3}` |
| Newlines | folded into `\s+` | special-cased: `\s*[\r\n]` |
| Punctuation greed | greedy `+` | possessive `++` |

Using GPT-4 splits in a GPT-2 reproduction breaks parity with any tiktoken baseline and changes the effective vocab distribution.

**Fix:** Default to `GPT2_SPLIT_PATTERN`. Pattern is still overridable via the constructor argument.

**Detection:** Not currently in `tests/test_tokenizer.py`. A parity test against `tiktoken.get_encoding("gpt2")` on a fixed string would catch it, but requires shipping with tiktoken's GPT-2 merges loaded, which is out of scope for the minimal tokenizer.

---

## Edge cases handled

### Whitespace
The GPT-2 regex pre-tokenizer splits on word/number/punctuation/whitespace boundaries. Leading-space-on-word is preserved as part of the same chunk: `" the"` becomes one chunk, distinct from `"the"`. Trailing whitespace before newlines is handled by `\s+(?!\S)`.

### Unknown bytes
Two layers:
- **Encode:** Byte-level BPE has no OOV. Any UTF-8-encodable string maps to a sequence of bytes (256 base tokens) plus learned merges. Specials are gated by `allowed_special`. The `"none_raise"` setting (the tiktoken default) raises if a registered special string appears in the input, preventing accidental encoding of user-supplied `<|endoftext|>` substrings.
- **Decode:** `b"".join(...).decode("utf-8", errors="replace")` recovers gracefully from an id sequence that splits a multi-byte UTF-8 codepoint. The replacement character `U+FFFD` is emitted in place of any incomplete sequence.

### Unicode
The `regex` package (not `re`) is required because the GPT-2 pattern uses Unicode property classes `\p{L}` and `\p{N}`. The stdlib `re` does not support these. `requirements.txt` should pin `regex` for any deployment.

Emoji and diacritics round-trip correctly because they are encoded byte-by-byte before BPE sees them. See `test_roundtrip_unicode` for the regression test.

---

## Edge cases not handled

- **GPT-2 tiktoken parity.** This tokenizer trains its own merges. It does not load OpenAI's GPT-2 merge table, so tokens it produces are not interchangeable with `tiktoken.get_encoding("gpt2")`. Loss curves against published GPT-2 baselines are not comparable.
- **Code blocks at scale.** The GPT-2 regex was tuned for natural language. Long runs of indentation, repeated `===` separators, and dense punctuation in source code produce verbose, low-frequency token sequences. GPT-4's pattern (with `\s*[\r\n]` and capped digit runs) is meaningfully better for code; consider passing `GPT4_SPLIT_PATTERN` explicitly when training a code-focused tokenizer.
- **Normalization.** No NFC/NFD normalization is applied. `"café"` (precomposed) and `"café"` (decomposed) hash to different byte sequences and therefore different token ids. If the corpus mixes both, the tokenizer will learn duplicate merges for visually identical strings.
- **Byte order marks.** A leading `U+FEFF` is treated as a normal codepoint and will consume an early merge slot if frequent in the corpus. Strip BOMs at the dataset layer.

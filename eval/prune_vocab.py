#!/usr/bin/env python3
"""
Prune the Omnilingual CTC vocabulary down to the ten required languages.

WHY THIS EXISTS
---------------
The Omnilingual ASR CTC model is a 1,600+ language character-level CTC model with **no
language parameter at all**. Its output layer is a single matrix over ~10k characters, and
greedy search is free to pick any of them, in any script, on any frame.

That produces a failure we measured on 20 September 2026 and again on 26 September 2026:
the model swaps a character for a *phonetically identical* character from an unrelated
script. On CLEAN audio, with no noise whatsoever:

    "hello"        -> "هello"      (Arabic heh for Latin h)
    "hi"           -> "हय"         (Devanagari ya for Latin y)
    "help me"      -> "help مe"     (Arabic meem for Latin m)
    "are you okay" -> "اr yوk"      (Arabic alif, Arabic waw)

16% of short English utterances produced non-Latin output. The LanguageFilter then scored
those at 0.60-0.83, above its 0.60 threshold, SENT them, and silently deleted the foreign
characters - so the receiving phone displayed "ello" and "help e". For an app whose entire
purpose is emergency text, that is the worst available failure: confident, unreadable text.

We could not fix this by tuning the filter, because the filter only runs *after* the model
has already chosen the wrong character. The filter can drop a message; it cannot repair one.

THE FIX
-------
Delete the output columns for every character outside the nine scripts the ten required
languages actually use. After pruning, the model **cannot emit** those characters - not
"usually avoids them", cannot. If the acoustic evidence is ambiguous between Latin "h" and
Arabic heh, the best *remaining* candidate wins, so the output degrades toward a readable
in-language guess instead of an unreadable foreign one.

The cost is small: the CTC head is ~3% of the file, so 10,288 -> 882 tokens saves about
9 MB. The point was never the size. The point is that the whole wrong-script failure
*class* stops existing, which also fixes the Hindi script-retention collapse documented in
NOISE_ROBUSTNESS_FINDINGS.md.

WHAT IS SLICED
--------------
The quantised head is a single MatMulInteger:

    Cast -> MatMulInteger(x_q[*,1024], W_int8[1024,V], x_zp, W_zp) -> int32
         -> Mul(W_scale) -> Add(final_proj.bias[V]) -> logits

`W_scale` and `W_zp` are SCALARS in this export (dims == []), so there is no per-column
scale vector to reslice. Only three things change:

    W_int8[1024, V]   ->  W_int8[1024, K]
    bias[V]           ->  bias[K]
    logits shape      ->  [N, num_frames, K]

plus a rewritten tokens.txt with contiguous indices 0..K-1.

USAGE
-----
    python eval/prune_vocab.py \
        --src models/asr-v2/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05 \
        --dst models/asr-v3-10lang/sherpa-onnx-omnilingual-asr-10lang-300M-ctc-int8

Then verify with:

    python eval/asr_run.py --model models/asr-v3-10lang/... --tokens .../tokens.txt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    import onnx
    from onnx import numpy_helper
except ImportError:  # pragma: no cover
    sys.exit("This script needs `onnx`. Run it with the project venv: .venv/bin/python")

# The nine scripts that cover all ten required languages. These mirror the ALLOWED_RANGES in
# app/app/src/main/java/com/bjp/itantra/LanguageFilter.kt - the two MUST stay in sync, because
# this tool decides what the model is even capable of emitting and the filter decides what is
# allowed through. If you widen one, widen the other.
#
#   Latin      -> English
#   Devanagari -> Hindi, Marathi
#   Bengali    -> Bengali
#   Gujarati   -> Gujarati
#   Oriya      -> Odia
#   Tamil      -> Tamil
#   Telugu     -> Telugu
#   Kannada    -> Kannada
#   Malayalam  -> Malayalam
SCRIPT_RANGES: list[tuple[int, int]] = [
    (0x0041, 0x005A),  # Latin A-Z
    (0x0061, 0x007A),  # Latin a-z
    (0x00C0, 0x024F),  # Latin-1 supplement + Latin extended-A/B
    (0x1E00, 0x1EFF),  # Latin extended additional
    (0x0900, 0x097F),  # Devanagari
    (0xA8E0, 0xA8FF),  # Devanagari extended
    (0x0980, 0x09FF),  # Bengali
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Oriya / Odia
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
]

# Always kept regardless of script.
#  - the four special tokens. CRITICAL: <unk> is the CTC BLANK in this tokenizer. Drop it and the
#    decoder cannot collapse repeated frames, so it emits a character for EVERY frame and the
#    output degenerates into something like "w-hellow-'-" instead of "hello". Do not "simplify"
#    this line into set("<s><pad></s><unk>") - that builds a set of single CHARACTERS, not of
#    these four strings, and silently drops all four. That mistake cost a debugging cycle.
#  - the space character, because the model emits it as a real token and the TTS needs it to
#    separate words.
#  - ASCII digits and common punctuation, because real emergency traffic contains them
#    ("call 112", "3rd floor", "don't wait").
ALWAYS_KEEP = (
    {"<s>", "<pad>", "</s>", "<unk>"}
    | set(" 0123456789.,!?;:'\"-()/%&#@+*=_")
)

# Hard post-condition. <unk> is the CTC blank; without it the decode is garbage. Checking this
# explicitly here is cheap insurance against a future edit to ALWAYS_KEEP quietly re-breaking it.
REQUIRED_TOKENS = ("<s>", "<pad>", "</s>", "<unk>", " ")

# Names of the tensors this tool rewrites. Asserted to exist so a future export with different
# naming fails loudly instead of silently producing a model that decodes to noise.
WEIGHT_NAME = "onnx::MatMul_4143_quantized"
BIAS_NAME = "model.final_proj.bias"
SCALE_NAME = "onnx::MatMul_4143_scale"
ZP_NAME = "onnx::MatMul_4143_zero_point"


def in_target_script(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in SCRIPT_RANGES)


def load_tokens(path: Path, expect_vocab: int) -> list[tuple[str, int]]:
    """
    Read tokens.txt as (symbol, id), returned sorted BY ID.

    The id is the model's own output-column index. The critical invariant this function enforces
    is that ids form exactly 0..expect_vocab-1 with no gaps and no duplicates. If that does not
    hold, file order is NOT column order, and every column we kept would be misaligned with the
    symbol we wrote into tokens.txt - producing a model that decodes to fluent-looking garbage.
    Failing loudly here is much cheaper than debugging that.
    """
    out: list[tuple[str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        # Format is "<symbol> <id>". Symbols can contain spaces, so split from the RIGHT.
        sym, _, idx = line.rpartition(" ")
        if not sym or not idx.isdigit():
            raise ValueError(f"malformed tokens.txt line: {line!r}")
        out.append((sym, int(idx)))

    ids = [i for _, i in out]
    if len(ids) != expect_vocab:
        sys.exit(f"tokens.txt has {len(ids)} entries but the head has {expect_vocab} columns")
    if len(set(ids)) != len(ids):
        sys.exit("tokens.txt contains duplicate ids - cannot map ids to head columns safely")
    if sorted(ids) != list(range(expect_vocab)):
        missing = sorted(set(range(expect_vocab)) - set(ids))
        sys.exit(
            "tokens.txt ids are not contiguous 0..V-1, so file order is not column order.\n"
            f"first missing ids: {missing[:10]}\n"
            "Refusing to guess the mapping. Re-export tokens.txt in id order."
        )

    out.sort(key=lambda t: t[1])
    return out


def choose_keep(tokens: list[tuple[str, int]]) -> tuple[list[tuple[str, int]], list[str]]:
    """
    Return (kept_as_symbol_and_ORIGINAL_ID, dropped_symbols).

    Keeping the original id alongside the symbol is essential: it is the head column to slice.
    Dropping it and reconstructing columns from position would reintroduce exactly the
    file-order-is-column-order assumption we just removed.
    """
    keep: list[tuple[str, int]] = []
    dropped: list[str] = []
    for sym, orig_id in tokens:
        # len(sym) == 1 guards against multi-codepoint entries sneaking through.
        keepable = sym in ALWAYS_KEEP or (len(sym) == 1 and in_target_script(sym))
        if keepable:
            keep.append((sym, orig_id))
        else:
            dropped.append(sym)
    return keep, dropped


def prune_model(
    src_model: Path,
    dst_model: Path,
    keep: list[tuple[str, int]],
    verbose: bool = True,
) -> None:
    """Slice the CTC head down to the columns named by `keep`. Writes only to dst."""
    model = onnx.load(str(src_model))
    graph = model.graph
    init = {t.name: t for t in graph.initializer}

    for required in (WEIGHT_NAME, BIAS_NAME, SCALE_NAME, ZP_NAME):
        if required not in init:
            sys.exit(
                f"FATAL: expected tensor {required!r} is not in the graph.\n"
                f"Found head-like tensors: "
                f"{[n for n in init if 'MatMul' in n and n.endswith('quantized')][:8]}\n"
                f"This script targets the onnx.quantize export layout. Re-check the export."
            )

    weight = numpy_helper.to_array(init[WEIGHT_NAME])  # int8 [1024, V]
    bias = numpy_helper.to_array(init[BIAS_NAME])      # float32 [V]
    scale = numpy_helper.to_array(init[SCALE_NAME])
    zp = numpy_helper.to_array(init[ZP_NAME])

    if weight.shape[0] != 1024:
        sys.exit(f"unexpected head input dim {weight.shape[0]}, expected 1024")
    vocab = weight.shape[1]
    if bias.shape[0] != vocab:
        sys.exit(f"vocab mismatch: head has {vocab} columns but bias has {bias.shape[0]}")
    if len(keep) > vocab:
        sys.exit(f"keep list ({len(keep)}) is larger than the vocab ({vocab})")

    # The export uses one scalar scale/zero-point for the whole head. Assert it, because if a
    # future export emits per-column scales this silent slice would be WRONG (each surviving
    # column would keep the wrong scale) and the model would decode to plausible garbage.
    if scale.ndim != 0 or zp.ndim != 0:
        sys.exit(
            "this export has per-column scale/zero_point, which this script does not handle.\n"
            "Slicing weights without reslicing scales would be silently incorrect."
        )

    # These are the ORIGINAL head column indices, straight from tokens.txt.
    columns = [orig_id for _, orig_id in keep]
    if len(set(columns)) != len(columns):
        sys.exit("duplicate column indices in keep list")
    if min(columns) < 0 or max(columns) >= vocab:
        sys.exit(f"column index out of range: min={min(columns)} max={max(columns)} vocab={vocab}")

    new_weight = weight[:, columns].copy()
    new_bias = bias[columns].copy()

    # Protobuf initializers are immutable views; replace the whole tensor.
    for t in graph.initializer:
        if t.name == WEIGHT_NAME:
            t.CopyFrom(numpy_helper.from_array(new_weight, WEIGHT_NAME))
        elif t.name == BIAS_NAME:
            t.CopyFrom(numpy_helper.from_array(new_bias, BIAS_NAME))

    # Graph output rank stays 3; only the last dim shrinks.
    for out in graph.output:
        dims = out.type.tensor_type.shape.dim
        if len(dims) == 3 and dims[2].dim_value == vocab:
            dims[2].Clear()
            dims[2].dim_value = len(keep)

    dst_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst_model))
    if verbose:
        print(f"  head  {list(weight.shape)} -> {list(new_weight.shape)}")
        print(f"  bias  {list(bias.shape)} -> {list(new_bias.shape)}")
        print(f"  scale {scale.item()!r} (scalar, unchanged)")
        print(f"  zero  {zp.item()!r} (scalar, unchanged)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--src", required=True, type=Path, help="source model directory")
    ap.add_argument("--dst", required=True, type=Path, help="output model directory")
    ap.add_argument(
        "--src-model",
        default="model.int8.onnx",
        help="model filename inside --src (default: model.int8.onnx)",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    src, dst = args.src, args.dst
    verbose = not args.quiet

    if not src.is_dir():
        sys.exit(f"source directory not found: {src}")

    src_model = src / args.src_model
    tokens_txt = src / "tokens.txt"
    for p in (src_model, tokens_txt):
        if not p.is_file():
            sys.exit(f"missing required file: {p}")

    if verbose:
        print(f"source : {src}")
        print(f"output : {dst}")

    # Read the head's vocabulary dimension first so load_tokens can assert the two agree.
    probe = onnx.load(str(src_model), load_external_data=False)
    init_names = {t.name: t for t in probe.graph.initializer}
    if WEIGHT_NAME not in init_names:
        sys.exit(f"{src_model} has no {WEIGHT_NAME}; cannot determine vocab size")
    vocab = init_names[WEIGHT_NAME].dims[1]

    tokens = load_tokens(tokens_txt, expect_vocab=vocab)
    keep, dropped = choose_keep(tokens)

    if verbose:
        print(f"\ntokens : {len(tokens)} total -> {len(keep)} kept, {len(dropped)} dropped")
        buckets: dict[str, int] = {}
        for sym in dropped:
            cp = ord(sym[0]) if sym else 0
            name = "other"
            for label, lo, hi in [
                ("CJK", 0x4E00, 0x9FFF), ("CJK-ext", 0x3400, 0x4DBF),
                ("Hangul", 0xAC00, 0xD7AF), ("Ethiopic", 0x1200, 0x137F),
                ("Arabic", 0x0600, 0x06FF), ("Cyrillic", 0x0400, 0x04FF),
                ("Greek", 0x0370, 0x03FF), ("Thaana", 0x0780, 0x07BF),
            ]:
                if lo <= cp <= hi:
                    name = label
                    break
            buckets[name] = buckets.get(name, 0) + 1
        for name, n in sorted(buckets.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    dropped {name:<10} {n:>6}")

    # Integrity: kept symbols must be unique, or renumbering would create duplicate ids and the
    # decoder would map some ids to two different characters depending on iteration order.
    kept_syms = [s for s, _ in keep]
    if len(set(kept_syms)) != len(kept_syms):
        dupes = {s for s in kept_syms if kept_syms.count(s) > 1}
        sys.exit(f"tokens.txt contains duplicate kept symbols: {sorted(dupes)[:10]}")

    # The blank is what makes CTC collapse repeated frames. Its absence is not a small accuracy
    # loss, it is a total decode failure, so refuse to write a model without it.
    missing_req = [t for t in REQUIRED_TOKENS if t not in kept_syms]
    if missing_req:
        sys.exit(
            f"refusing to write: required token(s) {missing_req} would be dropped.\n"
            f"<unk> is the CTC blank. Without it the model emits a character per frame and the\n"
            f"output is unreadable. This is a bug in ALWAYS_KEEP, not in the source model."
        )

    prune_model(src_model, dst / args.src_model, keep, verbose=verbose)

    # Renumber 0..K-1 in the SAME order as the sliced columns. sherpa-onnx builds its symbol
    # table from this file indexed by id, so id i must be the symbol whose column we kept at i.
    out_lines = [f"{sym} {i}" for i, (sym, _) in enumerate(keep)]
    (dst / "tokens.txt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    for extra in ("LICENSE", "README.md", "MODEL_CARD"):
        s = src / extra
        if s.is_file():
            shutil.copy2(s, dst / extra)

    if verbose:
        mb = (dst / args.src_model).stat().st_size / 1048576
        before = src_model.stat().st_size / 1048576
        print(f"\nwrote {dst / args.src_model}  {before:.1f} MB -> {mb:.1f} MB")
        print(f"wrote {dst / 'tokens.txt'}  {len(keep)} symbols")
        print("\nThe model can no longer emit any character outside the nine target scripts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

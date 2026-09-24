#!/usr/bin/env python3
"""Compute WER before/after fine-tuning on a held-out test manifest.

Evaluates the base (pretrained) ai4bharat IndicConformer hybrid CTC-RNNT
model and, optionally, a fine-tuned checkpoint, on the same NeMo-format
test manifest, and writes a before/after WER comparison to JSON.
"""

import argparse
import json
import os
import sys

import torch

# data/text_normalize_mr.py lives one directory up from this script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "data"))
from text_normalize_mr import normalize_marathi  # noqa: E402

import jiwer
import nemo.collections.asr as nemo_asr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate WER of the base and fine-tuned Marathi ASR models."
    )
    parser.add_argument(
        "--base-model",
        default="ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large",
        help="Hugging Face model id or path to a .nemo checkpoint for the base/pretrained model.",
    )
    parser.add_argument(
        "--finetuned-model",
        default=None,
        help="Path to a fine-tuned .nemo checkpoint. If omitted, only the base model is evaluated.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to a NeMo-format test manifest JSONL (audio_filepath, duration, text).",
    )
    parser.add_argument(
        "--output",
        default="eval_wer.json",
        help="Path to write the JSON results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Transcription batch size (kept small by default for limited GPU memory).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run inference on.",
    )
    return parser.parse_args()


def load_manifest(manifest_path):
    audio_filepaths = []
    references = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            audio_filepaths.append(entry["audio_filepath"])
            references.append(entry["text"])
    return audio_filepaths, references


def load_model(model_ref, device):
    """model_ref is either a HF model id or a path to a local .nemo checkpoint."""
    if model_ref.endswith(".nemo") and os.path.isfile(model_ref):
        model = nemo_asr.models.ASRModel.restore_from(restore_path=model_ref)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_ref)

    model = model.to(device)
    model.eval()
    model.freeze()
    return model


def extract_text(hypothesis):
    """NeMo's transcribe() returns plain strings in some versions and
    Hypothesis objects (exposing .text) in others -- handle both explicitly
    rather than silently stringifying the wrong thing."""
    if isinstance(hypothesis, str):
        return hypothesis
    if hasattr(hypothesis, "text"):
        return hypothesis.text
    raise TypeError(
        f"Unexpected transcription result type {type(hypothesis)!r}; "
        "expected a str or an object exposing a .text attribute."
    )


def transcribe(model, audio_filepaths, batch_size):
    # language_id is required: this checkpoint uses a multilingual tokenizer
    # with per-language output masks, and omitting it doesn't reliably
    # default to the right language.
    try:
        raw_outputs = model.transcribe(audio=audio_filepaths, batch_size=batch_size, language_id="mr")
    except TypeError:
        raw_outputs = model.transcribe(audio=audio_filepaths, batch_size=batch_size)

    # Hybrid CTC-RNNT models can return (best_hyps, all_hyps); flatten.
    if isinstance(raw_outputs, tuple):
        raw_outputs = raw_outputs[0]

    hypotheses = [extract_text(h) for h in raw_outputs]
    if len(hypotheses) != len(audio_filepaths):
        raise RuntimeError(
            f"transcribe() returned {len(hypotheses)} results for {len(audio_filepaths)} inputs."
        )
    return hypotheses


def compute_wer(model, audio_filepaths, references, batch_size):
    hypotheses = transcribe(model, audio_filepaths, batch_size)
    norm_references = [normalize_marathi(r) for r in references]
    norm_hypotheses = [normalize_marathi(h) for h in hypotheses]
    return jiwer.wer(norm_references, norm_hypotheses)


def main():
    args = parse_args()
    audio_filepaths, references = load_manifest(args.manifest)
    if not audio_filepaths:
        raise SystemExit(f"No samples found in manifest: {args.manifest}")

    results = {
        "manifest": args.manifest,
        "num_samples": len(audio_filepaths),
    }

    print(f"Loading base model: {args.base_model}")
    base_model = load_model(args.base_model, args.device)
    results["base_model"] = {
        "wer": compute_wer(base_model, audio_filepaths, references, args.batch_size),
        "model": args.base_model,
    }
    del base_model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    if args.finetuned_model:
        print(f"Loading fine-tuned model: {args.finetuned_model}")
        finetuned_model = load_model(args.finetuned_model, args.device)
        results["finetuned_model"] = {
            "wer": compute_wer(finetuned_model, audio_filepaths, references, args.batch_size),
            "model": args.finetuned_model,
        }
        del finetuned_model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults written to {args.output}")
    print(
        f"Base model WER     : {results['base_model']['wer']:.4f}  ({results['base_model']['model']})"
    )
    if "finetuned_model" in results:
        print(
            f"Fine-tuned WER     : {results['finetuned_model']['wer']:.4f}  "
            f"({results['finetuned_model']['model']})"
        )
        delta = results["base_model"]["wer"] - results["finetuned_model"]["wer"]
        print(f"Absolute WER improvement: {delta:.4f}")


if __name__ == "__main__":
    main()

# Marathi ASR Fine-tuning (IndicConformer)

Fine-tuning `ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large` — a Conformer-Large (17-layer, 512-dim, 129M param) hybrid CTC-RNNT speech-to-text model — on Marathi speech, end-to-end on a single shared GPU.

## Results

All numbers are WER on the same held-out **79-sample test split** (never used for training or validation):

| Model | Test WER |
| --- | --- |
| Base checkpoint (no fine-tuning) | 0.1540 |
| **Full fine-tune** (all 129M params, 1 epoch) | **0.1009** |
| Heads-only fine-tune (encoder frozen, CTC+RNNT decoder only) | 0.1215 |

Both fine-tuning strategies improve meaningfully over the base model. Full fine-tuning wins by ~2 points over heads-only — expected, since the 115M-param Conformer encoder has far more capacity to adapt to the new data's acoustics, at the cost of updating far more parameters. Freezing the encoder did **not** make training meaningfully faster in practice — the actual bottleneck was the loss function, not encoder compute.

## What was done

1. **Model**: restored the pretrained checkpoint as-is (encoder, decoder, joint, and its 22-language shared tokenizer all untouched) and fine-tuned with a small LR (2e-5) to avoid a vocab/architecture mismatch against the pretrained weights.
2. **Dataset**: [OpenSLR SLR64](https://openslr.org/64/) (~3 hours Marathi speech, CC BY-SA 4.0) — the two originally-planned datasets (Common Voice, then Kathbath) turned out to be inaccessible without a human clicking through an account gate; SLR64 was the one that actually worked headlessly. Built our own 90/5/5 train/dev/test split (1,412 / 78 / 79 samples) since none shipped with the source.
3. **Training**: NeMo + PyTorch Lightning (`scripts/finetune.py`):
   - Full fine-tune, all params, 1 completed epoch (`artifacts/run_20260923_052950/`)
   - Heads-only ablation, encoder frozen, best of 2 completed epochs (`artifacts/run_headsonly_20260923_071352/`)
4. **Evaluation**: `scripts/infer_eval.py`, base vs. fine-tuned WER on the same test manifest.

## Repo layout

```
requirements.txt, setup.sh       # environment setup (conda env, NeMo fork, pinned deps)
configs/                          # fine-tuning config (train_ds/optim overrides only)
data/                             # dataset download + manifest prep, text normalization
scripts/
  smoke_test.sh                   # tiny dry run before committing to a real training run
  finetune.py                     # training entrypoint (supports --freeze-encoder, --encoder-lr)
  infer_eval.py                   # before/after WER
  package_artifacts.sh            # tar up one run's artifacts
artifacts/<run_id>/                # gitignored; checkpoint.nemo, train.log, config_used.yaml, eval_wer.json
artifacts/<run_id>.tar.gz          # share-ready packaged output of a run
```

## Reproducing

```bash
bash setup.sh                                    # one-time environment setup
bash scripts/smoke_test.sh                       # cheap pipeline sanity check
python data/prepare_manifest.py                  # full dataset -> manifests
python scripts/finetune.py \
  --train-manifest data/processed/train_manifest.jsonl \
  --val-manifest data/processed/dev_manifest.jsonl \
  --run-id my_run --max-epochs 2 --batch-size 2
python scripts/infer_eval.py \
  --base-model models/indicconformer_stt_mr_hybrid_rnnt_large.nemo \
  --finetuned-model artifacts/my_run/checkpoint.nemo \
  --manifest data/processed/test_manifest.jsonl
bash scripts/package_artifacts.sh my_run
```

## Known limitations

- **Single-speaker-gender source data** (SLR64 has female-only speakers) means the WER improvement is somewhat speaker/gender-skewed rather than broadly representative.

## Artifacts

The best checkpoint (full fine-tune, WER 0.1009) is packaged as a share-ready tarball
(checkpoint + training log + resolved config + eval results) via `scripts/package_artifacts.sh`.
The heads-only run above is an internal comparison, not separately distributed.
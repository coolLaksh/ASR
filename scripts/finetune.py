#!/usr/bin/env python3
"""Fine-tune ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large on Marathi.

Restores the pretrained checkpoint as-is and only overrides
train_ds/validation_ds/optim before training.
"""

import argparse
import os
import shutil

import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.callbacks import Callback

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.utils.exp_manager import exp_manager


def use_pytorch_rnnt_loss(model, lang):
    """Swap the checkpoint's numba-JIT RNNT loss for NeMo's pure-PyTorch
    backend (the numba kernels are incompatible with this machine's CUDA
    driver). num_classes must be the per-language vocab size, not the
    aggregate: the joint already masks its output to one language's slice
    before the loss sees it, so the aggregate count gives an out-of-bounds
    blank index.
    """
    num_classes = sum(model.joint.language_masks[lang]) - 1
    model.loss = RNNTLoss(num_classes=num_classes, reduction=model.loss.reduction, loss_name="pytorch")
    if model.joint.fuse_loss_wer:
        model.joint.set_loss(model.loss)


def configure_encoder_training(model, freeze_encoder, encoder_lr):
    """Ablation hooks: full fine-tune (neither flag), frozen encoder
    (heads only), or discriminative LR (encoder slower than the heads).
    """
    if freeze_encoder and encoder_lr is not None:
        raise ValueError("--freeze-encoder and --encoder-lr are alternative ablations; pass at most one.")

    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("Encoder frozen: only decoder/joint/ctc_decoder heads will be trained.")

    if encoder_lr is not None:
        with open_dict(model.cfg):
            model.cfg.optim_param_groups = {"encoder": {"lr": encoder_lr}}
        print(f"Discriminative LR: encoder lr={encoder_lr}, other params use --lr as normal.")


class SaveNemoEachEpoch(Callback):
    """Writes a ready-to-use .nemo checkpoint after every epoch, so an
    interrupted run doesn't need manual recovery from the PTL .ckpt.
    """

    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        pl_module.save_to(self.checkpoint_path)
        print(f"[SaveNemoEachEpoch] checkpoint updated after epoch {trainer.current_epoch}: {self.checkpoint_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/finetune_indicconformer_mr.yaml")
    parser.add_argument(
        "--checkpoint",
        default="models/indicconformer_stt_mr_hybrid_rnnt_large.nemo",
        help="Pretrained .nemo checkpoint to restore and fine-tune from.",
    )
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--lang", default="mr", help="Language key into the checkpoint's AggregateTokenizer.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Smoke-test override: cap total optimizer steps instead of training by epoch.",
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Ablation: freeze the Conformer encoder, train only the CTC/RNNT decoder heads.",
    )
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=None,
        help="Ablation: discriminative LR -- train the encoder at this (lower) rate, "
        "everything else at --lr. Alternative to --freeze-encoder, not combinable with it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = os.path.join(args.artifacts_dir, args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    cfg = OmegaConf.load(args.config)

    cfg.model.train_ds.manifest_filepath = args.train_manifest
    cfg.model.validation_ds.manifest_filepath = args.val_manifest
    if args.batch_size is not None:
        cfg.model.train_ds.batch_size = args.batch_size
        cfg.model.validation_ds.batch_size = args.batch_size
    if args.lr is not None:
        cfg.model.optim.lr = args.lr

    if args.max_steps is not None:
        # max_epochs must be -1 for max_steps to be the stopping criterion.
        cfg.trainer.max_steps = args.max_steps
        cfg.trainer.max_epochs = -1
        cfg.trainer.val_check_interval = 1.0
        cfg.trainer.num_sanity_val_steps = 0
    elif args.max_epochs is not None:
        cfg.trainer.max_epochs = args.max_epochs

    cfg.exp_manager.exp_dir = os.path.join(run_dir, "nemo_experiments")
    cfg.exp_manager.name = args.run_id

    checkpoint_path = os.path.join(run_dir, "checkpoint.nemo")
    OmegaConf.save(cfg, os.path.join(run_dir, "config_used.yaml"))

    trainer = pl.Trainer(**cfg.trainer, callbacks=[SaveNemoEachEpoch(checkpoint_path)])
    log_dir = exp_manager(trainer, cfg.exp_manager)

    print(f"Restoring pretrained checkpoint: {args.checkpoint}")
    model = nemo_asr.models.ASRModel.restore_from(restore_path=args.checkpoint, trainer=trainer)
    use_pytorch_rnnt_loss(model, args.lang)
    configure_encoder_training(model, args.freeze_encoder, args.encoder_lr)

    model.setup_training_data(cfg.model.train_ds)
    model.setup_multiple_validation_data(cfg.model.validation_ds)
    model.setup_optimization(optim_config=cfg.model.optim)

    print(f"Starting training (run_id={args.run_id})")
    trainer.fit(model)

    model.save_to(checkpoint_path)
    print(f"Saved final fine-tuned checkpoint: {checkpoint_path}")

    if log_dir is not None:
        log_src = os.path.join(str(log_dir), "nemo_log_globalrank-0_localrank-0.txt")
        log_dst = os.path.join(run_dir, "train.log")
        if os.path.isfile(log_src):
            shutil.copyfile(log_src, log_dst)
            print(f"Copied training log: {log_dst}")
        else:
            print(f"Warning: expected NeMo log file not found at {log_src}")


if __name__ == "__main__":
    main()

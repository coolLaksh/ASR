"""
Build NeMo-format train/dev/test manifests from OpenSLR SLR64
(https://openslr.trmal.net/resources/64/, CC BY-SA 4.0).

SLR64 ships no official split, so this creates a fixed-seed random 90/5/5
split. The zip's internal layout isn't documented, so wav files are indexed
by basename via os.walk rather than assuming a fixed relative path.
"""

import argparse
import json
import logging
import os
import random
import sys
import zipfile

import librosa
import requests
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_normalize_mr import normalize_marathi

ZIP_URL = "https://openslr.trmal.net/resources/64/mr_in_female.zip"
TSV_URL = "https://openslr.trmal.net/resources/64/line_index.tsv"

TRAIN_FRAC = 0.90
DEV_FRAC = 0.05


def download_file(url: str, dest_path: str) -> None:
    if os.path.exists(dest_path):
        logging.info(f"already downloaded, skipping: {dest_path}")
        return
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    tmp_path = dest_path + ".part"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=os.path.basename(dest_path)
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    os.rename(tmp_path, dest_path)


def extract_zip(zip_path: str, extract_dir: str) -> None:
    marker = os.path.join(extract_dir, ".extracted")
    if os.path.exists(marker):
        logging.info(f"already extracted, skipping: {extract_dir}")
        return
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    with open(marker, "w") as f:
        f.write("done")


def build_wav_index(root_dir: str) -> dict:
    index = {}
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".wav"):
                file_id = os.path.splitext(fname)[0]
                index[file_id] = os.path.join(dirpath, fname)
    return index


def load_transcripts(tsv_path: str) -> list:
    pairs = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                logging.warning(f"malformed tsv line, skipping: {line!r}")
                continue
            file_id, text = parts
            pairs.append((file_id.strip(), text))
    return pairs


def process_entry(file_id: str, raw_text: str, wav_index: dict, wav_out_dir: str):
    src_path = wav_index.get(file_id)
    if src_path is None:
        return None, f"no audio file found for id={file_id}"

    try:
        audio, sr = librosa.load(src_path, sr=16000, mono=True)
    except Exception as e:
        return None, f"failed to load {src_path}: {e}"

    if audio.size == 0:
        return None, f"empty audio: {src_path}"

    text = normalize_marathi(raw_text)
    if not text:
        return None, f"empty text after normalization for id={file_id}"

    out_path = os.path.join(wav_out_dir, f"{file_id}.wav")
    try:
        sf.write(out_path, audio, sr, subtype="PCM_16")
    except Exception as e:
        return None, f"failed to write {out_path}: {e}"

    duration = len(audio) / sr
    return {
        "audio_filepath": os.path.abspath(out_path),
        "duration": round(float(duration), 3),
        "text": text,
        # Required: the base checkpoint's multilingual tokenizer routes text
        # and output masking per-language via this manifest field.
        "lang": "mr",
    }, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download OpenSLR SLR64 and build NeMo train/dev/test manifests."
    )
    parser.add_argument("--data-dir", default="data/raw/slr64")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit", type=int, default=None, help="only process the first N tsv entries (smoke test)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    zip_path = os.path.join(args.data_dir, "mr_in_female.zip")
    tsv_path = os.path.join(args.data_dir, "line_index.tsv")
    extract_dir = os.path.join(args.data_dir, "extracted")

    download_file(ZIP_URL, zip_path)
    download_file(TSV_URL, tsv_path)
    extract_zip(zip_path, extract_dir)

    pairs = load_transcripts(tsv_path)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    logging.info(f"loaded {len(pairs)} transcript entries (limit={args.limit})")

    wav_index = build_wav_index(extract_dir)
    logging.info(f"indexed {len(wav_index)} wav files under {extract_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    wav_out_dir = os.path.join(args.output_dir, "wavs")
    os.makedirs(wav_out_dir, exist_ok=True)

    processed = []
    skipped = 0
    for file_id, raw_text in tqdm(pairs, desc="resampling+normalizing"):
        entry, err = process_entry(file_id, raw_text, wav_index, wav_out_dir)
        if entry is None:
            skipped += 1
            logging.warning(err)
            continue
        processed.append(entry)

    random.Random(args.seed).shuffle(processed)

    n = len(processed)
    n_train = round(n * TRAIN_FRAC)
    n_dev = round(n * DEV_FRAC)
    n_test = n - n_train - n_dev

    splits = {
        "train": processed[:n_train],
        "dev": processed[n_train : n_train + n_dev],
        "test": processed[n_train + n_dev :],
    }
    assert sum(len(v) for v in splits.values()) == n
    assert n_test >= 0

    for split_name, entries in splits.items():
        manifest_path = os.path.join(args.output_dir, f"{split_name}_manifest.jsonl")
        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        hours = sum(e["duration"] for e in entries) / 3600
        print(f"{split_name}: {len(entries)} samples, {hours:.2f}h -> {manifest_path}")

    print(f"skipped: {skipped} files (missing/corrupt audio or empty text after normalization)")


if __name__ == "__main__":
    main()

import argparse
import string
import json
import re
import os
from whisper.normalizers import EnglishTextNormalizer
from tqdm import tqdm


PUNCTUATION_RE = re.compile(f"[{re.escape(string.punctuation)}]")
ARTICLES_RE = re.compile(r"\b(a|an|the|and)\b", re.IGNORECASE)
SPECIAL_TOKEN_RE = re.compile(r"<\|.*?\|>")
WORD_BOUNDARY = r"\b{}\b"
NORMALIZER = EnglishTextNormalizer()

def normalize_text(text: str) -> str:
    """
    Apply lowercasing, remove special tokens, punctuation, articles, collapse whitespace, then Whisper normalize.
    """
    text = text.lower()
    text = SPECIAL_TOKEN_RE.sub("", text)
    text = PUNCTUATION_RE.sub(" ", text)
    text = ARTICLES_RE.sub(" ", text)
    text = " ".join(text.split())
    return NORMALIZER(text)


def exact_match(pred: str, gt: str) -> bool:
    """Return True if normalized prediction exactly equals normalized ground truth."""
    return normalize_text(pred) == normalize_text(gt)


def exist_match(pred: str, gt: str) -> bool:
    """Return True if each comma-separated normalized part of gt appears in normalized pred as a whole word."""
    pred_norm = normalize_text(pred)
    parts = [normalize_text(p.strip()) for p in gt.split(",") if p.strip()]
    if not parts or (len(parts) == 1 and parts[0] == ""):
        return False
    for part in parts:
        if not re.search(WORD_BOUNDARY.format(re.escape(part)), pred_norm):
            return False
    return True


def read_tsv(path: str) -> dict:
    """Read TSV file into a dictionary: {key: value}"""
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            key, val = parts
            data[key] = val
    return data


def read_jsonl(path: str) -> tuple:
    """
    Reads a JSONL file and returns two dicts: preds, gts
    Assumes each line contains {"predict": ..., "label": ...}
    """
    preds, gts = {}, {}
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                key = f"sample_{i}"
                preds[key] = data["predict"]
                gts[key] = data["label"]
            except Exception as e:
                print(f"[Error] Failed to parse line {i}: {e}")
    return preds, gts


def evaluate(pred_file: str, gt_file: str, use_exist_match: bool = False, file_format: str = "tsv", show_mixmatch: bool = False):
    if file_format == "tsv":
        preds, gts = read_tsv(pred_file), read_tsv(gt_file)
    elif file_format == "jsonl":
        preds, gts = read_jsonl(pred_file)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    total = 0
    correct = 0
    mismatches = []

    for key in tqdm(gts, desc="Rule-based Evaluation"):
        if key not in preds:
            print(f"[Warning] Missing prediction for key: {key}")
            continue
        total += 1
        pred = preds[key]
        gt_text = gts[key]
        gt_list = [s.strip() for s in gt_text.split("|||") if s.strip()]  # split refs

        match = any(
            exist_match(pred, gt) if use_exist_match else exact_match(pred, gt)
            for gt in gt_list
        )

        if match:
            correct += 1
        else:
            mismatches.append((key, gt_text, pred))

    accuracy = correct / total if total > 0 else 0
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    if mismatches and show_mixmatch:
        pred_file_dir = os.path.dirname(pred_file)
        mismatch_file = os.path.join(pred_file_dir, "mismatch_examples_new.txt")
        with open(mismatch_file, 'w', encoding='utf-8') as f:
            f.write(f"[Examples of Incorrect Predictions] ({len(mismatches)} shown)\n")
            for key, gt, pred in mismatches:
                gt = gt.strip()
                pred = pred.strip()
                f.write(f"{key}\n  GTs : {gt}\n  Pred: {pred}\n\n")
        print(f"Mismatches saved to {mismatch_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions using exact or existence match.")
    parser.add_argument('--pred', type=str, required=True, help='Path to prediction TSV file.')
    parser.add_argument('--gt', type=str, required=True, help='Path to ground truth TSV file.')
    parser.add_argument('--exist', action='store_true', help='Use existence match instead of exact match.')
    parser.add_argument('--format', type=str, default='tsv', choices=['tsv', 'jsonl'], help='File format of the input files.')
    parser.add_argument('--show-mixmatch', action='store_true', help='Show mixed match examples.')

    args = parser.parse_args()
    evaluate(args.pred, args.gt, use_exist_match=args.exist, file_format=args.format, show_mixmatch=args.show_mixmatch)


if __name__ == "__main__":
    main()
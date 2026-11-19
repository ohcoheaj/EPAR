"""
Export inference Multimodal Scores (0–1) and per-metric class means.
"""

import argparse
import csv
from decimal import Decimal
from pathlib import Path
import re


CLASSES = (
    "bagel", "cable_gland", "carrot", "cookie", "dowel",
    "foam", "peach", "potato", "rope", "tire",
)
METRICS = {
    "I-AUROC": "I-AUROC",
    "P-AUROC": "P-AUROC",
    "AUPRO30": "AUPRO@30%",
    "AUPRO01": "AUPRO@1%",
}
SCORE = re.compile(
    r"@\s+MM\s+(I-AUROC|P-AUROC|AUPRO30|AUPRO01)\s+Score:\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*@"
)


def read_scores(path):
    text = path.read_text(encoding="utf-8")
    if "Multimodal Scores" not in text:
        raise ValueError(f"{path}: no Multimodal Scores section")
    section = text.rsplit("Multimodal Scores", 1)[1]
    scores = {}
    for match in SCORE.finditer(section):
        metric, raw_value = match.groups()
        if metric in scores:
            raise ValueError(f"{path}: duplicate {metric} in final section")
        value = Decimal(raw_value)
        if not Decimal(0) <= value <= Decimal(1):
            raise ValueError(f"{path}: {metric} outside [0, 1]: {value}")
        scores[metric] = value
    missing = set(METRICS) - scores.keys()
    if missing:
        raise ValueError(f"{path}: incomplete final section; missing {sorted(missing)}")
    return scores


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=directory / "inference",
                        help="Directory containing <class>/run.log")
    parser.add_argument("--output", type=Path,
                        default=directory / "multimodal_scores.csv")
    args = parser.parse_args()

    try:
        results = [(name, read_scores(args.log_dir / name / "run.log"))
                   for name in CLASSES]
    except (OSError, ValueError) as error:
        parser.error(str(error))
    averages = {
        metric: sum((scores[metric] for _, scores in results), Decimal(0))
        / Decimal(len(results))
        for metric in METRICS
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class", *METRICS.values()])
        for name, scores in [*results, ("Average", averages)]:
            writer.writerow([name, *(f"{scores[metric]:.4f}" for metric in METRICS)])
    print(f"Saved {len(results)} classes and their macro averages to {args.output}")
    for metric, label in METRICS.items():
        print(f"{label}: {averages[metric]:.4f} ({averages[metric] * 100:.2f}%)")


if __name__ == "__main__":
    main()

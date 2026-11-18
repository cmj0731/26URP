"""Compare the A and B-4 symbol SINR distributions on the same axes."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).with_name(".runtime-deps")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--b4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [json.loads(path.read_text(encoding="utf-8")) for path in (args.static, args.b4)]
    if not all(result["complete"] for result in results):
        raise ValueError("Both completed results are required.")
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.8), sharex=True, sharey=True, layout="constrained")
    titles = ("A · zero residual Doppler", "B-4 · ±46.875 kHz reflections")
    colors = ("#2961a6", "#aa5b17", "#5b448f")
    for ax, result, title in zip(axes, results, titles):
        rows = [row for row in result["results"] if row["waveform"] != "DD-a-OFDM"]
        for i, (row, color) in enumerate(zip(rows, colors)):
            scheme = row["waveform"]
            label = "OFDM / DD-a-OFDM" if scheme == "OFDM" else scheme
            hist = row["histogram"]
            cdf = np.cumsum(hist["symbol_counts"])*100/row["symbols"]
            edges = (hist["first_bin"]+np.arange(len(cdf)+1))*result["histogram_bin_size_db"]
            ax.step(np.r_[edges, 25], np.r_[.20, cdf, 100], where="post", color=color,
                    linewidth=2, label=label, linestyle=("-", "--", "-.")[i])
        ax.axvline(17, color="#6d7176", linewidth=1, linestyle=":")
        below = [b for b in rows[0]["bands_1_db"] if b["upper_db"] <= 17]
        errors = sum(b["bit_errors"] for b in below)
        error_share = 100*errors/rows[0]["bit_errors"]
        ax.text(.98, .055, f"OFDM errors below 17 dB: {errors:,} ({error_share:.1f}%)",
                transform=ax.transAxes, ha="right", color=colors[0], fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .94, "pad": 2})
        ax.set(title=title, xlabel="Post-MMSE SINR (dB)", xlim=(14.5, 24.4), ylim=(.25, 115))
        ax.set_yscale("log")
        ax.set_yticks([.3, 1, 5, 20, 100], ["0.3", "1", "5", "20", "100"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.23)
    axes[0].set_ylabel("Symbols below SINR (%)")
    axes[1].legend(loc="center right", fontsize=9)
    fig.suptitle("Pilot-free Perfect CSI · 15 dB Eb/N0 · same 160 frames", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

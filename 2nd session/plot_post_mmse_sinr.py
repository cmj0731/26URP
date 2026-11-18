"""Plot the lower SINR tail and bit errors by common SINR interval."""

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
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.run/"results.json").read_text(encoding="utf-8"))
    if not result["complete"]:
        raise ValueError("A completed result is required.")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = [row for row in result["results"] if row["waveform"] != "DD-a-OFDM"]
    error_bins = [band["lower_db"] for row in rows for band in row["bands_1_db"] if band["bit_errors"]]
    x = np.arange(min(error_bins), max(error_bins)+1)
    fig, (cdf, counts) = plt.subplots(1, 2, figsize=(12.2, 5), layout="constrained")
    colors = ("#2961a6", "#aa5b17", "#5b448f")
    for i, (row, color) in enumerate(zip(rows, colors)):
        scheme = row["waveform"]
        label = "OFDM / DD-a-OFDM" if scheme == "OFDM" else scheme
        h = row["histogram"]
        y = np.cumsum(h["symbol_counts"])*100/row["symbols"]
        edges = (h["first_bin"]+np.arange(y.size+1))*result["histogram_bin_size_db"]
        cdf.step(np.r_[edges, 25], np.r_[.20, y, 100], where="post", color=color, linewidth=2,
                 label=label, linestyle=("-", "--", "-.")[i])
        error_by_db = {b["lower_db"]: b["bit_errors"] for b in row["bands_1_db"]}
        for lo in x:
            count = error_by_db.get(lo, 0)
            if count:
                center = lo+(i-1)*.23
                counts.bar(center, count-.75, width=.22, color=color, bottom=.75)
                counts.annotate(str(count), (center, count), xytext=(0, 4), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8, color=color)
    cdf.set(xlim=(14.7, 24), ylim=(.25, 115), xlabel="Post-MMSE SINR (dB)", ylabel="Symbols at or below SINR (%)")
    cdf.set_yscale("log")
    cdf.set_yticks([.3, 1, 5, 20, 100], ["0.3", "1", "5", "20", "100"])
    cdf.legend(loc="lower right", fontsize=9)
    counts.set(xlabel="Post-MMSE SINR band (dB)", ylabel="Observed bit errors", ylim=(.75, 3100))
    counts.set_yscale("log")
    counts.set_xticks(x, [f"{lo}–{lo+1}" for lo in x])
    cdf.set_title("SINR distribution and lower tail")
    counts.set_title("Errors in each 1-dB band")
    for ax in (cdf, counts):
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("B-4 · Eb/N0 = 15 dB · pilot-free Perfect CSI · 160 paired frames", fontsize=14)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Render the completed A/B-4 OTFS reflection-spacing comparison."""

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
        raise ValueError("A completed run is required.")
    args.output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained")
    x = np.arange(4)
    for condition, label, color, marker in (
        ("A", "A · zero residual Doppler", "#2764a7", "o"),
        ("B-4", "B-4 · ±46.875 kHz reflections", "#a74176", "D"),
    ):
        rows = [r for r in result["results"] if r["condition"] == condition]
        ber = np.array([r["ber_estimate"] for r in rows])
        ci = np.array([r["confidence_interval"] for r in rows])
        floor = 1/result["measured_bits_per_point"]
        nonzero = ber > 0
        axes[0].plot(x, [r["reflection_delay_rmse_samples"] for r in rows], marker=marker, color=color, label=label)
        axes[1].plot(x, [r["mean_operator_nmse_db"] for r in rows], marker=marker, color=color)
        axes[2].plot(x, np.where(nonzero, ber, np.nan), marker=marker, color=color)
        axes[2].vlines(x, np.maximum(ci[:, 0], floor), ci[:, 1], color=color, alpha=.65)
        if not np.all(nonzero):
            axes[2].scatter(x[~nonzero], np.full(np.count_nonzero(~nonzero), floor), marker="v", facecolors="none", edgecolors=color)
    for ax in axes:
        ax.set_xticks(x, ["3", "1", "0.5", "0.25"])
        ax.set_xlabel("True reflection delay spacing (samples)")
        ax.grid(axis="y", alpha=.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Reflection delay RMSE (samples)")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Mean full-channel NMSE (dB)")
    axes[2].set_ylabel("BER and 95% frame intervals")
    axes[2].set_yscale("log")
    fig.suptitle("OTFS reflection delay spacing · continuous-delay receiver · CP = 5 samples", fontsize=14)
    fig.savefig(args.output/"reflection_spacing.png", dpi=180)
    plt.close(fig)

    table = ["<tr><th>Condition</th><th>Gap (samples)</th><th>Reflection delay RMSE (samples)</th>"
             "<th>Gap RMSE (samples)</th><th>95th percentile of worst reflection error (samples)</th>"
             "<th>Channel NMSE (dB)</th><th>Errors</th><th>BER</th><th>95% BER interval</th><th>Optimizer non-success</th></tr>"]
    for row in result["results"]:
        lo, hi = row["confidence_interval"]
        table.append(f"<tr><td>{row['condition']}</td><td>{row['reflection_spacing_samples']:g}</td>"
            f"<td>{row['reflection_delay_rmse_samples']:.5g}</td><td>{row['reflection_spacing_rmse_samples']:.5g}</td>"
            f"<td>{row['worst_reflection_delay_error_p95_samples']:.5g}</td><td>{row['mean_operator_nmse_db']:.2f}</td>"
            f"<td>{row['bit_errors']:,}</td><td>{row['ber_estimate']:.5g}</td><td>[{lo:.5g}, {hi:.5g}]</td>"
            f"<td>{row['optimizer_non_success_frames']}</td></tr>")
    count = result["measured_bits_per_point"]
    points = {(row["condition"], row["reflection_spacing_samples"]): row for row in result["results"]}
    a_base, a_close = points["A", 3.], points["A", .25]
    b_base, b_close = points["B-4", 3.], points["B-4", .25]
    findings = (
        f"<p>A에서 gap을 3→0.25 sample로 줄이자 reflection delay RMSE는 "
        f"{a_base['reflection_delay_rmse_samples']:.4g}→{a_close['reflection_delay_rmse_samples']:.4g} samples, "
        f"channel NMSE는 {a_base['mean_operator_nmse_db']:.2f}→{a_close['mean_operator_nmse_db']:.2f} dB였다. "
        f"B-4의 delay RMSE는 {b_base['reflection_delay_rmse_samples']:.4g}→{b_close['reflection_delay_rmse_samples']:.4g} samples였다. "
        "같은 delay 간격에서도 B-4의 두 reflection에는 서로 다른 Doppler가 있어 받는 pilot response를 구분할 단서가 있다.</p>"
    )
    page = ("<!doctype html><html lang='ko'><meta charset='utf-8'><title>OTFS reflection delay spacing</title>"
            "<style>body{font:15px system-ui;max-width:1500px;margin:32px auto;padding:0 16px;color:#202733}"
            "p{line-height:1.7;max-width:1100px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}"
            "td,th{border-bottom:1px solid #dce2eb;padding:9px;text-align:left}th{background:#eef2f7}</style>"
            "<h1>OTFS reflection delay spacing</h1>"
            "<p>Physical delays는 [0, 5−gap, 5] samples다. Original W64, single pilot 4,096 Es, CP=5, "
            f"Eb/N0=15 dB와 known P=3 continuous-delay receiver를 사용했다. 각 점은 214 frames 및 {count:,} bits다. "
            "Gap=3 기준점은 이전 실험의 continuous-delay 결과에서 가져왔으며 첫 frame을 다시 계산해 확인했다.</p>"
            +findings+
            "<img src='reflection_spacing.png' alt='A와 B-4의 reflection delay RMSE, channel NMSE, BER'>"
            "<table>"+"".join(table)+"</table>"
            "<p>Reflection delay RMSE는 frame마다 추정 delay 세 개를 오름차순으로 매칭하고, 두 reflection의 "
            "squared errors를 모아 계산했다. Gap RMSE는 두 추정 reflection 사이의 간격 오차다. 두 점수는 delay set의 "
            "복원을 평가하며 path별 Doppler 매칭을 점수에 포함하지 않는다. Operator NMSE는 full useful channel을 평가한다.</p>"
            "<p>BER interval은 pointwise 95% frame-cluster interval이다. Optimizer non-success는 numerical status를 "
            "뜻하며, 표는 같은 receiver가 반환한 추정값으로 모든 frame을 집계한 결과다.</p></html>")
    (args.output/"index.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()

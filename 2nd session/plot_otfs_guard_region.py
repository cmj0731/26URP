"""Render completed OTFS guard region redistribution diagnostics."""

import argparse
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).with_name(".runtime-deps")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from otfs_guard_design import WIDTHS, build_designs


def save(fig, root, name):
    fig.savefig(root/(name+".png"), dpi=180, bbox_inches="tight")
    vector = root/(name+".svg")
    fig.savefig(vector, bbox_inches="tight", metadata={"Date": None})
    vector.write_text("\n".join(line.rstrip() for line in vector.read_text(encoding="utf-8").splitlines())+"\n", encoding="utf-8")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.run/"results.json").read_text(encoding="utf-8"))
    if not result["complete"] or result["diagnostic_only"]:
        raise ValueError("Final plots require a completed full-quota run.")
    base = json.loads((args.run/"scenario_used.json").read_text(encoding="utf-8"))
    designs = build_designs(base)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "otfs-guard"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.1))
    x = np.arange(len(WIDTHS))
    positive = []
    for condition, color, marker, label in (
            ("A", "#2458a6", "o", "A: zero residual Doppler"),
            ("B-4", "#a54294", "D", "B-4: max residual Doppler 46.875 kHz")):
        rows = [next(r for r in result["results"] if r["condition"] == condition and r["local_zero_delay_row_count"] == w) for w in WIDTHS]
        y = np.array([r["ber_estimate"] for r in rows])
        lo = np.array([r["confidence_interval"][0] for r in rows])
        hi = np.array([r["confidence_interval"][1] for r in rows])
        nonzero = y > 0
        positive.extend(y[nonzero])
        axes[0].plot(x, np.where(nonzero, y, np.nan), color=color, marker=marker, label=label)
        axes[0].vlines(x[nonzero], np.maximum(lo[nonzero], .5/result["measured_bits_per_point"]), hi[nonzero], color=color, lw=1, alpha=.65)
        if np.any(~nonzero):
            axes[0].scatter(x[~nonzero], np.full(np.count_nonzero(~nonzero), 1/result["measured_bits_per_point"]), marker="v", facecolors="none", edgecolors=color)
        axes[1].plot(x, [r["mean_channel_nmse_db"] for r in rows], color=color, marker=marker)
        axes[2].plot(x, [100*r["mean_data_to_pilot_observation_energy_ratio"] for r in rows], color=color, marker=marker)
    for ax in axes:
        ax.set_xticks(x, ["64\n(original)", "11", "9", "5", "1"])
        ax.set_xlim(-.25, len(WIDTHS)-.75)
        ax.set_xlabel("Local zero region width (delay rows)")
        ax.grid(True, which="major", alpha=.25)
    axes[0].set_yscale("log")
    lower = 10**np.floor(np.log10(min(positive)/3)) if positive else .5/result["measured_bits_per_point"]
    if any(r["bit_errors"] == 0 for r in result["results"]):
        lower = min(lower, .5/result["measured_bits_per_point"])
    axes[0].set_ylim(lower, min(1., 10**np.ceil(np.log10(max(positive)*3))))
    axes[0].set_ylabel("BER")
    axes[0].set_title("Data-bit recovery")
    axes[0].legend(fontsize=7.3)
    axes[1].set_ylabel("Mean channel NMSE (dB)")
    axes[1].set_title("Channel estimation")
    axes[2].set_ylabel("Data / pilot observation energy (%)")
    axes[2].set_ylim(bottom=0)
    axes[2].set_title("Data interference in pilot observations")
    fig.suptitle(r"OTFS guard region sensitivity: pilot energy $4,096E_s$, data $E_b/N_0=15$ dB", fontsize=14)
    fig.text(.02, .015,
        "Five discrete layouts; 1 pilot, 4,095 zero cells and 12,288 data cells in each. Original channel estimator and observations are fixed.\n"
        "Same source data, noise and channel draws. 10,485,760 measured bits/point. BER bars: pointwise 95% frame-cluster intervals.", fontsize=8)
    fig.tight_layout(rect=(0, .12, 1, .94))
    save(fig, args.output, "guard_region_sensitivity")

    colors = ["#dce8f4", "#223b54", "#a8b4bd", "#ed8b23"]
    labels = ["Data", "Guard", "Reserved", "Pilot"]
    fig, axes = plt.subplots(1, 5, figsize=(13, 5.1), sharey=True)
    for ax, design in zip(axes, designs):
        masks, spec = design["masks"], design["descriptor"]
        grid = np.zeros(masks["data"].shape, dtype=int)
        grid[masks["guard"]] = 1
        grid[masks["reserved"]] = 2
        grid[masks["pilot"]] = 3
        ax.imshow(grid, origin="lower", aspect="auto", interpolation="nearest", cmap=ListedColormap(colors), vmin=0, vmax=3)
        ax.scatter([32], [128], s=25, facecolors="none", edgecolors="#ed8b23", linewidths=1)
        ax.set_title(f"W{spec['local_zero_delay_row_count']}")
        ax.set_xticks([0, 32, 63])
        ax.set_yticks([0, 64, 128, 192, 255])
        ax.set_xlabel("Doppler index")
    axes[0].set_ylabel("Delay index")
    fig.suptitle("DD layouts: local zero cells are exchanged with distant data cells", fontsize=13)
    fig.legend([Patch(facecolor=c) for c in colors], labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .1, 1, .93))
    save(fig, args.output, "guard_region_layouts")

    selected = {(r["condition"], r["local_zero_delay_row_count"]): r for r in result["results"]}
    original, sufficient, narrowed = [selected[("B-4", w)] for w in (64, 11, 9)]
    content = ["<h1>OTFS guard region sensitivity</h1>",
        "<p>pilot energy=4,096 Es, data Eb/N0=15 dB, A·B-4에서 local zero region width만 64·11·9·5·1 delay rows로 바꿨다. "
        "Doppler 방향은 모든 64 bins를 유지했다. 가까운 zero cells를 data로 바꾼 만큼 delay rows 0~62 범위에 reserved cells를 두었다. "
        "모든 layout은 pilot 1개, zero cells 4,095개, data cells 12,288개다.</p>",
        "<p>공통 data cells의 symbol은 그대로 두고, 새 reserved cells에서 밀려난 data만 새로 열린 cells로 옮겼다. "
        "복원 후 원래 information-bit 순서로 되돌려 집계했다. 같은 data·channel·noise·calibration을 사용했고, "
        "useful energy, CP energy와 sampled peak도 불변인지 검사했다. sampled peak는 continuous-time/RF peak 검증을 뜻하지 않는다.</p>",
        "<p>channel estimator는 원래 W64의 pilot-only model, observation region, public bounds와 search를 고정했다. "
        "W9·W5·W1은 data isolation 가정을 의도적으로 완화한 ablation이다. unknown-data interference를 수신기에서 정답 data로 제거하지 않았다. "
        "actual transmit mask는 MMSE data embedding과 known-pilot subtraction에 정확히 반영했다.</p>",
        "<p>각 점 10,485,760 bits, 214 frames다. W64는 기존 결과를 frame별로 재현했다. "
        "data/pilot 성분 분리와 true channel 비교는 leakage·NMSE 평가에만 사용했다.</p>",
        f"<h2>Observed findings</h2><p>W64→W11에서 B-4 channel NMSE는 "
        f"{original['mean_channel_nmse_db']:.3f}→{sufficient['mean_channel_nmse_db']:.3f} dB로 같았고, "
        "pilot observations와 estimated taps도 numerical precision 수준에서 일치했다. "
        "현재 model에서 extra reserved cells를 pilot 근처에 두는 것이 channel estimation을 더 정확하게 만든 것은 아니다.</p>",
        f"<p>W11→W9에서는 data/pilot observation energy ratio가 numerical zero에서 "
        f"{100*narrowed['mean_data_to_pilot_observation_energy_ratio']:.3f}%로 증가했고, "
        f"channel NMSE는 {sufficient['mean_channel_nmse_db']:.3f}→{narrowed['mean_channel_nmse_db']:.3f} dB, "
        f"BER은 {sufficient['ber_estimate']:.6g}→{narrowed['ber_estimate']:.6g}로 악화됐다. "
        "고정한 receiver에서 pilot observation의 data isolation이 channel estimation에 기여한다는 근거다.</p>",
        f"<p>W64/W11의 B-4 errors는 {original['bit_errors']}/{sufficient['bit_errors']}개였다. "
        "channel estimate가 같으므로 이 작은 BER 차이를 W11의 channel estimation 우위로 해석하지 않는다. "
        "data placement·detection 구조와 finite sampling의 영향이 남아 있다. W1도 pilot row의 63 zero cells를 유지하므로 완전한 no-guard case는 아니다.</p>",
        '<img src="guard_region_layouts.svg" alt="DD layouts의 Guard·Reserved·Data·Pilot 구분">',
        '<img src="guard_region_sensitivity.svg" alt="Guard region width에 따른 BER·NMSE·data interference">',
        "<table><tr><th>Condition</th><th>Local width (rows)</th><th>Errors / bits</th><th>BER</th>"
        "<th>95% frame interval</th><th>Channel NMSE (dB)</th><th>Data / pilot observation energy (%)</th><th>Delay-set match</th></tr>"]
    for row in result["results"]:
        low, high = row["confidence_interval"]
        ber = f"{row['ber_estimate']:.6g}" if row["bit_errors"] else "No errors observed"
        leakage = row["mean_data_to_pilot_observation_energy_ratio"]
        leak_label = "Numerical zero" if leakage < 1e-20 else f"{100*leakage:.5g}"
        content.append(f"<tr><td>{html.escape(row['condition'])}</td><td>{row['local_zero_delay_row_count']}</td>"
            f"<td>{row['bit_errors']:,} / {row['measured_information_bits']:,}</td><td>{ber}</td>"
            f"<td>[{low:.5g}, {high:.5g}]<br><small>Error-bearing frames {row['error_frame_count']}/{row['frame_count']}</small></td>"
            f"<td>{row['mean_channel_nmse_db']:.3f}</td><td>{leak_label}</td>"
            f"<td>{row['frames_with_correct_delay_multiset']}/{row['frame_count']}</td></tr>")
    content.extend(["</table><p>NMSE와 energy ratio는 frame별 linear ratio를 평균한 뒤 필요시 dB로 변환했다. "
        "BER interval은 기존 approximate CR1 frame-cluster 방식이며 동시 confidence band는 아니다. "
        "Data placement도 바뀌므로 BER 변화 전체를 channel estimation loss로만 해석하지 않는다. "
        "이 실험은 fixed receiver의 sensitivity이며 optimized reduced-guard receiver의 한계 성능이나 physical edge guard band의 효과를 뜻하지 않는다.</p>"])
    style = "body{font:15px system-ui;max-width:1500px;margin:36px auto;padding:0 20px;color:#202733}p{line-height:1.7;max-width:1200px}img{width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:8px;border-bottom:1px solid #dce2eb;text-align:left}th{background:#eef2f7}small{color:#576475}"
    (args.output/"index.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>OTFS guard region sensitivity</title><style>'+style+'</style><body>'+"".join(content)+"</body></html>", encoding="utf-8")
    print("Guard region figures and result table saved:", args.output)


if __name__ == "__main__":
    main()

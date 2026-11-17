"""Render completed within-OTFS pilot-energy measurements."""

import argparse
import html
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
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.results.read_text(encoding="utf-8"))
    if not result["complete"] or result["diagnostic_only"]:
        raise ValueError("Final plots require the completed full-quota experiment.")
    assert result["original_reference_reproduced_all_completed_frames"]
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "otfs-energy"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    positive = []
    for condition, color, marker, label in (
            ("A", "#2458a6", "o", "A: zero residual Doppler"),
            ("B-4", "#a54294", "D", "B-4: max residual Doppler 46.875 kHz")):
        rows = sorted([r for r in result["results"] if r["condition"] == condition],
                      key=lambda r: r["pilot_energy_Es"])
        x = np.array([r["pilot_energy_Es"] for r in rows])
        y = np.array([r["ber_estimate"] for r in rows])
        lo = np.array([r["confidence_interval"][0] for r in rows])
        hi = np.array([r["confidence_interval"][1] for r in rows])
        nonzero = y > 0
        positive.extend(y[nonzero])
        axes[0].plot(x, np.where(nonzero, y, np.nan), color=color, marker=marker, label=label)
        axes[0].vlines(x[nonzero], np.maximum(lo[nonzero], .5/result["measured_bits_per_point"]),
                      hi[nonzero], color=color, alpha=.65, lw=1)
        if np.any(~nonzero):
            axes[0].scatter(x[~nonzero], np.full(np.count_nonzero(~nonzero), 1/result["measured_bits_per_point"]),
                            marker="v", facecolors="none", edgecolors=color)
        axes[1].plot(x, [r["mean_channel_nmse_db"] for r in rows], color=color,
                     marker=marker, label=label)
    for ax in axes:
        ax.set_xscale("log", base=4)
        ax.set_xticks([1, 4, 16, 64, 256, 1024, 4096],
                      ["1", "4", "16", "64", "256", "1,024", "4,096"])
        ax.set_xlim(.75, 5500)
        ax.set_xlabel(r"Total energy of the single DD pilot ($E_s$ units)")
        ax.grid(True, which="major", alpha=.25)
        ax.axvline(4096, color="#586577", ls=":", lw=1)
    axes[0].set_yscale("log")
    lower = 10**np.floor(np.log10(min(positive)/3)) if positive else .5/result["measured_bits_per_point"]
    if any(r["bit_errors"] == 0 for r in result["results"]):
        lower = min(lower, .5/result["measured_bits_per_point"])
    axes[0].set_ylim(lower, 1)
    axes[0].set_ylabel("BER")
    axes[0].set_title("Data-bit recovery")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].set_ylabel("Mean channel NMSE (dB)")
    axes[1].set_title("Channel estimation (lower is better)")
    fig.suptitle(r"OTFS single-pilot energy sensitivity: fixed data $E_b/N_0=15$ dB", fontsize=14)
    fig.text(.02, .015,
        "One pilot; same masks, Es=1 data, noise and channel draws, and receiver. No frame-power renormalization.\n"
        "Each point: 10,485,760 measured bits. Bars: pointwise 95% frame-cluster intervals. Dotted line: original pilot energy.",
        fontsize=8)
    fig.tight_layout(rect=(0, .1, 1, .94))
    fig.savefig(args.output/"pilot_energy_sensitivity.png", dpi=180, bbox_inches="tight")
    vector = args.output/"pilot_energy_sensitivity.svg"
    fig.savefig(vector, bbox_inches="tight", metadata={"Date": None})
    vector.write_text("\n".join(line.rstrip() for line in vector.read_text(encoding="utf-8").splitlines())+"\n", encoding="utf-8")
    plt.close(fig)
    selected = {(r["condition"], r["pilot_energy_Es"]): r for r in result["results"]}
    original, reduced, smallest = [selected[("B-4", energy)] for energy in (4096, 256, 1)]
    content = ["<h1>OTFS 단일 파일럿 에너지 영향 검증</h1>",
        "<p>파일럿 1개를 유지하고 총 파일럿 에너지를 4,096·1,024·256·64·16·4·1 Es로 변경했다. "
        "데이터 Es=1, 데이터 Eb/N0=15 dB, CP 5샘플, 보호·예약칸 4,095개와 데이터 12,288칸은 고정했다.</p>",
        "<p>프레임별 데이터·경로 초기 위상·데이터 구간 잡음·별도 잡음 측정값을 모든 에너지에서 재사용했다. "
        "파일럿 에너지를 바꿔도 잡음 전력과 데이터 에너지를 바꾸거나 프레임 전력을 다시 정규화하지 않았다. "
        "총 전송 에너지가 다른 OTFS 내부의 원인 분리 실험이며, 네 파형의 동일 총에너지 성능 비교가 아니다.</p>",
        "<p>A는 세 경로의 잔여 도플러가 모두 0, B-4는 [0,+46.875,−46.875] kHz이다. "
        "지연 0·2·5샘플과 상대 전력 0·−10·−15 dB의 정규화된 통제 채널을 그대로 사용했다.</p>",
        "<p><strong>각 점 10,485,760비트, 214프레임.</strong> 원래 4,096 Es 조건의 오류 수는 "
        "기존 실험의 모든 프레임과 일치했다. NMSE는 추정 후 평가에서만 실제 채널을 사용해 계산했다. "
        "수신기에는 실제 지연·도플러·이득이나 실제 잡음 전력을 전달하지 않았다.</p>",
        f"<h2>관측 결과</h2><p>B-4에서 파일럿 에너지 4,096→1 Es로 낮췄을 때 "
        f"오류는 {original['bit_errors']:,}→{smallest['bit_errors']:,}개, "
        f"BER은 {original['ber_estimate']:.6g}→{smallest['ber_estimate']:.6g}, "
        f"채널 NMSE는 {original['mean_channel_nmse_db']:.2f}→{smallest['mean_channel_nmse_db']:.2f} dB로 변했다. "
        "고정한 이 모델에서 파일럿 에너지의 감소가 채널 추정 및 복원 성능을 악화시켰음을 확인했다.</p>",
        f"<p>한편 256 Es에서도 B-4의 오류는 {reduced['bit_errors']:,}개, BER은 {reduced['ber_estimate']:.6g}로 "
        "원래 측정값 근처였다. 적은 오류 수에서의 작은 차이를 동등성 증명이나 최적 에너지 선정으로 해석하지 않는다. "
        "원래의 매우 큰 파일럿 에너지만으로 낮은 BER 전체가 설명된다는 결론도 내리지 않는다. "
        "현재의 보호영역과 3경로 추정 모델은 그대로 남아 있다.</p>",
        '<img src="pilot_energy_sensitivity.svg" alt="파일럿 에너지에 따른 BER과 채널 추정 오차">',
        "<table><tr><th>조건</th><th>파일럿 에너지 (Es)</th><th>오류 수 / 비트 수</th><th>BER</th>"
        "<th>95% 프레임 구간</th><th>평균 채널 NMSE (dB)</th><th>지연 3개 복원 성공 프레임</th></tr>"]
    for row in result["results"]:
        low, high = row["confidence_interval"]
        ber = f"{row['ber_estimate']:.6g}" if row["bit_errors"] else "오류 미관측"
        content.append(f"<tr><td>{html.escape(row['condition'])}</td><td>{row['pilot_energy_Es']:,.0f}</td>"
            f"<td>{row['bit_errors']:,} / {row['measured_information_bits']:,}</td><td>{ber}</td>"
            f"<td>[{low:.5g}, {high:.5g}]<br><small>오류 발생 프레임 {row['error_frame_count']}/{row['frame_count']}</small></td>"
            f"<td>{row['mean_channel_nmse_db']:.3f}</td>"
            f"<td>{row['frames_with_correct_delay_multiset']}/{row['frame_count']}</td></tr>")
    content.extend(["</table><p>NMSE는 프레임별 선형 오차비를 평균한 뒤 dB로 변환했다. "
        "BER 신뢰구간은 기존과 같은 프레임 기준 근사 CR1 방식이며 동시 신뢰대역은 아니다. "
        "오류 미관측 시 빈 삼각형의 1/N 위치는 표시용이고, 보수적 상한은 표에 별도로 보고한다.</p>",
        "<p>이 실험은 파일럿 에너지의 효과만 분리한다. 보호영역·격자 정렬·3경로 추정 모델의 영향이나 "
        "OFDM과의 전체 BER 격차의 원인을 모두 설명한 것으로 해석하지 않는다.</p>"])
    style = "body{font:15px system-ui;max-width:1400px;margin:36px auto;padding:0 20px;color:#202733}p{line-height:1.7;max-width:1150px}img{width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:9px;border-bottom:1px solid #dce2eb;text-align:left}th{background:#eef2f7}small{color:#576475}"
    (args.output/"index.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>OTFS 파일럿 에너지 검증</title><style>'+style+"</style><body>"+"".join(content)+"</body></html>", encoding="utf-8")
    print("OTFS energy figures and result table saved:", args.output)


if __name__ == "__main__":
    main()

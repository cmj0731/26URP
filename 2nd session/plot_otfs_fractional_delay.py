"""Render completed CP=5 OTFS fractional-delay sensitivity results."""
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


def save(fig, output, name):
    fig.savefig(output/(name+".png"), dpi=180, bbox_inches="tight")
    path = output/(name+".svg")
    fig.savefig(path, bbox_inches="tight", metadata={"Date": None})
    path.write_text("\n".join(s.rstrip() for s in path.read_text(encoding="utf-8").splitlines())+"\n", encoding="utf-8")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.run/"results.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.run/"manifest.json").read_text(encoding="utf-8"))
    if not result["complete"] or result["diagnostic_only"]:
        raise ValueError("A completed full-quota run is required.")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "otfs-fractional-cp5"})
    colors = {"original_integer_delay": "#b63d41", "continuous_delay": "#2468ad"}
    labels = {"original_integer_delay": "Original integer-delay receiver", "continuous_delay": "Continuous-delay receiver"}
    selected = {(r["condition"], r["receiver"], r["delta_samples"]): r for r in result["results"]}
    deltas = manifest["delta_samples"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3))
    for column, condition in enumerate(manifest["conditions"]):
        for receiver in manifest["receivers"]:
            rows = [selected[(condition, receiver, d)] for d in deltas]
            ber = np.array([r["ber_estimate"] for r in rows])
            lo, hi = np.array([r["confidence_interval"] for r in rows]).T
            axes[0, column].plot(deltas, ber, color=colors[receiver], marker="o", label=labels[receiver])
            axes[0, column].vlines(deltas, np.maximum(lo, 1e-8), hi, color=colors[receiver], alpha=.65)
            axes[1, column].plot(deltas, [r["mean_operator_nmse_db"] for r in rows], color=colors[receiver], marker="o")
        axes[0, column].set_title("A: zero residual Doppler" if condition=="A" else "B-4: max residual Doppler 46.875 kHz")
        axes[0, column].set_yscale("log")
        axes[0, column].set_ylim(1e-6, 1e-1)
        axes[0, column].set_ylabel("BER")
        axes[0, column].legend(fontsize=8)
        axes[1, column].set_ylabel("Mean full-channel operator NMSE (dB)")
    for ax in axes.flat:
        ax.set_xticks(deltas)
        ax.set_xlabel(r"Reflection 1 fractional delay $\delta$ (samples)")
        ax.grid(True, alpha=.25)
    fig.suptitle(r"OTFS: physical delays $[0,2+\delta,5]$ samples, CP=5 unchanged", fontsize=15)
    fig.text(.02, .015,
        "Original W64 mask; one 4,096 Es pilot; data Eb/N0=15 dB. Same data, channel gains, noise and calibration draws.\n"
        "10,485,760 measured bits/point; pointwise 95% frame-cluster intervals. Neither receiver receives active path truth.", fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .95))
    save(fig, args.output, "fractional_delay_sensitivity")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for condition, color, marker in (("A", "#2458a6", "o"), ("B-4", "#a54294", "D")):
        rows = [selected[(condition, "original_integer_delay", d)] for d in deltas]
        axes[0].plot(deltas, [100*r["mean_data_to_pilot_observation_energy_ratio"] for r in rows], color=color, marker=marker, label=condition)
        axes[1].plot(deltas, [100*r["mean_pilot_energy_capture_fraction"] for r in rows], color=color, marker=marker, label=condition)
    for ax in axes:
        ax.set_xticks(deltas)
        ax.set_xlabel(r"Reflection 1 fractional delay $\delta$ (samples)")
        ax.grid(True, alpha=.25)
        ax.legend()
    axes[0].set_ylabel("Data / pilot observation energy (%)")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel("Pilot energy captured in original CE region (%)")
    fig.suptitle("Fixed pilot observation region: delay rows 128..133, all 64 Doppler bins", fontsize=12)
    fig.tight_layout()
    save(fig, args.output, "pilot_observation_sensitivity")

    integer0 = selected[("B-4", "original_integer_delay", 0.)]
    integer5 = selected[("B-4", "original_integer_delay", .5)]
    aware0 = selected[("B-4", "continuous_delay", 0.)]
    aware5 = selected[("B-4", "continuous_delay", .5)]
    content = ["<h1>OTFS fractional-delay sensitivity · CP=5</h1>",
        "<p>Reflection 1 delay만 2→2.1→2.25→2.5 samples로 바꿨다. LOS=0, reflection 2=5 samples, "
        "CP=5 samples, W64 layout, pilot energy=4,096 Es, data Eb/N0=15 dB를 유지했다. "
        "모든 점은 같은 data/channel-gain/noise/calibration draws와 10,485,760 measured bits를 사용한다.</p>",
        "<p>각 CP+useful slot의 기존 waveform을 signed FFT frequencies의 finite Fourier series로 정의하고, "
        "serial channel이 t−delay의 실제 source slot을 선택해 평가했다. Whole-frame circular channel이나 "
        "추가 interpolation FIR를 사용하지 않았다. Physical delays가 CP 이내이므로 useful outputs는 자기 slot의 CP+useful waveform에서 온다. "
        "DD response의 spreading은 CP 부족에 따른 inter-slot interference를 의미하지 않는다. RF/matched filtering은 추가하지 않았다.</p>",
        "<p>두 receiver 모두 같은 six receive delay rows 128..133와 모든 64 Doppler bins, known P=3, "
        "delay bounds 0..5 samples와 Doppler bounds ±46.875 kHz만 사용한다. Actual delta/delays/frequencies/gains를 알려주지 않았다. "
        "기존 receiver는 integer-delay hypotheses를 유지한다. Continuous-delay receiver는 공개된 0.25-sample 초기 lattice와 "
        "기존 frequency grid에서 시작해 세 paths의 continuous delays/frequencies와 joint-LS gains를 추정한다. "
        "최적화는 local approximate ML이며 global optimum을 보장하지 않는다. Full DD MMSE와 estimated-channel self-gain 기반 Gray decisions를 유지했다.</p>",
        "<h2>Observed findings</h2>",
        f"<p>B-4에서 δ=0→0.5일 때 기존 receiver BER은 {integer0['ber_estimate']:.6g}→{integer5['ber_estimate']:.6g}, "
        f"continuous-delay receiver BER은 {aware0['ber_estimate']:.6g}→{aware5['ber_estimate']:.6g}였다. "
        f"δ=0.5의 full-channel operator NMSE는 각각 {integer5['mean_operator_nmse_db']:.3f} / {aware5['mean_operator_nmse_db']:.3f} dB다. "
        "같은 fractional channel에서 기존 receiver와 continuous-delay receiver를 구분해야 channel assumption과 model mismatch를 분리해 해석할 수 있다.</p>",
        "<p>이 model에서는 fractional delay를 추정해도 BER이 10^-6 수준으로 유지됐다. "
        "기존 성능 격차를 integer-grid alignment만으로 설명할 수 있다는 결론은 얻지 못했다. "
        "Known P=3, fixed path gains/Dopplers, ideal waveform, wide guard와 high pilot energy 등 다른 조건의 영향은 남아 있다.</p>",
        "<p>기존 receiver의 δ=0은 원래 214개 frames의 error counts를 재현했다. Continuous-delay receiver는 δ=0에서도 "
        "delay를 추정할 수 있는 parameter space가 커지므로 기존 receiver와 완전히 같은 algorithm은 아니다. "
        "작은 BER 차이만으로 우열이나 동등성을 단정하지 않는다. 이 결과는 고정된 sparse three-path, ideal Fourier-waveform model에 대한 검증이다.</p>",
        '<img src="fractional_delay_sensitivity.svg" alt="Fractional delay에 따른 두 receiver의 BER과 operator NMSE">',
        '<img src="pilot_observation_sensitivity.svg" alt="고정된 CE region의 data interference와 pilot energy capture">',
        "<table><tr><th>Condition</th><th>δ (samples)</th><th>Receiver</th><th>Errors / bits</th><th>BER</th><th>95% frame interval</th><th>Operator NMSE (dB)</th><th>Data / pilot energy (%)</th></tr>"]
    for r in result["results"]:
        low, high = r["confidence_interval"]
        content.append(f"<tr><td>{html.escape(r['condition'])}</td><td>{r['delta_samples']:g}</td>"
            f"<td>{html.escape(labels[r['receiver']])}</td><td>{r['bit_errors']:,} / {r['measured_information_bits']:,}</td>"
            f"<td>{r['ber_estimate']:.6g}</td><td>[{low:.5g}, {high:.5g}]</td><td>{r['mean_operator_nmse_db']:.3f}</td>"
            f"<td>{100*r['mean_data_to_pilot_observation_energy_ratio']:.5g}</td></tr>")
    content.extend(["</table><p>NMSE는 full useful-channel operator의 Frobenius error다. Fractional response를 six integer taps로 truncate하지 않는다. "
        "Data/pilot decomposition과 true operator는 evaluator 전용이며 receiver cancellation에 사용하지 않는다.</p>",
        f"<p>Numerical status: max MMSE iterations={result['max_iterations']}, max relative normal residual={result['max_relative_normal_residual']:.6g}, "
        f"continuous optimizer non-success count={result['continuous_optimizer_non_success_count']}. No frames were dropped. "
        "BER intervals use the previous approximate CR1 frame-cluster method; they are pointwise, not simultaneous bands. "
        "Pilot energy capture is the clean pilot component energy in the original CE region divided by its energy in all useful received cells.</p>"])
    style = "body{font:15px system-ui;max-width:1450px;margin:36px auto;padding:0 20px;color:#202733}p{line-height:1.75;max-width:1250px}img{width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #dce2eb;text-align:left}th{background:#eef2f7}"
    (args.output/"index.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>OTFS fractional delay · CP=5</title><style>'+style+'</style><body>'+"".join(content)+"</body></html>", encoding="utf-8")
    print("Fractional-delay report saved:", args.output)


if __name__ == "__main__":
    main()

"""Recount completed fractional-delay frames and verify reproduction/provenance."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from ber_statistics import summarize_frames


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    manifest, result = read(run/"manifest.json"), read(run/"results.json")
    frames = sorted([json.loads(s) for s in (run/"frames.jsonl").read_text(encoding="utf-8").splitlines()], key=lambda r: r["frame_index"])
    assert result["complete"] and not result["diagnostic_only"]
    assert len(frames)==manifest["requested_frame_count"]==214
    assert [r["frame_index"] for r in frames]==list(range(214))
    assert [r["measured_bits"] for r in frames]==manifest["measurement_counts_by_frame"]
    assert sum(r["measured_bits"] for r in frames)==manifest["requested_bits_per_point"]==10485760
    assert len(result["results"])==16
    assert manifest["CP_samples"]==5 and manifest["pilot_energy_Es"]==4096
    for name, digest in manifest["code_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest, name
        ast.parse(Path(name).read_text(encoding="utf-8"))
    original_run = Path(manifest["reference_run"])
    original_manifest, original_result = read(original_run/"manifest.json"), read(original_run/"results.json")
    assert all(manifest["code_sha256"][name]==digest for name, digest in original_manifest["code_sha256"].items())
    assert (run/"scenario_used.json").read_bytes()==(original_run/"scenario_used.json").read_bytes()
    original_frames = sorted([json.loads(s) for s in (original_run/"frames.jsonl").read_text(encoding="utf-8").splitlines()], key=lambda r: r["frame_index"])
    for row in frames:
        i = row["frame_index"]
        assert row["reference_errors_match"]
        for ci, oi in enumerate((0, 4)):
            assert row["errors"][ci][0][0]==original_frames[i]["errors"][oi][4][2]
        assert np.isfinite(row["operator_nmse"]).all() and np.min(row["operator_nmse"])>=0
        assert np.max(row["relative_normal_residual"])<=1e-8
        leakage = np.asarray(row["data_to_pilot_observation_energy_ratio"])
        assert np.max(leakage[:, 0])<1e-20 and np.min(leakage[:, 1:])>1e-12
        capture = np.asarray(row["pilot_energy_capture_fraction"])
        assert np.min(capture)>0 and np.max(capture)<=1+1e-12
        for condition in row["estimated_paths_delay_samples_frequency_hz"]:
            for delta in condition:
                for paths in delta:
                    assert len(paths)==3
                    assert all(0<=d<=5 and abs(f)<=46875 for d, f in paths)
    for row in result["results"]:
        ci = manifest["conditions"].index(row["condition"])
        di = manifest["delta_samples"].index(row["delta_samples"])
        ri = manifest["receivers"].index(row["receiver"])
        summary = summarize_frames([r["errors"][ci][di][ri] for r in frames], [r["measured_bits"] for r in frames], confidence=.95, full_frame_capacity=manifest["frame_capacity_bits"])
        assert all(row[key]==value for key, value in summary.items())
        assert np.isclose(row["mean_operator_nmse"], np.mean([r["operator_nmse"][ci][di][ri] for r in frames]), rtol=1e-13)
        if di==ri==0:
            reference = next(r for r in original_result["results"] if r["condition"]==row["condition"] and r["waveform"]=="OTFS" and r["Eb_N0_db"]==15.)
            assert row["confidence_interval"]==reference["confidence_interval"]
    checks = read(Path("data/ber_diagnostics/otfs_fractional_checks_2026_11_18.json"))
    assert checks["complete"] and checks["preceding_slot_change_next_useful_max_absolute_error"]==0.
    success_count = sum(item["success"] for r in frames for condition in r["continuous_optimizer"] for item in condition)
    payload = {"complete": True, "frames": 214, "information_bits_per_point": 10485760, "result_rows": 16,
        "CP_samples": 5, "original_sources_unchanged": True, "numerical_source_files_checked": len(manifest["code_sha256"]),
        "original_snapshot_bytes_preserved": True, "original_reference_reproduced_every_frame": True,
        "original_confidence_intervals_reproduced": True, "checkpoint_counts_and_CIs_match": True,
        "all_estimated_paths_within_public_bounds": True,
        "integer_channel_pilot_data_isolation": True, "fractional_channel_nonzero_data_interference": True,
        "max_integer_serial_relative_error": max(r["integer_serial_relative_error"] for r in frames),
        "max_iterations": result["max_iterations"], "max_relative_normal_residual": result["max_relative_normal_residual"],
        "continuous_optimizer_success_count": success_count, "continuous_optimizer_total": 214*2*4,
        "continuous_optimizer_non_success_count": result["continuous_optimizer_non_success_count"],
        "independent_checks": "data/ber_diagnostics/otfs_fractional_checks_2026_11_18.json",
        "scope": "Truth only in generator/evaluator and labeled noiseless unit controls. BER receivers see received observations, known pilot, public framing/support/P=3 and measured noise. CP5 coverage checked by changing the preceding slot; full useful-channel operator NMSE and dense-reference self-gain checked independently."}
    (run/"completion_check.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

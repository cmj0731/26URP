"""Check estimated-pilot subtraction and data-only native channel columns.

Controls compare against independent serial channels. Real receiver integrations
retain channel-estimation errors and observation noise. No MMSE solve or BER.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from afdm_estimation import build_estimator as build_afdm
from channel_operators import build_channel_operator
from check_channel_operators import SCHEMES, path_taps, receive, relative_error, serial_tap_channel, transmit
from dd_a_estimation import build_estimator as build_dd_a
from ofdm_bem import build_pilot_ls_model
from otfs_estimation import build_estimator as build_otfs
from pilot_data_model import build_pilot_data_model
from pilot_layouts import build_grid, build_masks
from review_afdm_estimator import apply_paths


def check_linear_problem(model, data, probe):
    predicted = model.apply(data)
    backward = model.adjoint(probe)
    scale = max(np.linalg.norm(predicted)*np.linalg.norm(probe),
                np.linalg.norm(data)*np.linalg.norm(backward),1e-300)
    adjoint_error = float(abs(np.vdot(predicted,probe)-np.vdot(data,backward))/scale)
    block_error = block_adjoint_error = 0.0
    if model.scheme != "OTFS":
        for block in (0,1,63):
            matrix = model.block_matrix(block)
            selected = model.block_data_slice(block)
            assert matrix.shape == (256,selected.stop-selected.start)
            block_error = max(block_error,relative_error(matrix@data[selected],predicted[:,block]))
            block_adjoint_error = max(block_adjoint_error,
                relative_error(matrix.conj().T@probe[:,block],backward[selected]))
    assert max(adjoint_error,block_error,block_adjoint_error) < 1e-11
    return {"data_adjoint_inner_product_error":adjoint_error,
            "local_data_matrix_relative_error":block_error,
            "local_data_matrix_adjoint_relative_error":block_adjoint_error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    settings = json.loads(Path(__file__).with_name("scenario.json").read_text(encoding="utf-8"))
    public = {key:settings[key] for key in ("waveform_design","prefix_policy","sampling_model",
              "receiver_information_policy","pilot_comparison_policy","receiver_equalization_policy")}
    # Once H_hat exists, data preparation needs only the public pilot policy.
    pilot_settings = {"pilot_comparison_policy":public["pilot_comparison_policy"]}
    count = public["prefix_policy"]["transmitted_samples_per_frame"]
    fs = public["sampling_model"]["sample_rate_hz"]
    rng = np.random.Generator(np.random.PCG64(20261104))
    data = (rng.choice([-3,-1,1,3],12288)+1j*rng.choice([-3,-1,1,3],12288))/np.sqrt(10)
    probe = (rng.standard_normal((256,64))+1j*rng.standard_normal((256,64)))/np.sqrt(2)
    grids = {scheme:build_grid(scheme,data,scenario=public) for scheme in SCHEMES}
    signals = {scheme:transmit(scheme,grids[scheme],public) for scheme in SCHEMES}
    masks = {scheme:build_masks(scheme,scenario=public) for scheme in SCHEMES}
    profile = settings["channel_model"]["multipath_followup"]
    paths = profile["path_delay_power_doppler_profile"]
    delays = [p["excess_delay_samples_at_60mhz"] for p in paths]
    powers = 10**(np.array([p["relative_power_db_to_los"] for p in paths])/10)
    gains = np.sqrt(powers/powers.sum())*np.exp(1j*np.r_[0.0,rng.uniform(0,2*np.pi,2)])
    sweep = profile["doppler_effect_followup"]["sweep_magnitude_design"]
    controls,geometry = [],{}
    for label,magnitude in zip(sweep["condition_labels"],sweep["maximum_residual_doppler_levels_hz"]):
        frequencies = [0.0,magnitude,-magnitude]
        taps = path_taps(count,fs,delays,frequencies,gains)
        for scheme in SCHEMES:
            channel = build_channel_operator(scheme,public,taps)
            model = build_pilot_data_model(channel,pilot_settings)
            assert model.shape == (16384,12288)
            assert np.array_equal(model.embed_data(data)+model.pilot_grid,grids[scheme])
            assert np.array_equal(model.embed_data(data).reshape(-1,order="F")[model.data_indices],data)
            assert np.vdot(model.pilot_grid,model.pilot_grid).real == 4096
            observed = receive(scheme,apply_paths(signals[scheme],delays,gains,frequencies,fs),public)
            unchanged = observed.copy()
            corrected = model.subtract_pilots(observed)
            assert corrected.shape == (256,64) and np.array_equal(observed,unchanged)
            error = relative_error(corrected,model.apply(data))
            assert error < 1e-11
            data_response = model.apply(data)
            outside_data = float(np.linalg.norm(data_response[~masks[scheme]["data"]])**2
                                 /np.linalg.norm(data_response)**2)
            controls.append({"condition":label,"scheme":scheme,
                "matched_channel_pilot_subtraction_relative_error":error,
                "data_response_energy_on_receive_nondata_coordinates_fraction":outside_data,
                **check_linear_problem(model,data,probe)})
            if label == "A":
                if scheme == "OTFS":
                    geometry[scheme] = {"observation_count":16384,"data_unknown_count":12288,
                        "data_operator_shape":[16384,12288],"scope":"whole DD frame"}
                    try:
                        model.block_matrix(0)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("OTFS must retain the full-frame data problem.")
                else:
                    columns = [model.block_data_slice(b).stop-model.block_data_slice(b).start for b in range(64)]
                    assert sum(columns) == 12288
                    geometry[scheme] = {"observations_per_block":256,"unknown_counts_by_block":columns,
                        "data_unknown_count_per_frame":sum(columns),"scope":"64 independent native blocks"}
    # Deliberate gain mismatch is a mathematical control, not a newly chosen
    # channel-estimation error distribution or an oracle subtraction policy.
    mismatch = []
    taps = path_taps(count,fs,delays,[0.0,46875.0,-46875.0],gains)
    for scheme in SCHEMES:
        true_channel = build_channel_operator(scheme,public,taps)
        estimated_channel = build_channel_operator(scheme,public,(.9+.1j)*taps)
        model = build_pilot_data_model(estimated_channel,pilot_settings)
        observed = receive(scheme,serial_tap_channel(signals[scheme],taps),public)
        remaining = model.subtract_pilots(observed)-true_channel.apply(model.embed_data(data))
        expected = true_channel.apply(model.pilot_grid)-estimated_channel.apply(model.pilot_grid)
        error = relative_error(remaining,expected)
        fraction = float(np.linalg.norm(remaining)/np.linalg.norm(true_channel.apply(model.pilot_grid)))
        assert error < 1e-11 and fraction > .01
        mismatch.append({"scheme":scheme,"remaining_pilot_identity_relative_error":error,
                         "remaining_pilot_relative_L2_norm":fraction})
    print("Matched-channel, data-column/adjoint and pilot-mismatch controls passed; checking actual estimates.",flush=True)
    bem = build_pilot_ls_model(public)
    estimators = {"AFDM":build_afdm(public),"OTFS":build_otfs(public),"DD-a-OFDM":build_dd_a(public)}
    noise = (rng.standard_normal(count)+1j*rng.standard_normal(count))/np.sqrt(2)
    integration = []
    for label,magnitude in (("A",0.0),("B-4",46875.0)):
        for scheme in SCHEMES:
            received = apply_paths(signals[scheme],delays,gains,[0.0,magnitude,-magnitude],fs)
            received += noise*np.sqrt(1/(4*10**(20/10)))
            # Estimation runs first on the original samples. Data-stage pilot
            # subtraction never modifies the estimator's pilot observations.
            if scheme == "OFDM":
                estimate = bem.estimate(received)
                estimated_taps = bem.reconstruct_taps(estimate)
            else:
                estimate = estimators[scheme].estimate(received)
                estimated_taps = estimators[scheme].model.reconstruct_taps(estimate)
            model = build_pilot_data_model(build_channel_operator(scheme,public,estimated_taps),pilot_settings)
            observed = receive(scheme,received,public)
            original = observed.copy()
            corrected = model.subtract_pilots(observed)
            assert np.array_equal(observed,original) and corrected.shape == (256,64)
            pilot_signal = transmit(scheme,model.pilot_grid,public)
            independent_pilot = receive(scheme,serial_tap_channel(pilot_signal,estimated_taps),public)
            pilot_error = relative_error(model.pilot_response,independent_pilot)
            # Evaluate the residual identity with data truth outside the model.
            # It retains AWGN, data-channel mismatch AND pilot mismatch.
            expected = observed-receive(scheme,serial_tap_channel(signals[scheme],estimated_taps),public)
            error = relative_error(corrected-model.apply(data),expected)
            assert max(pilot_error,error) < 1e-11
            integration.append({"condition":label,"scheme":scheme,"diagnostic_Eb_N0_db":20,
                "estimated_pilot_serial_prediction_relative_error":pilot_error,
                "noise_and_channel_error_residual_identity_relative_error":error,
                **check_linear_problem(model,data,probe)})
    report = {
        "purpose":"Estimated-channel known-pilot subtraction and data-only linear problem preparation; no MMSE solve, symbol decisions or BER.",
        "numpy_version":np.__version__,"diagnostic_seed":20261104,
        "receiver_inputs":"Construction/subtraction receive an estimated native channel, public mask/fixed pilots and received observations only, without true channel, transmitted data, noise variance or decoder output. Forward/residual checks use data truth as evaluation probes outside pilot cancellation.",
        "derived_geometry":geometry,"matched_channel_controls":controls,"mismatched_channel_controls":mismatch,
        "actual_estimated_channel_integrations":integration,
        "observation_rule":"Keep all 16384 native receive cells; reduce transmit unknown columns only. Full TF/DAFT block receive rows remain 256; full DD observation count remains 16384.",
        "noise_rule_in_integrations":"Add complex AWGN once after serial path summation at the existing diagnostic Eb/N0=20 dB; do not give variance to data preparation.",
        "limits":[
            "Known-channel and deliberate gain-mismatch controls validate algebra only. Actual integrations use each receiver's own pilot estimates and retain estimation errors.",
            "Cancellation uses H_hat*p. The remaining (H_true-H_hat)*p is generally nonzero and is not declared white, independent or removed perfectly.",
            "No data-coordinate rescaling, per-frame power normalization, pilot-resource change, estimator feedback or true-data interference cancellation is added.",
            "Data covariance, observation-noise knowledge, residual-error statistics, MMSE regularization/solver/stopping, symbol detection and BER remain pending.",
            "These mathematical controls and eight small integrations are not a BER run, the common bit quota or a four-waveform ranking.",
        ],
    }
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("20 matched-channel controls, four pilot-mismatch controls and eight actual-estimate integrations passed.")
    print("Estimated-pilot subtraction and data-only operators are ready; no MMSE or BER has been run.")


if __name__ == "__main__":
    main()

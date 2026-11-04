"""Validate estimated native channel construction; no MMSE solve or data BER.

Truth is used for mathematical controls and the simulation channel only. The
integration cases construct operators from each receiver's own estimated taps.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import afdm
import otfs
from afdm_estimation import build_estimator as build_afdm
from channel_operators import NativeChannelOperator, build_channel_operator
from cyclic_prefix import remove_cyclic_prefix
from dd_a_estimation import build_estimator as build_dd_a
from ofdm_bem import build_pilot_ls_model, modulate as ofdm_modulate
from otfs_estimation import build_estimator as build_otfs
from pilot_layouts import build_grid
from review_afdm_estimator import apply_paths


SCHEMES = ("OFDM","AFDM","OTFS","DD-a-OFDM")


def relative_error(actual, expected):
    return float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-300))


def transmit(scheme, grid, settings):
    cp = settings["prefix_policy"]["prefix_samples_per_block"]
    if scheme == "AFDM":
        chirps = settings["waveform_design"][scheme]["chirp_parameters"]
        return afdm.modulate(grid,chirps["c1"],chirps["c2"],cp_length=cp)
    if scheme == "OTFS":
        return otfs.modulate(grid,cp_length=cp)
    return ofdm_modulate(grid,cp)


def receive(scheme, samples, settings):
    prefix = settings["prefix_policy"]
    n,cp = prefix["useful_samples_per_block"],prefix["prefix_samples_per_block"]
    blocks = prefix["transmitted_samples_per_frame"]//(n+cp)
    if scheme == "AFDM":
        chirps = settings["waveform_design"][scheme]["chirp_parameters"]
        return afdm.demodulate(samples,n,blocks,chirps["c1"],chirps["c2"],cp_length=cp)
    if scheme == "OTFS":
        return otfs.demodulate(samples,n,blocks,cp_length=cp)
    useful = remove_cyclic_prefix(samples,n,cp).reshape((n,blocks),order="F")
    return np.fft.fft(useful,axis=0,norm="ortho")


def serial_tap_channel(samples, taps):
    """Independent serial receive-time FIR, without cyclic block indexing."""
    result = np.zeros_like(samples)
    for delay in range(taps.shape[1]):
        result[delay:] += taps[delay:,delay]*samples[:samples.size-delay]
    return result


def path_taps(count, fs, delays, frequencies, gains, delay_count=6):
    result = np.zeros((count,delay_count),dtype=complex)
    time = np.arange(count)/fs
    for delay,frequency,gain in zip(delays,frequencies,gains):
        result[:,delay] += gain*np.exp(2j*np.pi*frequency*time)
    return result


def operator_checks(operator, grid, probe):
    forward,adjoint = operator.apply(grid),operator.adjoint(probe)
    scale = max(np.linalg.norm(forward)*np.linalg.norm(probe),
                np.linalg.norm(grid)*np.linalg.norm(adjoint),1e-300)
    inner_error = float(abs(np.vdot(forward,probe)-np.vdot(grid,adjoint))/scale)
    matrix_error = matrix_adjoint_error = 0.0
    if operator.scheme != "OTFS":
        for block in (0,31,63):
            matrix = operator.block_matrix(block)
            matrix_error = max(matrix_error,relative_error(matrix@grid[:,block],forward[:,block]))
            matrix_adjoint_error = max(matrix_adjoint_error,
                relative_error(matrix.conj().T@probe[:,block],adjoint[:,block]))
    assert max(inner_error,matrix_error,matrix_adjoint_error) < 1e-11
    return {"adjoint_inner_product_error":inner_error,"block_matrix_relative_error":matrix_error,
            "block_matrix_adjoint_relative_error":matrix_adjoint_error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    settings = json.loads(Path(__file__).with_name("scenario.json").read_text(encoding="utf-8"))
    public = {key:settings[key] for key in ("waveform_design","prefix_policy","sampling_model",
              "receiver_information_policy","pilot_comparison_policy","receiver_equalization_policy")}
    prefix = settings["prefix_policy"]
    n,cp,count = prefix["useful_samples_per_block"],prefix["prefix_samples_per_block"],prefix["transmitted_samples_per_frame"]
    blocks = count//(n+cp)
    fs = settings["sampling_model"]["sample_rate_hz"]
    rng = np.random.Generator(np.random.PCG64(20261104))
    data = (rng.choice([-3,-1,1,3],12288)+1j*rng.choice([-3,-1,1,3],12288))/np.sqrt(10)
    probe = (rng.standard_normal((n,blocks))+1j*rng.standard_normal((n,blocks)))/np.sqrt(2)
    grids = {scheme:build_grid(scheme,data,scenario=public) for scheme in SCHEMES}
    signals = {scheme:transmit(scheme,grids[scheme],public) for scheme in SCHEMES}
    profile = settings["channel_model"]["multipath_followup"]
    paths = profile["path_delay_power_doppler_profile"]
    delays = [p["excess_delay_samples_at_60mhz"] for p in paths]
    powers = 10**(np.array([p["relative_power_db_to_los"] for p in paths])/10)
    gains = np.sqrt(powers/powers.sum())*np.exp(1j*np.r_[0.0,rng.uniform(0,2*np.pi,2)])
    sweep = profile["doppler_effect_followup"]["sweep_magnitude_design"]
    controls = []
    for label,magnitude in zip(sweep["condition_labels"],sweep["maximum_residual_doppler_levels_hz"]):
        frequencies = [0.0,magnitude,-magnitude]
        taps = path_taps(count,fs,delays,frequencies,gains)
        for scheme in SCHEMES:
            operator = build_channel_operator(scheme,public,taps)
            expected = receive(scheme,apply_paths(signals[scheme],delays,gains,frequencies,fs),public)
            error = relative_error(operator.apply(grids[scheme]),expected)
            assert error < 1e-11
            controls.append({"condition":label,"scheme":scheme,"serial_channel_relative_error":error,
                             **operator_checks(operator,grids[scheme],probe)})
    # General time-varying six-tap controls also cover nonparametric OFDM BEM
    # estimates; these random taps are not a newly selected physical channel.
    random_taps = (rng.standard_normal((count,6))+1j*rng.standard_normal((count,6)))/np.sqrt(12)
    general = []
    for scheme in SCHEMES:
        operator = build_channel_operator(scheme,public,random_taps)
        expected = receive(scheme,serial_tap_channel(signals[scheme],random_taps),public)
        error = relative_error(operator.apply(grids[scheme]),expected)
        assert error < 1e-11
        general.append({"scheme":scheme,"serial_channel_relative_error":error,
                        **operator_checks(operator,grids[scheme],probe)})
    # A small complete dense reference is safe here; never allocate the
    # 16384-by-16384 DD matrix for the production frame.
    small_n,small_b,small_cp = 8,4,2
    small_taps = (rng.standard_normal((40,3))+1j*rng.standard_normal((40,3)))/np.sqrt(6)
    small = NativeChannelOperator("OTFS",small_taps,block_length=small_n,
                 block_count=small_b,cp_length=small_cp,delays=range(3))
    dense = np.zeros((32,32),dtype=complex)
    for column in range(32):
        basis = np.eye(32,dtype=complex)[:,column].reshape((small_n,small_b),order="F")
        serial = otfs.modulate(basis,cp_length=small_cp)
        dense[:,column] = otfs.demodulate(serial_tap_channel(serial,small_taps),
                            small_n,small_b,cp_length=small_cp).reshape(-1,order="F")
    vector = rng.standard_normal((small_n,small_b))+1j*rng.standard_normal((small_n,small_b))
    small_forward = relative_error(small.apply(vector).reshape(-1,order="F"),dense@vector.reshape(-1,order="F"))
    small_adjoint = relative_error(small.adjoint(vector).reshape(-1,order="F"),dense.conj().T@vector.reshape(-1,order="F"))
    assert max(small_forward,small_adjoint) < 1e-11
    # Nonzero fractional Doppler spreads the selected OTFS pilot over all
    # Doppler bins; no receive-kernel truncation is permitted by this operator.
    spreading_taps = path_taps(count,fs,[2],[11718.75],[1.0])
    otfs_operator = build_channel_operator("OTFS",public,spreading_taps)
    impulse = np.zeros((n,blocks),dtype=complex); impulse[128,32] = 1
    nonzero_doppler_bins = int(np.count_nonzero(abs(otfs_operator.apply(impulse)[130]) > 1e-12))
    assert nonzero_doppler_bins == 64
    # A too-short prefix must not silently create an incorrect block model.
    short_cp_rejected = False
    try:
        NativeChannelOperator("OFDM",np.zeros(((n+4)*blocks,6)),block_length=n,
                              block_count=blocks,cp_length=4,delays=range(6))
    except ValueError:
        short_cp_rejected = True
    assert short_cp_rejected
    # Receiver integration: operators consume each estimator's output only.
    print("Serial-channel, dense block/DD and adjoint controls passed; checking estimated-tap integration.",flush=True)
    bem = build_pilot_ls_model(public)
    estimators = {"AFDM":build_afdm(public),"OTFS":build_otfs(public),"DD-a-OFDM":build_dd_a(public)}
    noise = (rng.standard_normal(count)+1j*rng.standard_normal(count))/np.sqrt(2)
    integration = []
    for label,magnitude in (("A",0.0),("B-4",46875.0)):
        for scheme in SCHEMES:
            received = apply_paths(signals[scheme],delays,gains,[0,magnitude,-magnitude],fs)
            received += noise*np.sqrt(1/(4*10**(20/10)))
            if scheme == "OFDM":
                estimate = bem.estimate(received)
                estimated_taps = bem.reconstruct_taps(estimate)
            else:
                estimate = estimators[scheme].estimate(received)
                estimated_taps = estimators[scheme].model.reconstruct_taps(estimate)
            operator = build_channel_operator(scheme,public,estimated_taps)
            expected = receive(scheme,serial_tap_channel(signals[scheme],estimated_taps),public)
            error = relative_error(operator.apply(grids[scheme]),expected)
            assert error < 1e-11
            integration.append({"condition":label,"scheme":scheme,"diagnostic_Eb_N0_db":20,
                                "estimated_channel_prediction_relative_error":error,
                                **operator_checks(operator,grids[scheme],probe)})
    report = {
        "purpose":"Native estimated-channel construction and adjoint checks for the agreed processing scope; no pilot cancellation, MMSE solve, data decisions or BER.",
        "numpy_version":np.__version__,"diagnostic_seed":20261104,
        "receiver_scope":"Public settings and receiver-reconstructed estimated taps only. True paths are used in mathematical controls and the simulation channel, never copied into receiver integration operators.",
        "grid_shape":[n,blocks],"block_native_matrix_shape":[n,n],"block_native_matrix_count":blocks,
        "OTFS_conceptual_operator_shape":list(otfs_operator.shape),
        "OTFS_stored_useful_tap_array_bytes":otfs_operator.taps.nbytes,
        "OTFS_hypothetical_dense_matrix_bytes":(n*blocks)**2*np.dtype(complex).itemsize,
        "known_model_A_B_controls":controls,"general_time_varying_six_tap_controls":general,
        "small_OTFS_dense_reference":{"grid_shape":[small_n,small_b],"matrix_shape":[32,32],
            "forward_relative_error":small_forward,"adjoint_relative_error":small_adjoint},
        "fractional_Doppler_OTFS_nonzero_Doppler_bins":nonzero_doppler_bins,
        "short_prefix_negative_control_rejected":short_cp_rejected,
        "estimated_tap_integration":integration,
        "limits":[
            "Known channel controls validate algebra only; integration uses actual pilot estimates with their errors retained. Matching an operator to its estimated serial channel does not prove channel-estimation accuracy.",
            "Per-slot CP-covered integer delays and aligned receive windows are required. Fractional delays, timing uncertainty or filters need a separate operator review.",
            "Native matrix sizes cover all useful grid cells before a data-only reduction. Pilot treatment, unknown-data covariance, noise-variance knowledge, MMSE solver/tolerances and complete BER remain pending.",
            "All modeled couplings are retained; the implicit DD representation introduces no Doppler-kernel truncation or one-tap equalizer approximation.",
            "These are small implementation controls, not the fixed BER bit quota, an MMSE performance comparison or a four-waveform ranking.",
        ],
    }
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("All 20 A/B native controls, four general-tap controls and eight estimated-channel integrations passed.")
    print("Native channel operators are implemented; MMSE data recovery and BER have not been run.")


if __name__ == "__main__":
    main()

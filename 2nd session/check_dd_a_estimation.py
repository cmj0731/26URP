"""Check the selected full-frame DD-a-OFDM path estimator, without data BER.

Known-parameter fits are algebraic controls only. In the actual search cases,
the receiver gets public design bounds and received samples, never generator
path parameters, data, phases or the receiver-noise variance.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from ofdm_bem import modulate
from cyclic_prefix import remove_cyclic_prefix
from dd_a_estimation import build_estimator
from pilot_layouts import build_grid
from review_afdm_estimator import apply_paths


def match_frequency_error(estimate, truth):
    errors = []
    for order in itertools.permutations(range(len(truth))):
        if all(estimate[i][0] == truth[j][0] for i,j in enumerate(order)):
            errors.append(max(abs(estimate[i][1]-truth[j][1]) for i,j in enumerate(order)))
    return min(errors) if errors else None


def taps_from_paths(length, delay_count, fs, paths, gains):
    taps = np.zeros((length,delay_count),dtype=complex)
    time = np.arange(length)/fs
    for (delay,frequency),gain in zip(paths,gains):
        taps[:,delay] += gain*np.exp(2j*np.pi*frequency*time)
    return taps


def direct_component_tf(model, paths, gains):
    """Evaluation-only self terms from independent dense tone actions.

    No actual path parameters from this helper are passed to the receiver.
    """
    q = np.arange(model.length)
    m = model.frequency_rows
    result = np.zeros(model.shape,dtype=complex)
    for (delay,frequency),gain in zip(paths,gains):
        shifted_tones = np.exp(2j*np.pi*((q-delay) % model.length)[:,None]
                               * m[None,:]/model.length)/np.sqrt(model.length)
        received = shifted_tones*np.exp(2j*np.pi*frequency*q/model.fs)[:,None]
        tones = np.fft.fft(received,axis=0,norm="ortho")
        diagonal = tones[m,np.arange(m.size)]
        result += gain*diagonal[:,None]*np.exp(2j*np.pi*frequency*model.starts)[None,:]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",type=int,default=16)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be positive")
    root = Path(__file__).resolve().parent
    settings = json.loads((root/"scenario.json").read_text(encoding="utf-8"))
    public = {key:settings[key] for key in ("waveform_design","prefix_policy","sampling_model",
                                          "receiver_information_policy","pilot_comparison_policy",
                                          "receiver_equalization_policy")}
    estimator = build_estimator(public)
    model = estimator.model
    assert estimator.dictionary.shape == (4096,642)
    assert estimator.grid.size == 107 and 0.0 in estimator.grid
    assert (estimator.grid[0],estimator.grid[-1]) == (-46875.0,46875.0)
    assert estimator.tolerance == 1.0
    assert model.starts[-1] == ((model.length+model.cp)*62+model.cp)/model.fs
    assert estimator.outer_sweeps == 3
    pilot_grid = build_grid("DD-a-OFDM",np.zeros(12288,dtype=complex),scenario=public)
    pilot_samples = modulate(pilot_grid,cp_length=model.cp)
    # Validate one normalized period against a zero-filled full-grid SFFT.
    rng = np.random.Generator(np.random.PCG64(20261105))
    z = rng.standard_normal(model.shape)+1j*rng.standard_normal(model.shape)
    full = np.zeros((model.length,model.blocks),dtype=complex)
    full[np.ix_(model.frequency_rows,model.time_columns)] = z
    periodic = np.fft.fft(np.fft.ifft(full,axis=0,norm="ortho"),axis=1,norm="ortho")
    normalized_period = 2*periodic[:model.shape[0],:model.shape[1]]
    period_error = float(np.linalg.norm(model.to_dd(z)-normalized_period)/np.linalg.norm(model.to_dd(z)))
    energy_error = abs(float(np.linalg.norm(model.to_dd(z))**2/np.linalg.norm(z)**2)-1)
    assert max(period_error,energy_error) < 1e-12
    # A serial single-tone channel isolates the desired self-subcarrier term;
    # the full multi-pilot serial channel also has ICI, which is NOT model error
    # to erase by silently changing the selected paper estimator.
    probes = np.unique(np.r_[np.linspace(-46875,46875,5),-31415.9,-11718.75,-2343.75,
                             -.125,.125,2343.75,11718.75,12555.55])
    maximum_direct_error = maximum_serial_tone_error = 0.0
    tone_grid = np.zeros((model.length,model.blocks),dtype=complex)
    tone_row = 7
    tone_grid[model.frequency_rows[tone_row],model.time_columns] = model.pilots[tone_row]
    tone_samples = modulate(tone_grid,cp_length=model.cp)
    for delay in model.delays:
        for frequency in probes:
            paths = [(delay,float(frequency))]
            direct_tf = direct_component_tf(model,paths,[1.0])
            expected = model.to_dd(direct_tf).reshape(-1,order="F")
            maximum_direct_error = max(maximum_direct_error,
                float(np.linalg.norm(expected-model.atoms(paths)[:,0])/np.linalg.norm(expected)))
            received = apply_paths(tone_samples,[delay],[1.0],[frequency],model.fs)
            useful = remove_cyclic_prefix(received,model.length,model.cp)
            tf = np.fft.fft(useful.reshape((model.length,model.blocks),order="F"),axis=0,norm="ortho")
            observed_self = tf[model.frequency_rows[tone_row],model.time_columns]/model.pilots[tone_row]
            maximum_serial_tone_error = max(maximum_serial_tone_error,
                float(np.linalg.norm(observed_self-direct_tf[tone_row])/np.linalg.norm(observed_self)))
    assert max(maximum_direct_error,maximum_serial_tone_error) < 1e-10
    powers = np.array([1,.1,10**(-1.5)])
    control_gains = np.sqrt(powers/powers.sum())*np.exp(1j*np.array([0.0,.4,1.1]))
    control_paths = [((0,0.0),(2,11718.75),(5,-11718.75)),
                     ((1,12555.55),(3,-31415.9),(4,7777.25)),
                     ((0,-20000.25),(0,15000.75),(4,-35000.5)),
                     ((0,-46875.0),(2,0.0),(5,46875.0))]
    controls = []
    for paths in control_paths:
        direct_tf = direct_component_tf(model,paths,control_gains)
        observations = model.to_dd(direct_tf).reshape(-1,order="F")
        predicted = model.atoms(paths)@control_gains
        matrix_error = float(np.linalg.norm(observations-predicted)/np.linalg.norm(observations))
        conditional = model.fit_gains(observations,paths)
        gain_error = float(np.linalg.norm(conditional["gains"]-control_gains)/np.linalg.norm(control_gains))
        assert max(matrix_error,gain_error) < 1e-10
        # The clean-model search gets observations only, not the hypotheses above.
        searched = estimator.fit_observations(observations)
        outer = searched["outer_history"]
        trace = [searched["initial_estimate"]["residual_power"]]+[v["joint_LS_residual_power"] for v in outer]
        assert len(outer) == 3 and all(len(v["path_updates"]) == 3 for v in outer)
        assert np.max(np.diff(trace)) < 1e-8
        frequency_error = match_frequency_error(searched["paths"],paths)
        assert frequency_error is not None and frequency_error < 897.9885057471264/2
        true_taps = taps_from_paths(model.sample_count,max(model.delays)+1,model.fs,paths,control_gains)
        conditional_taps = model.reconstruct_taps(conditional)
        tap_error = float(np.linalg.norm(conditional_taps-true_taps)/np.linalg.norm(true_taps))
        assert tap_error < 1e-10
        controls.append({"true_paths":paths,"direct_model_matrix_relative_error":matrix_error,
                         "conditional_known_hypothesis_gain_relative_error":gain_error,
                         "conditional_tap_relative_error":tap_error,
                         "searched_paths":searched["paths"],"search_maximum_frequency_error_hz":frequency_error})

    profile = settings["channel_model"]["multipath_followup"]
    paths = profile["path_delay_power_doppler_profile"]
    delays = [p["excess_delay_samples_at_60mhz"] for p in paths]
    powers = 10**(np.array([p["relative_power_db_to_los"] for p in paths])/10)
    magnitudes = np.sqrt(powers/powers.sum())
    sweep = profile["doppler_effect_followup"]["sweep_magnitude_design"]
    labels,levels = sweep["condition_labels"],sweep["maximum_residual_doppler_levels_hz"]
    eb_n0 = [None,*settings["evaluation"]["eb_n0_exploratory_sweep_db"]]
    records = [[{"nmse":[],"initial_nmse":[],"frequency":[],"matched":0,"last_change":[]} for _ in eb_n0] for _ in levels]
    interference = [{"known_pilot_ici":[],"data_ici":[]} for _ in levels]
    streams = np.random.SeedSequence(20261104).spawn(4)
    data_rng,phase_rng,noise_rng = [np.random.Generator(np.random.PCG64(s)) for s in streams[:3]]
    useful_indices = (np.arange(model.blocks)[:,None]*(model.length+model.cp)+model.cp
                      + np.arange(model.length)[None,:]).reshape(-1)
    maximum_bracket = 0.0
    maximum_residual_increase = 0.0
    for trial in range(args.trials):
        data = (data_rng.choice([-3,-1,1,3],12288)+1j*data_rng.choice([-3,-1,1,3],12288))/np.sqrt(10)
        grid = build_grid("DD-a-OFDM",data,scenario=public)
        samples = modulate(grid,cp_length=model.cp)
        gains = magnitudes*np.exp(1j*np.r_[0.0,phase_rng.uniform(0,2*np.pi,2)])
        noise = (noise_rng.standard_normal(model.sample_count)+1j*noise_rng.standard_normal(model.sample_count))/np.sqrt(2)
        for i,magnitude in enumerate(levels):
            truth = tuple(zip(delays,[0.0,magnitude,-magnitude]))
            received = apply_paths(samples,delays,gains,[p[1] for p in truth],model.fs)
            true_taps = taps_from_paths(model.sample_count,max(model.delays)+1,model.fs,truth,gains)[useful_indices]
            pilot_received = apply_paths(pilot_samples,delays,gains,[p[1] for p in truth],model.fs)
            pilot_observations = model.observations(pilot_received)
            desired = model.atoms(truth)@gains
            desired_power = float(np.vdot(desired,desired).real)
            interference[i]["known_pilot_ici"].append(float(np.linalg.norm(pilot_observations-desired)**2/desired_power))
            interference[i]["data_ici"].append(float(np.linalg.norm(model.observations(received)-pilot_observations)**2/desired_power))
            for j,db in enumerate(eb_n0):
                values = received if db is None else received+noise*np.sqrt(1/(4*10**(db/10)))
                result = estimator.estimate(values)
                assert len(result["paths"]) == model.path_count and result["rank"] == model.path_count
                assert np.isfinite(result["gains"]).all()
                assert all(delay in model.delays and abs(frequency) <= model.maximum_doppler for delay,frequency in result["paths"])
                history = result["search_history"]
                outer = result["outer_history"]
                assert len(history) == 3 and len(outer) == 3
                assert all(len(v["path_updates"]) == 3 for v in outer)
                all_searches = history+[step for sweep in outer for step in sweep["path_updates"]]
                maximum_bracket = max(maximum_bracket,max(v["refinement_final_interval_hz"] for v in all_searches))
                residuals = np.array([v["joint_LS_residual_power"] for v in history+outer])
                maximum_residual_increase = max(maximum_residual_increase,float(np.max(np.diff(residuals))))
                estimated_taps = model.reconstruct_taps(result)[useful_indices]
                nmse = float(np.linalg.norm(estimated_taps-true_taps)**2/np.linalg.norm(true_taps)**2)
                initial_taps = model.reconstruct_taps(result["initial_estimate"])[useful_indices]
                initial_nmse = float(np.linalg.norm(initial_taps-true_taps)**2/np.linalg.norm(true_taps)**2)
                error = match_frequency_error(result["paths"],truth)
                record = records[i][j]
                record["nmse"].append(nmse)
                record["initial_nmse"].append(initial_nmse)
                record["last_change"].append(float((residuals[-2]-residuals[-1])/max(float(np.vdot(values,values).real),1e-300)))
                if error is not None:
                    record["matched"] += 1
                    record["frequency"].append(error)
        if (trial+1)%4 == 0 or trial+1 == args.trials:
            print(f"Completed {trial+1}/{args.trials} DD-a-OFDM diagnostic frames per condition.",flush=True)
    assert maximum_bracket <= estimator.tolerance
    assert maximum_residual_increase < 1e-8
    results = []
    for label,magnitude,condition in zip(labels,levels,records):
        points = []
        for db,record in zip(eb_n0,condition):
            mean = float(np.mean(record["nmse"]))
            points.append({"received_Eb_N0_dB":db,"no_AWGN":db is None,
                           "mean_channel_NMSE":mean,"mean_channel_NMSE_db":float(10*np.log10(max(mean,1e-300))),
                           "mean_initial_channel_NMSE":float(np.mean(record["initial_nmse"])),
                           "mean_last_sweep_residual_decrease_over_observation_energy":float(np.mean(record["last_change"])),
                           "frames_with_correct_delay_multiset":record["matched"],
                           "mean_maximum_frequency_error_hz_when_delays_match":float(np.mean(record["frequency"])) if record["frequency"] else None})
        ratios = interference[len(results)]
        results.append({"condition":label,"maximum_residual_doppler_hz":magnitude,
                        "mean_known_pilot_ICI_to_desired_power":float(np.mean(ratios["known_pilot_ici"])),
                        "mean_unknown_data_ICI_to_desired_power":float(np.mean(ratios["data_ici"])),"points":points})
    report = {
        "purpose":"Implementation diagnostic for the selected DD-a-OFDM joint-frame approximate-ML/LS estimator; no equalization or BER.",
        "numpy_version":np.__version__,"diagnostic_seed":20261104,"diagnostic_frames_per_condition":args.trials,
        "receiver_input_scope":"Public settings projection excluding channel_model, evaluation and noise_model, plus received samples only. Generator truth is confined to the channel, algebraic controls and result evaluator.",
        "processing_window_blocks":model.blocks,"pilot_observation_count":model.observation_count,
        "coarse_dictionary_shape":list(estimator.dictionary.shape),"coarse_frequency_count":int(estimator.grid.size),
        "DD_period_shape":list(model.shape),
        "outer_sweeps_after_initialization":estimator.outer_sweeps,
        "initial_grid_spacing_hz":1/(4*model.sample_count/model.fs),
        "fixed_residual_doppler_bound_hz":model.maximum_doppler,
        "numerical_refinement_tolerance_hz":estimator.tolerance,"maximum_final_refinement_interval_hz":maximum_bracket,
        "maximum_joint_LS_residual_power_increase":maximum_residual_increase,
        "algebraic_controls":controls,
        "observation_model_checks":{
            "diagnostic_probe_seed":20261105,
            "single_path_probe_count":len(model.delays)*int(probes.size),
            "normalized_period_relative_error":period_error,
            "unitary_DD_energy_relative_error":energy_error,
            "maximum_independent_direct_term_relative_error":maximum_direct_error,
            "maximum_serial_single_tone_relative_error":maximum_serial_tone_error,
            "scope":"Independent dense tone responses and serial single-tone diagonal controls. The full serial pilot/data channel includes off-diagonal ICI, intentionally retained in the estimator residual."},
        "noise_rule":"Inject complex AWGN once after serial path summation, variance 1/(4*10^(Eb/N0/10)); do not give that variance to the estimator.",
        "pairing":"Replay the independent PCG64 data/initial-phase/noise streams used by the OFDM and AFDM diagnostics; pair realizations across A/B and Eb/N0. Fixed known pilots retain their independent agreed seed.",
        "source_scope":"Paper equations (19)-(25), Algorithm 1 path-wise AP and joint LS; our selected sequential initialization, quarter-bin grid, one-Hz refinement and fixed three outer sweeps. Physical gain reference includes the within-block diagonal Doppler coefficient and CP clock.",
        "limits":[
            "Three sequential initial paths are followed by exactly three full AP sweeps; global ML optimality is not claimed and no post-hoc condition-specific tuning is performed.",
            "One Hz is the numerical local-search bracket tolerance, not a physical Doppler bin or an estimation-accuracy guarantee.",
            "Known-hypothesis LS fits and the alternate/same-delay profiles are mathematical implementation controls, not extra selected physical experiments or receiver oracles.",
            "Channel NMSE and parameter recovery are diagnostics, not BER or a four-waveform ranking. Residual estimator errors are retained for future BER evaluation rather than optimized away here.",
            "Off-diagonal ICI from known pilots and unknown data is kept in the residual; the paper's independent Gaussian interference assumptions are not proven for our fixed QPSK pilots and 16-QAM data.",
            "The selected three sweeps are checked through their residual trace; these small diagnostics do not establish universal convergence or sufficient iteration count.",
            "Sixteen diagnostic frames are not a new BER stopping rule or bit quota. TF data-channel matrices, noise-variance knowledge for equalization, MMSE data recovery and BER remain pending.",
        ],"results":results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("Direct-model, alternate-delay, same-delay and endpoint controls passed; fixed initialization and three AP sweeps preserved.")
    for result in results:
        first,last = result["points"][0],result["points"][-1]
        print(f"{result['condition']}: no-AWGN delay matches {first['frames_with_correct_delay_multiset']}/{args.trials}; 30-dB matches {last['frames_with_correct_delay_multiset']}/{args.trials}")
    print("DD-a-OFDM path estimation is implemented; no BER has been calculated.")


if __name__ == "__main__":
    main()

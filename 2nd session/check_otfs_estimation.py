"""Check the selected full-frame OTFS path estimator, without data BER.

Known-parameter fits are algebraic controls only. In the actual search cases,
the receiver gets public design bounds and received samples, never generator
path parameters, data, phases or the receiver-noise variance.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import otfs
from otfs_estimation import build_estimator
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
    assert estimator.dictionary.shape == (384,642)
    assert estimator.grid.size == 107 and 0.0 in estimator.grid
    assert (estimator.grid[0],estimator.grid[-1]) == (-46875.0,46875.0)
    assert estimator.tolerance == 1.0
    assert model.starts[-1] == ((model.length+model.cp)*(model.blocks-1)+model.cp)/model.fs
    pilot_grid = build_grid("OTFS",np.zeros(12288,dtype=complex),scenario=public)
    pilot_samples = otfs.modulate(pilot_grid,cp_length=model.cp)
    assert np.array_equal(model.rows,np.arange(128,134))
    probe_rng = np.random.Generator(np.random.PCG64(20261105))
    probe_data = (probe_rng.choice([-3,-1,1,3],12288)
                  + 1j*probe_rng.choice([-3,-1,1,3],12288))/np.sqrt(10)
    probe_grid = build_grid("OTFS",probe_data,scenario=public)-pilot_grid
    probe_samples = otfs.modulate(probe_grid,cp_length=model.cp)
    probe_frequencies = np.unique(np.r_[np.linspace(-46875,46875,5),
                                        -31415.9,-11718.75,-2343.75,-.125,
                                        .125,2343.75,11718.75,12555.55])
    maximum_template_error = maximum_data_leakage = maximum_energy_error = 0.0
    for delay in model.delays:
        for frequency in probe_frequencies:
            received = apply_paths(pilot_samples,[delay],[1.0],[frequency],model.fs)
            observed = model.observations(received)
            predicted = model.atoms([(delay,float(frequency))])[:,0]
            maximum_template_error = max(maximum_template_error,
                float(np.linalg.norm(observed-predicted)/np.linalg.norm(observed)))
            maximum_energy_error = max(maximum_energy_error,
                abs(float(np.vdot(predicted,predicted).real)/4096-1))
            data_received = apply_paths(probe_samples,[delay],[1.0],[frequency],model.fs)
            maximum_data_leakage = max(maximum_data_leakage,
                float(np.linalg.norm(model.observations(data_received))/np.linalg.norm(observed)))
    assert max(maximum_template_error,maximum_data_leakage,maximum_energy_error) < 1e-10
    # Negative control: dropping the prefix time produces the wrong response.
    wrong_times = (model.length*np.arange(model.blocks)+model.pilot_delay+5)/model.fs
    wrong_response = np.fft.fft(model.pilot/np.sqrt(model.blocks)
        * np.exp(2j*np.pi*model.pilot_doppler*np.arange(model.blocks)/model.blocks)
        * np.exp(2j*np.pi*46875*wrong_times),norm="ortho")
    correct = model.atoms([(5,46875.0)]).reshape((6,64),order="F")[5]
    omitted_cp_error = float(np.linalg.norm(correct-wrong_response)/np.linalg.norm(correct))
    assert omitted_cp_error > .01
    powers = np.array([1,.1,10**(-1.5)])
    control_gains = np.sqrt(powers/powers.sum())*np.exp(1j*np.array([0.0,.4,1.1]))
    control_paths = [((0,0.0),(2,11718.75),(5,-11718.75)),
                     ((1,12555.55),(3,-31415.9),(4,7777.25)),
                     ((0,-20000.25),(0,15000.75),(4,-35000.5)),
                     ((0,-46875.0),(2,0.0),(5,46875.0))]
    controls = []
    for paths in control_paths:
        received = apply_paths(pilot_samples,[p[0] for p in paths],control_gains,[p[1] for p in paths],model.fs)
        observations = model.observations(received)
        predicted = model.atoms(paths)@control_gains
        matrix_error = float(np.linalg.norm(observations-predicted)/np.linalg.norm(observations))
        conditional = model.fit_gains(observations,paths)
        gain_error = float(np.linalg.norm(conditional["gains"]-control_gains)/np.linalg.norm(control_gains))
        assert max(matrix_error,gain_error) < 1e-10
        # The search receives only the serial samples, not the hypotheses above.
        searched = estimator.estimate(received)
        frequency_error = match_frequency_error(searched["paths"],paths)
        assert frequency_error is not None and frequency_error < 897.9885057471264/2
        true_taps = taps_from_paths(model.sample_count,max(model.delays)+1,model.fs,paths,control_gains)
        conditional_taps = model.reconstruct_taps(conditional)
        tap_error = float(np.linalg.norm(conditional_taps-true_taps)/np.linalg.norm(true_taps))
        assert tap_error < 1e-10
        controls.append({"true_paths":paths,"pilot_matrix_serial_relative_error":matrix_error,
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
    records = [[{"nmse":[],"frequency":[],"matched":0} for _ in eb_n0] for _ in levels]
    streams = np.random.SeedSequence(20261104).spawn(4)
    data_rng,phase_rng,noise_rng = [np.random.Generator(np.random.PCG64(s)) for s in streams[:3]]
    useful_indices = (np.arange(model.blocks)[:,None]*(model.length+model.cp)+model.cp
                      + np.arange(model.length)[None,:]).reshape(-1)
    maximum_bracket = 0.0
    maximum_residual_increase = 0.0
    for trial in range(args.trials):
        data = (data_rng.choice([-3,-1,1,3],12288)+1j*data_rng.choice([-3,-1,1,3],12288))/np.sqrt(10)
        grid = build_grid("OTFS",data,scenario=public)
        samples = otfs.modulate(grid,cp_length=model.cp)
        gains = magnitudes*np.exp(1j*np.r_[0.0,phase_rng.uniform(0,2*np.pi,2)])
        noise = (noise_rng.standard_normal(model.sample_count)+1j*noise_rng.standard_normal(model.sample_count))/np.sqrt(2)
        for i,magnitude in enumerate(levels):
            truth = tuple(zip(delays,[0.0,magnitude,-magnitude]))
            received = apply_paths(samples,delays,gains,[p[1] for p in truth],model.fs)
            true_taps = taps_from_paths(model.sample_count,max(model.delays)+1,model.fs,truth,gains)[useful_indices]
            for j,db in enumerate(eb_n0):
                values = received if db is None else received+noise*np.sqrt(1/(4*10**(db/10)))
                result = estimator.estimate(values)
                assert len(result["paths"]) == model.path_count and result["rank"] == model.path_count
                assert np.isfinite(result["gains"]).all()
                assert all(delay in model.delays and abs(frequency) <= model.maximum_doppler for delay,frequency in result["paths"])
                history = result["search_history"]
                maximum_bracket = max(maximum_bracket,max(v["refinement_final_interval_hz"] for v in history))
                residuals = np.array([v["joint_LS_residual_power"] for v in history])
                maximum_residual_increase = max(maximum_residual_increase,float(np.max(np.diff(residuals))))
                estimated_taps = model.reconstruct_taps(result)[useful_indices]
                nmse = float(np.linalg.norm(estimated_taps-true_taps)**2/np.linalg.norm(true_taps)**2)
                error = match_frequency_error(result["paths"],truth)
                record = records[i][j]
                record["nmse"].append(nmse)
                if error is not None:
                    record["matched"] += 1
                    record["frequency"].append(error)
        if (trial+1)%4 == 0 or trial+1 == args.trials:
            print(f"Completed {trial+1}/{args.trials} OTFS diagnostic frames per condition.",flush=True)
    assert maximum_bracket <= estimator.tolerance
    assert maximum_residual_increase < 1e-8
    results = []
    for label,magnitude,condition in zip(labels,levels,records):
        points = []
        for db,record in zip(eb_n0,condition):
            mean = float(np.mean(record["nmse"]))
            points.append({"received_Eb_N0_dB":db,"no_AWGN":db is None,
                           "mean_channel_NMSE":mean,"mean_channel_NMSE_db":float(10*np.log10(max(mean,1e-300))),
                           "frames_with_correct_delay_multiset":record["matched"],
                           "mean_maximum_frequency_error_hz_when_delays_match":float(np.mean(record["frequency"])) if record["frequency"] else None})
        results.append({"condition":label,"maximum_residual_doppler_hz":magnitude,"points":points})
    report = {
        "purpose":"Implementation diagnostic for the selected OTFS joint-frame approximate-ML/LS estimator; no equalization or BER.",
        "numpy_version":np.__version__,"diagnostic_seed":20261104,"diagnostic_frames_per_condition":args.trials,
        "receiver_input_scope":"Public settings projection excluding channel_model, evaluation and noise_model, plus received samples only. Generator truth is confined to the channel, algebraic controls and result evaluator.",
        "processing_window_blocks":model.blocks,"pilot_observation_count":model.rows.size*model.blocks,
        "coarse_dictionary_shape":list(estimator.dictionary.shape),"coarse_frequency_count":int(estimator.grid.size),
        "observed_delay_rows":model.rows.tolist(),
        "initial_grid_spacing_hz":1/(4*model.sample_count/model.fs),
        "fixed_residual_doppler_bound_hz":model.maximum_doppler,
        "numerical_refinement_tolerance_hz":estimator.tolerance,"maximum_final_refinement_interval_hz":maximum_bracket,
        "maximum_joint_LS_residual_power_increase":maximum_residual_increase,
        "algebraic_controls":controls,
        "native_pilot_template_checks":{
            "data_only_probe_seed":20261105,
            "single_path_probe_count":len(model.delays)*int(probe_frequencies.size),
            "maximum_serial_template_relative_error":maximum_template_error,
            "maximum_data_leakage_relative_to_pilot_L2":maximum_data_leakage,
            "maximum_pilot_energy_relative_error":maximum_energy_error,
            "omitted_CP_clock_negative_control_relative_error":omitted_cp_error,
            "scope":"All six public delay candidates, boundary/off-grid frequencies and data-only isolation probes. Alternate profiles are algebraic controls, not new selected physical experiments."},
        "noise_rule":"Inject complex AWGN once after serial path summation, variance 1/(4*10^(Eb/N0/10)); do not give that variance to the estimator.",
        "pairing":"Replay the independent PCG64 data/initial-phase/noise streams used by the OFDM and AFDM diagnostics; pair realizations across A/B and Eb/N0. Fixed known pilots retain their independent agreed seed.",
        "source_scope":"The embedded-pilot reference uses threshold-based estimation; this continuous pilot-template approximate-ML/joint-LS design is our selected adaptation.",
        "limits":[
            "The search is a fixed three-step sequential approximation to joint nonlinear ML; global optimality is not claimed and no post-hoc tuning is performed for a test condition.",
            "One Hz is the numerical local-search bracket tolerance, not a physical Doppler bin or an estimation-accuracy guarantee.",
            "Known-hypothesis LS fits and the alternate/same-delay profiles are mathematical implementation controls, not extra selected physical experiments or receiver oracles.",
            "Channel NMSE and parameter recovery are diagnostics, not BER or a four-waveform ranking. Residual estimator errors are retained for future BER evaluation rather than optimized away here.",
            "Sixteen diagnostic frames are not a new BER stopping rule or bit quota. DD data-channel matrices, noise-variance knowledge for equalization, MMSE data recovery and BER remain pending.",
        ],"results":results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("Exact-model, alternate-delay and same-delay controls passed; all selected search settings stayed fixed.")
    for result in results:
        first,last = result["points"][0],result["points"][-1]
        print(f"{result['condition']}: no-AWGN delay matches {first['frames_with_correct_delay_multiset']}/{args.trials}; 30-dB matches {last['frames_with_correct_delay_multiset']}/{args.trials}")
    print("OTFS path estimation is implemented; no BER has been calculated.")


if __name__ == "__main__":
    main()

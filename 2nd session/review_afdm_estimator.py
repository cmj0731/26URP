"""Review the AFDM embedded-pilot approximate-ML candidate without selecting it.

Check native transforms, a fractional-Doppler pilot template and a source-derived
receive window. There is no parameter search, channel estimate, data equalizer
or BER run. True paths are used only by this diagnostic's channel and evaluator.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import afdm
from cyclic_prefix import remove_cyclic_prefix
from pilot_layouts import build_grid, build_masks, known_pilot_symbols


def apply_paths(samples, delays, gains, frequencies, fs):
    """Serial channel on the full CP-inclusive frame; independent of templates."""
    result = np.zeros_like(samples)
    time = np.arange(samples.size) / fs
    for delay, gain, frequency in zip(delays, gains, frequencies):
        result[delay:] += gain * np.exp(2j * np.pi * frequency * time[delay:]) * samples[:samples.size-delay]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    settings = json.loads((root / "scenario.json").read_text(encoding="utf-8"))
    spec = settings["waveform_design"]["AFDM"]
    n, blocks = spec["daft_length"], spec["blocks_per_frame"]
    c1, c2 = spec["chirp_parameters"]["c1"], spec["chirp_parameters"]["c2"]
    cp = settings["prefix_policy"]["prefix_samples_per_block"]
    fs = settings["sampling_model"]["sample_rate_hz"]
    lmax = settings["receiver_information_policy"]["delay_support"]["maximum_excess_delay_samples"]
    design = spec["chirp_parameters"]["selection_rule"]["c1_numeric_formula"]
    radius = design["alpha_max"] + design["xi_nu"]
    slope = int(round(2*n*c1))
    assert slope == 2*radius + 1
    # Source VI's Tr,E under the fixed alpha/xi design: union of the five
    # neighboring bins around each possible pilot center -5*l, l=0,...,5.
    window = np.unique(np.arange(-slope*lmax-radius, radius+1) % n)
    masks = build_masks("AFDM", scenario=settings)
    naive_window = np.flatnonzero(~masks["data"][:, 0])
    assert window.size == (lmax+1)*slope == 30
    assert naive_window.size == 64
    data_count = int(masks["data"].sum())
    pilot_grid = build_grid("AFDM", np.zeros(data_count, dtype=complex), scenario=settings)
    pilot_samples = afdm.modulate(pilot_grid, c1, c2, cp_length=cp)
    pilot_useful = remove_cyclic_prefix(pilot_samples,n,cp)
    pilot_cp_samples = pilot_samples.reshape((blocks,n+cp))[:,:cp]
    pilots = known_pilot_symbols("AFDM", scenario=settings)
    starts = (np.arange(blocks)*(n+cp) + cp) / fs
    demodulate = lambda values: afdm.demodulate(values,n,blocks,c1,c2,cp_length=cp)
    seed = 20261104
    streams = np.random.SeedSequence(seed).spawn(4)
    data_rng, phase_rng = [np.random.Generator(np.random.PCG64(s)) for s in streams[:2]]
    sample_data = (data_rng.choice([-3,-1,1,3],data_count)
                   + 1j*data_rng.choice([-3,-1,1,3],data_count)) / np.sqrt(10)
    grid = build_grid("AFDM", sample_data, scenario=settings)
    full_samples = afdm.modulate(grid,c1,c2,cp_length=cp)
    roundtrip = float(np.linalg.norm(demodulate(full_samples)-grid)/np.linalg.norm(grid))
    useful = remove_cyclic_prefix(full_samples,n,cp)
    energy_error = abs(float(np.vdot(useful,useful).real/np.vdot(grid,grid).real)-1)
    index = np.arange(n)
    dense_idaft = (np.exp(2j*np.pi*c1*index**2)[:,None]
                   * np.exp(2j*np.pi*index[:,None]*index[None,:]/n)/np.sqrt(n)
                   * np.exp(2j*np.pi*c2*index**2)[None,:])
    dense_error = float(np.linalg.norm(dense_idaft@grid-useful.reshape((n,blocks),order="F"))
                        /np.linalg.norm(useful))
    assert max(roundtrip,energy_error,dense_error) < 1e-10

    profile = settings["channel_model"]["multipath_followup"]
    sweep = profile["doppler_effect_followup"]["sweep_magnitude_design"]
    maximum = sweep["maximum_residual_doppler_hz"]
    sentinels = [-maximum,-maximum/2,-maximum/20,0.0,maximum/20,maximum/2,maximum]
    template_error = 0.0
    maximum_unit_template_energy_error = 0.0
    for delay in range(lmax+1):
        for frequency in sentinels:
            observed = demodulate(apply_paths(pilot_samples,[delay],[1.0],[frequency],fs))
            response = afdm.pilot_path_response(n,c1,c2,delay,frequency,fs)
            predicted = response[:,None]*pilots[None,:]*np.exp(2j*np.pi*frequency*starts)[None,:]
            template_error = max(template_error,float(np.linalg.norm(observed-predicted)/np.linalg.norm(observed)))
            maximum_unit_template_energy_error = max(maximum_unit_template_energy_error,abs(float(np.vdot(response,response).real)-1))
        response0 = afdm.pilot_path_response(n,c1,c2,delay,0.0,fs)
        assert int(np.argmax(np.abs(response0))) == (-slope*delay) % n
    assert template_error < 1e-10
    assert maximum_unit_template_energy_error < 1e-12
    minimum_capture = 1.0
    capture_at = None
    for delay in range(lmax+1):
        for frequency in np.linspace(-maximum,maximum,401):
            response = afdm.pilot_path_response(n,c1,c2,delay,float(frequency),fs)
            capture = float(np.vdot(response[window],response[window]).real/np.vdot(response,response).real)
            if capture < minimum_capture:
                minimum_capture,capture_at = capture,{"delay_samples":delay,"doppler_hz":float(frequency)}

    paths = profile["path_delay_power_doppler_profile"]
    true_delays = [p["excess_delay_samples_at_60mhz"] for p in paths]
    powers = 10**(np.array([p["relative_power_db_to_los"] for p in paths])/10)
    magnitudes = np.sqrt(powers/powers.sum())
    trials = 16
    # Reset the data stream so these frames match the OFDM diagnostic's data
    # and initial phases; native grid placement still differs by waveform.
    data_rng = np.random.Generator(np.random.PCG64(streams[0]))
    ratios = [[[],[]] for _ in sweep["condition_labels"]]
    max_decomposition_error = 0.0
    for _ in range(trials):
        data = (data_rng.choice([-3,-1,1,3],data_count)
                + 1j*data_rng.choice([-3,-1,1,3],data_count))/np.sqrt(10)
        full_grid = build_grid("AFDM",data,scenario=settings)
        full_samples = afdm.modulate(full_grid,c1,c2,cp_length=cp)
        data_samples = full_samples-pilot_samples
        gains = magnitudes*np.exp(1j*np.r_[0.0,phase_rng.uniform(0,2*np.pi,2)])
        for i,magnitude in enumerate(sweep["maximum_residual_doppler_levels_hz"]):
            frequencies = [0.0,magnitude,-magnitude]
            received_pilot = demodulate(apply_paths(pilot_samples,true_delays,gains,frequencies,fs))
            received_data = demodulate(apply_paths(data_samples,true_delays,gains,frequencies,fs))
            received_full = demodulate(apply_paths(full_samples,true_delays,gains,frequencies,fs))
            max_decomposition_error = max(max_decomposition_error,
                float(np.linalg.norm(received_full-received_pilot-received_data)/np.linalg.norm(received_full)))
            for j,rows in enumerate((window,naive_window)):
                ratios[i][j].append(float(np.linalg.norm(received_data[rows])**2/np.linalg.norm(received_pilot[rows])**2))
    assert max_decomposition_error < 1e-12
    results = []
    for label,magnitude,values in zip(sweep["condition_labels"],sweep["maximum_residual_doppler_levels_hz"],ratios):
        templates = np.column_stack([afdm.pilot_path_response(n,c1,c2,l,f,fs)[window]
                                    for l,f in zip(true_delays,[0.0,magnitude,-magnitude])])
        templates /= np.linalg.norm(templates,axis=0)[None,:]
        gram = templates.conj().T@templates
        np.fill_diagonal(gram,0)
        entry = {"condition":label,"maximum_residual_doppler_hz":magnitude,
                 "source_window_data_to_pilot_received_energy_ratio_mean":float(np.mean(values[0])),
                 "naive_transmit_guard_window_data_to_pilot_received_energy_ratio_mean":float(np.mean(values[1])),
                 "normalized_path_template_maximum_offdiagonal_inner_product":float(np.max(np.abs(gram)))}
        results.append(entry)
    assert results[0]["source_window_data_to_pilot_received_energy_ratio_mean"] < 1e-20
    assert results[0]["naive_transmit_guard_window_data_to_pilot_received_energy_ratio_mean"] > 1e-4
    report = {
        "purpose":"Structural review of an AFDM embedded-pilot approximate-ML candidate; no estimator or BER is selected or run.",
        "reference_url":"https://arxiv.org/pdf/2204.12798", "reference_scope":"VI, equations (67)-(76), especially fractional Doppler VI-B. Their single AFDM frame corresponds to one of our 256-sample useful blocks, not our whole 64-block frame.",
        "numpy_version":np.__version__,"diagnostic_seed":seed,"diagnostic_frames_per_condition":trials,
        "diagnostic_rng":"Independent PCG64 data/phase streams from SeedSequence(20261104).spawn(4); replay the first two OFDM diagnostic streams.",
        "receive_window_status":"Source-derived proposal only; not a selected receiver-observation policy.",
        "source_window_indices":window.tolist(),"source_window_observations_per_block":int(window.size),
        "source_window_observations_per_64_block_frame":int(window.size*blocks),
        "source_window_rule":"R=alpha_max+xi_nu=2, S=2*N*c1=5; union of -S*l-R through -S*l+R modulo N for l=0,...,5. This is [-27,...,2] modulo 256.",
        "naive_transmit_nondata_indices":naive_window.tolist(),
        "fixed_pilot_energy_per_block":float(np.abs(pilots[0])**2),
        "fixed_total_pilot_useful_energy":float(np.vdot(pilot_useful,pilot_useful).real),
        "isolated_pilot_CP_energy_separately_measured":float(np.vdot(pilot_cp_samples,pilot_cp_samples).real),
        "transform_controls":{"roundtrip_relative_error":roundtrip,"useful_energy_relative_error":energy_error,"dense_IDAFT_relative_error":dense_error},
        "native_pilot_template_controls":{"serial_channel_relative_error":template_error,
                                           "maximum_unit_energy_error":maximum_unit_template_energy_error,
                                           "delay_count":lmax+1,"sentinel_frequencies_hz":sentinels},
        "maximum_full_equals_pilot_plus_data_relative_error":max_decomposition_error,
        "fractional_pilot_energy_capture":{"minimum_fraction_in_source_window":minimum_capture,
                                           "minimum_at":capture_at,"probe_frequency_count_per_delay":401},
        "native_sign_convention":"Channel exp(+j*2*pi*f*t); peak near (f*N/Fs-2*N*c1*l) modulo N. Unit template gain is referenced to first useful receive sample; across blocks multiply by exp(+j*2*pi*f*((N+CP)*b+CP)/Fs).",
        "limits":[
            "Source-window indices and per-block versus joint-frame processing are proposals, not chosen estimator settings. No Doppler grid, initialization, search refinement or stopping criterion is selected.",
            "The 401-frequency grid is a structural energy-capture probe over the existing maximum residual envelope; it is not a receiver search grid or supplied true Doppler.",
            "Fractional Doppler tails remain across all DAFT bins; the source's inter-path orthogonality and guard isolation are approximations for our finite margin. No leakage cancellation or guard enlargement is added.",
            "True paths are used only by the diagnostic channel and evaluator; no true active delay, Doppler, complex gain or transmitted data is passed to an estimator.",
            "Observed ratios measure data energy divided by pilot-component energy in a receive window. They are neither channel NMSE nor BER, and do not rank complete receivers.",
            "There is no received AWGN or path-parameter estimator in this structural check, and no DAFT data equalizer or BER detector. The agreed bit quota is unchanged.",
        ],"results":results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(f"AFDM transforms and fractional pilot response passed; proposed receive window has {window.size} bins/block.")
    for row in results:
        a,b = row["source_window_data_to_pilot_received_energy_ratio_mean"],row["naive_transmit_guard_window_data_to_pilot_received_energy_ratio_mean"]
        print(f"{row['condition']}: proposed window data/pilot {10*np.log10(max(a,1e-300)):.2f} dB; naive guard window {10*np.log10(b):.2f} dB")
    print("No path estimator, equalizer or BER run; candidate selection remains pending.")


if __name__ == "__main__":
    main()

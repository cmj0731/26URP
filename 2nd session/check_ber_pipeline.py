"""Check exact quotas, cluster statistics and accelerated solver equivalence."""

import json
from pathlib import Path

import numpy as np

from ber_statistics import frame_plan, summarize_frames
from channel_operators import NativeChannelOperator
from check_qam16_receiver import SmallMaskedModel
from mmse_demapping import effective_symbol_gain
from mmse_problem import PlugInMMSEProblem
from mmse_solver import solve_mmse_problem
from qam16 import hard_demodulate


def main():
    s = json.loads(Path(__file__).with_name("scenario.json").read_text(encoding="utf-8"))
    assert frame_plan(10485760,65536) == [65536]*160
    assert frame_plan(10485760,49152) == [49152]*213+[16384]
    zero = summarize_frames([0]*214,[49152]*213+[16384],full_frame_capacity=49152)
    assert zero["measured_information_bits"] == 10485760
    assert zero["confidence_interval"][0] == 0 and zero["confidence_interval"][1] > 0
    assert abs(zero["confidence_interval"][1]-(1-.025**(1/213))) < 1e-14
    nonzero = summarize_frames([0,10,0,20],[100]*4,full_frame_capacity=100)
    assert nonzero["ber_estimate"] == .075
    assert nonzero["confidence_interval"][0] <= .075 <= nonzero["confidence_interval"][1]
    rng = np.random.default_rng(20261104)
    taps = (rng.standard_normal((40,3))+1j*rng.standard_normal((40,3)))*.04
    taps[:,0] += 1
    records = []
    for scheme in ("OFDM","AFDM","OTFS","DD-a-OFDM"):
        mask = np.zeros((8,4),dtype=bool); mask[[0,1,2,4,5,6],:] = True
        if scheme in ("OFDM","DD-a-OFDM"):
            mask[:,1::2] = True
        kwargs = {"c1":5/16,"c2":np.sqrt(2)/64} if scheme == "AFDM" else {}
        channel = NativeChannelOperator(scheme,taps,block_length=8,block_count=4,cp_length=2,
                                        delays=(0,1,2),**kwargs)
        model = SmallMaskedModel(channel,mask)
        received = (rng.standard_normal((8,4))+1j*rng.standard_normal((8,4)))/np.sqrt(2)
        for variance in (.25,.0025,0.):
            problem = PlugInMMSEProblem(model,received,estimated_noise_variance=variance,data_symbol_energy=1.)
            reference = solve_mmse_problem(problem,s)
            optimized = solve_mmse_problem(problem,s,include_effective_gain=True)
            a,b = reference.require_converged(),optimized.require_converged()
            relative = float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-300))
            assert relative < 1e-7
            gain = optimized.effective_gain
            if gain is None:
                gain = effective_symbol_gain(problem)
            independent_gain = effective_symbol_gain(problem)
            assert np.max(abs(gain-independent_gain)) < 1e-11
            assert np.array_equal(hard_demodulate(a,independent_gain),hard_demodulate(b,gain))
            records.append({"scheme":scheme,"variance":variance,"solution_relative_difference":relative,
                            "gain_max_difference":float(np.max(abs(gain-independent_gain))),
                            "backend":optimized.diagnostics.get("backend",optimized.method)})
    report={"exact_full_bit_quota":10485760,"baseline_frame_count":160,"pilot_frame_count":214,
            "pilot_tail_bits":16384,"zero_error_confidence":zero,
            "accelerated_reference_controls":records,
            "scope":"Exact numerical rearrangements and accounting only; no physical or receiver-information changes."}
    Path('data/ber_diagnostics/pipeline_checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('Exact quotas, frame confidence rules and 12 accelerated/dense controls passed.')


if __name__ == '__main__':
    main()

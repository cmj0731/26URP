"""Estimated native channel operators for the agreed aligned per-slot-CP model.

Inputs are receiver-reconstructed taps on the CP-inclusive receive clock. This
module reads no channel-generator truth and performs no pilot cancellation,
noise estimation, MMSE solve, symbol decision or BER calculation.
"""

import numpy as np

import afdm
import otfs


class NativeChannelOperator:
    def __init__(self, scheme, estimated_taps, *, block_length, block_count,
                 cp_length, delays, c1=None, c2=None):
        if scheme not in ("OFDM","AFDM","OTFS","DD-a-OFDM"):
            raise ValueError("Unknown waveform scheme.")
        dimensions = (block_length,block_count,cp_length)
        if (any(not isinstance(v,(int,np.integer)) or isinstance(v,(bool,np.bool_)) for v in dimensions)
                or block_length <= 0 or block_count <= 0 or not 0 <= cp_length <= block_length):
            raise ValueError("Positive block dimensions and a valid integer prefix are required.")
        delays = tuple(delays)
        if (not delays or len(set(delays)) != len(delays)
                or any(not isinstance(v,(int,np.integer)) or isinstance(v,(bool,np.bool_))
                       or not 0 <= v <= cp_length or v >= block_length for v in delays)):
            raise ValueError("Distinct integer tap positions covered by the prefix are required.")
        self.scheme = scheme
        self.length,self.blocks,self.cp = int(block_length),int(block_count),int(cp_length)
        self.delays = delays
        self.grid_shape = (self.length,self.blocks)
        self.shape = (self.length*self.blocks,)*2
        values = np.asarray(estimated_taps,dtype=np.complex128)
        if (values.shape != ((self.length+self.cp)*self.blocks,len(delays))
                or not np.isfinite(values).all()):
            raise ValueError("Finite estimated taps on every CP-inclusive receive sample are required.")
        indices = (np.arange(self.blocks)[:,None]*(self.length+self.cp)+self.cp
                   + np.arange(self.length)[None,:]).reshape(-1)
        self.taps = values[indices].reshape((*self.grid_shape,len(delays)),order="F").copy()
        self.taps.setflags(write=False)
        self.chirp1 = self.chirp2 = None
        if scheme == "AFDM":
            if c1 is None or c2 is None or not np.isfinite([c1,c2]).all():
                raise ValueError("The selected finite AFDM chirp parameters are required.")
            afdm._ordinary_cp_condition(self.length,c1)
            self.chirp1,self.chirp2 = afdm._chirps(self.length,c1,c2)

    def _grid(self, values):
        grid = np.asarray(values,dtype=np.complex128)
        if grid.shape != self.grid_shape or not np.isfinite(grid).all():
            raise ValueError("A finite native grid with the selected frame dimensions is required.")
        return grid

    def _to_time(self, grid):
        if self.scheme == "OTFS":
            # Exact inverse-Zak identity: the frequency FFT/IFFT cancel.
            return np.fft.ifft(grid,axis=1,norm="ortho")
        if self.scheme == "AFDM":
            return self.chirp1[:,None]*np.fft.ifft(self.chirp2[:,None]*grid,axis=0,norm="ortho")
        return np.fft.ifft(grid,axis=0,norm="ortho")

    def _from_time(self, grid):
        if self.scheme == "OTFS":
            return np.fft.fft(grid,axis=1,norm="ortho")
        if self.scheme == "AFDM":
            return self.chirp2.conj()[:,None]*np.fft.fft(self.chirp1.conj()[:,None]*grid,axis=0,norm="ortho")
        return np.fft.fft(grid,axis=0,norm="ortho")

    def apply(self, native_grid):
        """Apply the full native channel, including every modeled coupling."""
        useful = self._to_time(self._grid(native_grid))
        result = np.zeros(self.grid_shape,dtype=complex)
        for column,delay in enumerate(self.delays):
            result += self.taps[:,:,column]*np.roll(useful,delay,axis=0)
        return self._from_time(result)

    def adjoint(self, native_grid):
        """Apply the conjugate transpose, without an inverse or regularizer."""
        useful = self._to_time(self._grid(native_grid))
        result = np.zeros(self.grid_shape,dtype=complex)
        for column,delay in enumerate(self.delays):
            # Move both the received sample and its receive-time tap weight.
            result += np.roll(self.taps[:,:,column].conj()*useful,-delay,axis=0)
        return self._from_time(result)

    def block_matrix(self, block):
        """Return one full TF/DAFT matrix; OTFS uses the full-frame operator."""
        if self.scheme == "OTFS":
            raise ValueError("A single slot matrix does not represent the selected full DD channel.")
        if (not isinstance(block,(int,np.integer)) or isinstance(block,(bool,np.bool_))
                or not 0 <= block < self.blocks):
            raise ValueError("A valid block index is required.")
        q = np.arange(self.length)
        matrix = np.zeros((self.length,self.length),dtype=complex)
        for column,delay in enumerate(self.delays):
            matrix[q,(q-delay) % self.length] += self.taps[:,block,column]
        if self.scheme == "AFDM":
            matrix = self.chirp1.conj()[:,None]*matrix*self.chirp1[None,:]
        matrix = np.fft.ifft(np.fft.fft(matrix,axis=0,norm="ortho"),axis=1,norm="ortho")
        if self.scheme == "AFDM":
            matrix = self.chirp2.conj()[:,None]*matrix*self.chirp2[None,:]
        return matrix


def build_channel_operator(scheme, settings, estimated_taps):
    """Build from public design plus a receiver's reconstructed estimated taps.

    Typical input is model.reconstruct_taps(estimate), including OFDM BEM
    coefficients or the other receivers' path estimates. Keep all six candidate
    tap columns; do not prune from true active delays or renormalize the taps.
    """
    prefix = settings["prefix_policy"]
    information = settings["receiver_information_policy"]["delay_support"]
    length,cp = prefix["useful_samples_per_block"],prefix["prefix_samples_per_block"]
    blocks,remainder = divmod(prefix["transmitted_samples_per_frame"],length+cp)
    if remainder:
        raise ValueError("Complete prefixed blocks are required.")
    delays = tuple(range(information["minimum_excess_delay_samples"],
                         information["maximum_excess_delay_samples"]+1))
    kwargs = {}
    if scheme == "AFDM":
        spec = settings["waveform_design"]["AFDM"]
        if (spec["daft_length"],spec["blocks_per_frame"]) != (length,blocks):
            raise ValueError("The AFDM waveform and channel dimensions disagree.")
        kwargs = {key:spec["chirp_parameters"][key] for key in ("c1","c2")}
    elif scheme == "OTFS":
        spec = settings["waveform_design"]["OTFS"]
        if (spec["delay_bin_count"],spec["doppler_bin_count"]) != (length,blocks):
            raise ValueError("The OTFS waveform and channel dimensions disagree.")
    elif scheme in ("OFDM","DD-a-OFDM"):
        layout = settings["pilot_comparison_policy"]["shared_ofdm_dd_a_tf_layout"]
        if tuple(layout["grid_shape_frequency_by_time"]) != (length,blocks):
            raise ValueError("The TF waveform and channel dimensions disagree.")
    return NativeChannelOperator(scheme,estimated_taps,block_length=length,block_count=blocks,
                                 cp_length=cp,delays=delays,**kwargs)

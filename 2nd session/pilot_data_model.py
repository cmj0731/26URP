"""Known-pilot subtraction and data-column selection for native MMSE inputs.

Keep every native received cell. Only transmitted data cells are unknown;
pilots are known and guards/reservations are zero. Subtraction uses the supplied
estimated channel, never generator truth. No covariance, noise policy, MMSE
solver, decision feedback, symbol decisions or BER calculation is selected here.
"""

import numpy as np

from pilot_layouts import build_grid, build_masks


class PilotDataModel:
    def __init__(self, channel_operator, settings):
        self.channel = channel_operator
        self.scheme = channel_operator.scheme
        self.grid_shape = channel_operator.grid_shape
        masks = build_masks(self.scheme,scenario=settings)
        if masks["data"].shape != self.grid_shape:
            raise ValueError("The native pilot/data mask and channel dimensions disagree.")
        self.data_indices = np.flatnonzero(masks["data"].reshape(-1,order="F"))
        self.data_count = self.data_indices.size
        if not self.data_count:
            raise ValueError("At least one unknown data cell is required.")
        self.data_indices.setflags(write=False)
        self.shape = (int(np.prod(self.grid_shape)),self.data_count)
        self.data_mask = masks["data"].copy()
        self.data_mask.setflags(write=False)
        self._column_offsets = np.r_[0,np.cumsum(np.sum(self.data_mask,axis=0))]
        self.pilot_grid = build_grid(self.scheme,np.zeros(self.data_count,dtype=complex),scenario=settings)
        self.pilot_grid.setflags(write=False)
        self.pilot_response = self.channel.apply(self.pilot_grid)
        self.pilot_response.setflags(write=False)

    def _observations(self, values):
        grid = np.asarray(values,dtype=np.complex128)
        if grid.shape != self.grid_shape or not np.isfinite(grid).all():
            raise ValueError("A finite full native receive grid is required.")
        return grid

    def embed_data(self, data_symbols):
        """Embed unknowns in their original cells; all non-data cells are zero."""
        values = np.asarray(data_symbols,dtype=np.complex128)
        if values.shape != (self.data_count,) or not np.isfinite(values).all():
            raise ValueError("One finite vector in the agreed data-cell order is required.")
        flat = np.zeros(self.shape[0],dtype=complex)
        flat[self.data_indices] = values
        return flat.reshape(self.grid_shape,order="F")

    def apply(self, data_symbols):
        """Apply A_hat = H_hat P_data, without adding the known pilots."""
        return self.channel.apply(self.embed_data(data_symbols))

    def adjoint(self, received_grid):
        """Apply P_data^H H_hat^H using all native receive observations."""
        native = self.channel.adjoint(self._observations(received_grid))
        return native.reshape(-1,order="F")[self.data_indices]

    def subtract_pilots(self, received_grid):
        """Return r-H_hat*p; preserve the input and every receive coordinate."""
        return self._observations(received_grid)-self.pilot_response

    def block_data_slice(self, block):
        """Locate one TF/DAFT block's data in the frame data vector."""
        if self.scheme == "OTFS":
            raise ValueError("The selected OTFS data problem couples the whole DD frame.")
        if (not isinstance(block,(int,np.integer)) or isinstance(block,(bool,np.bool_))
                or not 0 <= block < self.grid_shape[1]):
            raise ValueError("A valid block index is required.")
        return slice(int(self._column_offsets[block]),int(self._column_offsets[block+1]))

    def block_matrix(self, block):
        """Keep all block receive rows and only unknown-data transmit columns."""
        self.block_data_slice(block)
        return self.channel.block_matrix(block)[:,self.data_mask[:,block]]


def build_pilot_data_model(channel_operator, settings):
    return PilotDataModel(channel_operator,settings)

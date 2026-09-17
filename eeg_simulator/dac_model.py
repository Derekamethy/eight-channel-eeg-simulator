"""Virtual DAC conversion model; no physical analogue transfer is implied."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class VirtualDAC:
    """Map requested EEG microvolts to unsigned virtual DAC codes.

    In real hardware, the code-to-voltage relationship would depend on the
    selected DAC, voltage reference, output topology and precision
    attenuation/analogue stage.
    """

    resolution_bits: int = 16
    full_scale_uv: float = 200.0

    def __post_init__(self) -> None:
        if not 2 <= self.resolution_bits <= 32:
            raise ValueError("resolution_bits must be between 2 and 32")
        if self.full_scale_uv <= 0:
            raise ValueError("full_scale_uv must be positive")

    @property
    def max_code(self) -> int:
        return (1 << self.resolution_bits) - 1

    @property
    def midscale_code(self) -> int:
        return 1 << (self.resolution_bits - 1)

    def microvolts_to_code(self, value_uv: ArrayLike) -> NDArray[np.uint32] | int:
        values = np.asarray(value_uv, dtype=np.float64)
        clipped = np.clip(values, -self.full_scale_uv, self.full_scale_uv)
        normalised = clipped / self.full_scale_uv
        codes = np.rint((normalised + 1.0) * self.max_code / 2.0)
        result = np.clip(codes, 0, self.max_code).astype(np.uint32)
        if result.ndim == 0:
            return int(result)
        return result

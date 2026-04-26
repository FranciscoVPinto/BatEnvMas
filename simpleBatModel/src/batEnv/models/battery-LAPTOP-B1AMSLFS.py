from __future__ import annotations

"""
Compatibility shim.

The project now has a single canonical optimization model:
`batEnv.models.MultiHouseModel`.

For one-house simulations, use `MultiHouseModel` with a single house id.
This module keeps the old `SimpleBatteryModel` import alive by adapting the
legacy constructor and `make_instance(...)` signature to the unified model.
"""

from dataclasses import dataclass
from typing import Sequence

from .multi_house import MultiHouseModel


@dataclass
class SimpleBatteryModel:
    dt: float
    E_init: float
    E_min: float
    E_max: float
    P_ch_max: float
    P_dis_max: float
    eta_ch: float
    eta_dis: float
    P_grid_max: float
    allow_export: bool = True

    _HOUSE_ID = "_single"

    def make_instance(
        self,
        load: Sequence[float],
        pv: Sequence[float],
        c_grid: Sequence[float],
        c_sell: Sequence[float],
    ):
        unified = MultiHouseModel(dt=self.dt, allow_export=self.allow_export)
        return unified.make_instance(
            houses=[self._HOUSE_ID],
            loads_by_house={self._HOUSE_ID: list(load)},
            pv_by_house={self._HOUSE_ID: list(pv)},
            bat_params_by_house={
                self._HOUSE_ID: {
                    "E_init": self.E_init,
                    "E_min": self.E_min,
                    "E_max": self.E_max,
                    "P_ch_max": self.P_ch_max,
                    "P_dis_max": self.P_dis_max,
                    "eta_ch": self.eta_ch,
                    "eta_dis": self.eta_dis,
                    "P_grid_max": self.P_grid_max,
                }
            },
            c_grid={self._HOUSE_ID: list(c_grid)},
            c_sell={self._HOUSE_ID: list(c_sell)},
        )

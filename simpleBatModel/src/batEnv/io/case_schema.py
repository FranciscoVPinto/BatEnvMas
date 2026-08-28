"""
JSON schema for case YAMLs.

Kept lenient (additionalProperties=True) so that:
  - debug/meta keys injected by other tooling don't trigger errors;
  - new optional knobs can be added without immediately editing the schema.

The schema catches the common typos and shape mistakes:
  - missing 'case' / 'data' / 'houses'
  - 'data' without 'loads' or both 'pv' and 'pv_total' set
  - wrong types (e.g. tariffs as a string when a dict was expected)
  - houses without 'battery'
  - sharing.alpha as a list instead of a dict
"""
from __future__ import annotations


_BATTERY_SCHEMA = {
    "type": "object",
    "properties": {
        "E_init": {"type": "number"},
        "E_min": {"type": "number"},
        "E_max": {"type": "number"},
        "P_ch_max": {"type": "number"},
        "P_dis_max": {"type": "number"},
        "eta_ch": {"type": "number", "minimum": 0, "maximum": 1},
        "eta_dis": {"type": "number", "minimum": 0, "maximum": 1},
        # P_grid_max is the legacy field; P_contracted is the canonical name
        # (Potência Contratada, DL 15/2022). run_case.py accepts both.
        "P_grid_max": {"type": "number"},
        "P_contracted": {"type": "number"},
        # Optional physical parameters for automatic degradation cost (lambda) calculation
        # via compute_degradation_cost_per_kwh() in battery_economics.py (Wohler model).
        "battery_cost_eur": {"type": "number", "minimum": 0},
        "N_rated_cycles": {"type": "number", "minimum": 1},
        "DoD_rated": {"type": "number", "minimum": 0, "maximum": 1},
        "aging_exponent": {"type": "number", "minimum": 0},
    },
    "additionalProperties": False,
}

_PER_HOUSE_SCHEMA = {
    "type": "object",
    "properties": {
        "battery": _BATTERY_SCHEMA,
        "tariffs": {"type": "object"},
    },
    "additionalProperties": True,
}

_TARIFF_SPEC_SCHEMA = {
    "anyOf": [
        {"type": "number"},
        {"type": "array", "items": {"type": "number"}},
        {"type": "object"},
    ],
}

_FALLBACK_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": {"enum": ["none", "consumption_instant", "consumption_mean"]},
        "apply_when": {"enum": ["invalid_or_missing", "always"]},
        "zero_denom": {"enum": ["equal_split", "keep_previous"]},
    },
    "additionalProperties": False,
}

_BATTERY_DEGRADATION_PWL_SCHEMA = {
    "type": "object",
    "properties": {
        "soc_breakpoints": {
            "type": "array",
            "items": {"type": "number", "minimum": 0, "maximum": 1},
            "minItems": 2,
        },
        "lambda_by_bin": {
            "type": "array",
            "items": {"type": "number", "minimum": 0},
            "minItems": 1,
        },
        "lambda_by_bin_per_house": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "number", "minimum": 0},
                "minItems": 1,
            },
        },
    },
    "required": ["soc_breakpoints", "lambda_by_bin"],
    "additionalProperties": False,
}

_ROLLING_HORIZON_SCHEMA = {
    "type": "object",
    "properties": {
        # Opt-in receding-horizon solve (approximate) for long horizons.
        "enabled": {"type": "boolean"},
        # window = total timesteps per window (commit + look-ahead).
        "window": {"type": "integer", "minimum": 2},
        # step = committed timesteps per window (1 <= step <= window).
        "step": {"type": "integer", "minimum": 1},
    },
    "additionalProperties": False,
}

CASE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "BatEnv case YAML",
    "type": "object",
    "required": ["case", "data", "houses"],
    "properties": {
        "case": {"type": "string", "minLength": 1},
        "extends": {
            "anyOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
            ],
        },
        "time": {
            "type": "object",
            "properties": {
                "horizon": {"type": ["integer", "null"], "minimum": 1},
                "dt_hours": {"type": "number", "exclusiveMinimum": 0},
                "start": {"type": ["string", "null"]},
            },
            "additionalProperties": False,
        },
        "data": {
            "type": "object",
            "required": ["loads"],
            "properties": {
                "loads": {
                    "type": "object",
                    "minProperties": 1,
                    "additionalProperties": {"type": "string"},
                },
                "pv": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "object", "additionalProperties": {"type": "string"}},
                    ],
                },
                "pv_total": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    ],
                },
            },
            "oneOf": [
                {"required": ["pv"], "not": {"required": ["pv_total"]}},
                {"required": ["pv_total"], "not": {"required": ["pv"]}},
            ],
            "additionalProperties": True,
        },
        "tariffs": {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "grid_buy": _TARIFF_SPEC_SCHEMA,
                "grid_sell": _TARIFF_SPEC_SCHEMA,
            },
            "additionalProperties": True,
        },
        "grid": {
            "type": "object",
            "properties": {"allow_export": {"type": "boolean"}},
            "additionalProperties": False,
        },
        "sharing": {
            "type": "object",
            "properties": {
                "mode": {"enum": ["fixed_alpha", "optimal"]},
                "normalize": {"type": "boolean"},
                "strict_sum_to_one": {"type": "boolean"},
                "alpha": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                },
                "alpha_profile": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "fallback": _FALLBACK_SCHEMA,
            },
            "additionalProperties": True,
        },
        "model": {
            "type": "object",
            "properties": {
                "cyclic_soc": {"type": "boolean"},
                "battery_degradation_eur_per_kwh": {"type": "number", "minimum": 0},
                "battery_degradation_pwl": _BATTERY_DEGRADATION_PWL_SCHEMA,
                "rolling_horizon": _ROLLING_HORIZON_SCHEMA,
            },
            "additionalProperties": False,
        },
        "houses": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": _PER_HOUSE_SCHEMA,
        },
        "debug": {"type": "object"},
    },
    "additionalProperties": True,
}

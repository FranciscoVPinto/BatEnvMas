from __future__ import annotations


# Shared sub-schemas
_SOLVER_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "options": {"type": "object"},
    },
    "additionalProperties": False,
}

_TIME_DEFAULTS_SCHEMA = {
    "type": "object",
    "properties": {
        "horizon": {"type": ["integer", "null"], "minimum": 1},
        "dt_hours": {"type": "number", "exclusiveMinimum": 0},
        "start": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}

_PLOTS_SCHEMA = {
    "type": "object",
    "properties": {
        "per_case": {"type": "boolean"},
        "comparisons": {"type": "boolean"},
        "include_community_per_case": {"type": "boolean"},
    },
    "additionalProperties": False,
}

_DEFAULTS_RUN_SCHEMA = {
    "type": "object",
    "properties": {
        "outputs_dir": {"type": "string"},
        "tee": {"type": "boolean"},
        "solver": _SOLVER_SCHEMA,
        "time": _TIME_DEFAULTS_SCHEMA,
        "cases_base_dir": {"type": "string"},
        "plots": _PLOTS_SCHEMA,  # allowed in single-experiment YAMLs that drive plotting too
    },
    "additionalProperties": False,
}

_DEFAULTS_PLOT_SCHEMA = {
    "type": "object",
    "properties": {
        "outputs_dir": {"type": "string"},
        "cases_base_dir": {"type": "string"},
        "plots": _PLOTS_SCHEMA,
    },
    "additionalProperties": False,
}

_ENABLED_BOOL_MAP = {
    "type": "object",
    "additionalProperties": {"type": "boolean"},
}

_CASES_LIST_SCHEMA = {"type": "array", "items": {"type": "string"}}

_CASES_GLOB_SCHEMA = {
    "anyOf": [
        {"type": "string"},
        {"type": "array", "items": {"type": "string"}, "minItems": 1},
    ],
}

_SWEEP_ENTRY_SCHEMA = {
    "type": "object",
    "required": ["suffix"],
    "properties": {
        "suffix": {"type": "string", "minLength": 1},
        "overrides": {"type": "object"},
        "time_override": {
            "type": "object",
            "properties": {
                "horizon": {"type": ["integer", "null"], "minimum": 1},
                "dt_hours": {"type": "number", "exclusiveMinimum": 0},
                "start": {"type": ["string", "null"]},
            },
            "additionalProperties": False,
        },
        "outputs_subdir": {"type": "string", "minLength": 1},
    },
    "additionalProperties": False,
}
_SWEEP_SCHEMA = {
    "type": "array",
    "items": _SWEEP_ENTRY_SCHEMA,
    "minItems": 1,
}


# ----- runset shapes -----

_PARENT_RUNSET = {
    "type": "object",
    "required": ["runset"],
    "properties": {
        "runset": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "enabled": _ENABLED_BOOL_MAP,
        "defaults": _DEFAULTS_RUN_SCHEMA,
        "_runset_path": {"type": "string"},  # injected by loader
    },
    "additionalProperties": False,
}

_SINGLE_RUNSET = {
    "type": "object",
    "required": ["runset"],
    "properties": {
        "runset": {"type": "string"},
        "cases_base_dir": {"type": "string"},
        "cases": _CASES_LIST_SCHEMA,
        "cases_glob": _CASES_GLOB_SCHEMA,
        "sweep": _SWEEP_SCHEMA,
        "enabled": _ENABLED_BOOL_MAP,
        "defaults": _DEFAULTS_RUN_SCHEMA,
        "_runset_path": {"type": "string"},
    },
    # at least one of cases / cases_glob must be present
    "anyOf": [
        {"required": ["cases"]},
        {"required": ["cases_glob"]},
    ],
    "additionalProperties": False,
}

_SINGLE_EXPERIMENT = {
    "type": "object",
    "required": ["experiment", "case_yaml"],
    "properties": {
        "experiment": {"type": "string"},
        "case_yaml": {"type": "string"},
        "enabled": {"type": "boolean"},
        "defaults": _DEFAULTS_RUN_SCHEMA,
        "_runset_path": {"type": "string"},
    },
    "additionalProperties": False,
}

RUNSET_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "BatEnv runset / experiment YAML",
    "oneOf": [_PARENT_RUNSET, _SINGLE_RUNSET, _SINGLE_EXPERIMENT],
}


# ----- plotset shapes -----

_PARENT_PLOTSET = {
    "type": "object",
    "required": ["plotset"],
    "properties": {
        "plotset": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "enabled": _ENABLED_BOOL_MAP,
        "defaults": _DEFAULTS_PLOT_SCHEMA,
        "_plotset_path": {"type": "string"},
    },
    "additionalProperties": False,
}

_SINGLE_PLOTSET = {
    "type": "object",
    "required": ["plotset"],
    "properties": {
        "plotset": {"type": "string"},
        "cases_base_dir": {"type": "string"},
        "outputs_dir": {"type": "string"},
        "cases": _CASES_LIST_SCHEMA,
        "cases_glob": _CASES_GLOB_SCHEMA,
        "sweep": _SWEEP_SCHEMA,
        "enabled": _ENABLED_BOOL_MAP,
        "plots": _PLOTS_SCHEMA,
        "defaults": _DEFAULTS_PLOT_SCHEMA,
        "_plotset_path": {"type": "string"},
    },
    "anyOf": [
        {"required": ["cases"]},
        {"required": ["cases_glob"]},
    ],
    "additionalProperties": False,
}

# Plotting from a single-experiment YAML reuses the same shape as runset's.
_SINGLE_EXPERIMENT_PLOT = {
    "type": "object",
    "required": ["experiment", "case_yaml"],
    "properties": {
        "experiment": {"type": "string"},
        "case_yaml": {"type": "string"},
        "enabled": {"type": "boolean"},
        "defaults": _DEFAULTS_RUN_SCHEMA,
        "_plotset_path": {"type": "string"},
    },
    "additionalProperties": False,
}

PLOTSET_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "BatEnv plotset / experiment YAML",
    "oneOf": [_PARENT_PLOTSET, _SINGLE_PLOTSET, _SINGLE_EXPERIMENT_PLOT],
}

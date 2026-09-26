from __future__ import annotations

from typing import Any, Dict, Iterable


UNIT_ALIASES: Dict[str, Dict[str, str]] = {
    "c": {"canonical_unit": "Celsius", "ucum_code": "Cel"},
    "celsius": {"canonical_unit": "Celsius", "ucum_code": "Cel"},
    "fahrenheit": {"canonical_unit": "Fahrenheit", "ucum_code": "[degF]"},
    "f": {"canonical_unit": "Fahrenheit", "ucum_code": "[degF]"},
    "kelvin": {"canonical_unit": "Kelvin", "ucum_code": "K"},
    "k": {"canonical_unit": "Kelvin", "ucum_code": "K"},
    "millimeter": {"canonical_unit": "millimeter", "ucum_code": "mm"},
    "mm": {"canonical_unit": "millimeter", "ucum_code": "mm"},
    "meter": {"canonical_unit": "meter", "ucum_code": "m"},
    "m": {"canonical_unit": "meter", "ucum_code": "m"},
    "kilogram": {"canonical_unit": "kilogram", "ucum_code": "kg"},
    "kg": {"canonical_unit": "kilogram", "ucum_code": "kg"},
    "degree": {"canonical_unit": "degree", "ucum_code": "deg"},
    "deg": {"canonical_unit": "degree", "ucum_code": "deg"},
    "percent": {"canonical_unit": "percent", "ucum_code": "%"},
    "%": {"canonical_unit": "percent", "ucum_code": "%"},
    "kilowatt": {"canonical_unit": "kilowatt", "ucum_code": "kW"},
    "kw": {"canonical_unit": "kilowatt", "ucum_code": "kW"},
    "volt": {"canonical_unit": "volt", "ucum_code": "V"},
    "v": {"canonical_unit": "volt", "ucum_code": "V"},
    "ntu": {"canonical_unit": "NTU", "ucum_code": "{NTU}"},
    "psu": {"canonical_unit": "PSU", "ucum_code": "{PSU}"},
    "milligram_per_liter": {"canonical_unit": "milligram_per_liter", "ucum_code": "mg/L"},
    "mg/l": {"canonical_unit": "milligram_per_liter", "ucum_code": "mg/L"},
    "mg_l": {"canonical_unit": "milligram_per_liter", "ucum_code": "mg/L"},
    "meter_per_second": {"canonical_unit": "meter_per_second", "ucum_code": "m/s"},
    "m/s": {"canonical_unit": "meter_per_second", "ucum_code": "m/s"},
}


def _unit_key(unit: str) -> str:
    return unit.strip().lower().replace(" ", "_")


def _evidence_basis(evidence_types: Iterable[str]) -> str:
    evidence_type_set = set(evidence_types)
    if {"hdf5_attribute", "netcdf_attribute", "zarr_attribute"} & evidence_type_set:
        return "explicit_metadata"
    if "column_name_unit_hint" in evidence_type_set:
        return "name_pattern"
    return "not_evidence_backed"


def normalize_unit_claim(unit: str | None, evidence_types: Iterable[str]) -> Dict[str, Any]:
    if unit is None:
        return {
            "status": "no_unit_claim",
            "original_unit": None,
            "canonical_unit": None,
            "ucum_code": None,
            "evidence_basis": "not_applicable",
        }

    evidence_basis = _evidence_basis(evidence_types)
    normalized = UNIT_ALIASES.get(_unit_key(unit))
    if normalized is None:
        return {
            "status": "unmapped_unit",
            "original_unit": unit,
            "canonical_unit": unit,
            "ucum_code": None,
            "evidence_basis": evidence_basis,
        }

    return {
        "status": "normalized_ucum",
        "original_unit": unit,
        "canonical_unit": normalized["canonical_unit"],
        "ucum_code": normalized["ucum_code"],
        "evidence_basis": evidence_basis,
    }

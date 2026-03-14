"""Lookup / reference tables for the insurance-claims data generator.

Every list and mapping lives here so the generator modules stay thin.
"""

from __future__ import annotations

from typing import Dict, List

# ── Medical Specialties ──────────────────────────────────────────────────
# Each dict carries a human label and a risk_tier used downstream for
# severity skewing and loss-amount multipliers.

SPECIALTIES: List[Dict] = [
    {"specialty_id": "SP01", "name": "General Surgery",          "risk_tier": "high"},
    {"specialty_id": "SP02", "name": "Orthopedic Surgery",       "risk_tier": "high"},
    {"specialty_id": "SP03", "name": "Neurosurgery",             "risk_tier": "high"},
    {"specialty_id": "SP04", "name": "Obstetrics/Gynecology",    "risk_tier": "high"},
    {"specialty_id": "SP05", "name": "Anesthesiology",           "risk_tier": "high"},
    {"specialty_id": "SP06", "name": "Emergency Medicine",       "risk_tier": "medium"},
    {"specialty_id": "SP07", "name": "Cardiology",               "risk_tier": "medium"},
    {"specialty_id": "SP08", "name": "Radiology",                "risk_tier": "medium"},
    {"specialty_id": "SP09", "name": "Gastroenterology",         "risk_tier": "medium"},
    {"specialty_id": "SP10", "name": "Oncology",                 "risk_tier": "medium"},
    {"specialty_id": "SP11", "name": "Internal Medicine",        "risk_tier": "low"},
    {"specialty_id": "SP12", "name": "Family Medicine",          "risk_tier": "low"},
    {"specialty_id": "SP13", "name": "Pediatrics",               "risk_tier": "low"},
    {"specialty_id": "SP14", "name": "Psychiatry",               "risk_tier": "low"},
    {"specialty_id": "SP15", "name": "Dermatology",              "risk_tier": "low"},
]

# ── US States ────────────────────────────────────────────────────────────
# region + urban_rural classification (majority classification for the state).

STATES: List[Dict] = [
    # Northeast
    {"state_code": "CT", "name": "Connecticut",     "region": "Northeast", "urban_rural": "urban"},
    {"state_code": "DE", "name": "Delaware",         "region": "Northeast", "urban_rural": "suburban"},
    {"state_code": "MA", "name": "Massachusetts",    "region": "Northeast", "urban_rural": "urban"},
    {"state_code": "MD", "name": "Maryland",         "region": "Northeast", "urban_rural": "urban"},
    {"state_code": "ME", "name": "Maine",            "region": "Northeast", "urban_rural": "rural"},
    {"state_code": "NH", "name": "New Hampshire",    "region": "Northeast", "urban_rural": "rural"},
    {"state_code": "NJ", "name": "New Jersey",       "region": "Northeast", "urban_rural": "urban"},
    {"state_code": "NY", "name": "New York",         "region": "Northeast", "urban_rural": "urban"},
    {"state_code": "PA", "name": "Pennsylvania",     "region": "Northeast", "urban_rural": "suburban"},
    {"state_code": "RI", "name": "Rhode Island",     "region": "Northeast", "urban_rural": "urban"},
    {"state_code": "VT", "name": "Vermont",          "region": "Northeast", "urban_rural": "rural"},
    # Southeast
    {"state_code": "AL", "name": "Alabama",          "region": "Southeast", "urban_rural": "rural"},
    {"state_code": "FL", "name": "Florida",          "region": "Southeast", "urban_rural": "urban"},
    {"state_code": "GA", "name": "Georgia",          "region": "Southeast", "urban_rural": "suburban"},
    {"state_code": "KY", "name": "Kentucky",         "region": "Southeast", "urban_rural": "rural"},
    {"state_code": "LA", "name": "Louisiana",        "region": "Southeast", "urban_rural": "suburban"},
    {"state_code": "MS", "name": "Mississippi",      "region": "Southeast", "urban_rural": "rural"},
    {"state_code": "NC", "name": "North Carolina",   "region": "Southeast", "urban_rural": "suburban"},
    {"state_code": "SC", "name": "South Carolina",   "region": "Southeast", "urban_rural": "suburban"},
    {"state_code": "TN", "name": "Tennessee",        "region": "Southeast", "urban_rural": "suburban"},
    {"state_code": "VA", "name": "Virginia",         "region": "Southeast", "urban_rural": "suburban"},
    {"state_code": "WV", "name": "West Virginia",    "region": "Southeast", "urban_rural": "rural"},
    # Midwest
    {"state_code": "IA", "name": "Iowa",             "region": "Midwest", "urban_rural": "rural"},
    {"state_code": "IL", "name": "Illinois",         "region": "Midwest", "urban_rural": "urban"},
    {"state_code": "IN", "name": "Indiana",          "region": "Midwest", "urban_rural": "suburban"},
    {"state_code": "KS", "name": "Kansas",           "region": "Midwest", "urban_rural": "rural"},
    {"state_code": "MI", "name": "Michigan",         "region": "Midwest", "urban_rural": "suburban"},
    {"state_code": "MN", "name": "Minnesota",        "region": "Midwest", "urban_rural": "suburban"},
    {"state_code": "MO", "name": "Missouri",         "region": "Midwest", "urban_rural": "suburban"},
    {"state_code": "ND", "name": "North Dakota",     "region": "Midwest", "urban_rural": "rural"},
    {"state_code": "NE", "name": "Nebraska",         "region": "Midwest", "urban_rural": "rural"},
    {"state_code": "OH", "name": "Ohio",             "region": "Midwest", "urban_rural": "suburban"},
    {"state_code": "SD", "name": "South Dakota",     "region": "Midwest", "urban_rural": "rural"},
    {"state_code": "WI", "name": "Wisconsin",        "region": "Midwest", "urban_rural": "suburban"},
    # West
    {"state_code": "AK", "name": "Alaska",           "region": "West", "urban_rural": "rural"},
    {"state_code": "CA", "name": "California",       "region": "West", "urban_rural": "urban"},
    {"state_code": "CO", "name": "Colorado",         "region": "West", "urban_rural": "suburban"},
    {"state_code": "HI", "name": "Hawaii",           "region": "West", "urban_rural": "suburban"},
    {"state_code": "ID", "name": "Idaho",            "region": "West", "urban_rural": "rural"},
    {"state_code": "MT", "name": "Montana",          "region": "West", "urban_rural": "rural"},
    {"state_code": "NV", "name": "Nevada",           "region": "West", "urban_rural": "urban"},
    {"state_code": "OR", "name": "Oregon",           "region": "West", "urban_rural": "suburban"},
    {"state_code": "UT", "name": "Utah",             "region": "West", "urban_rural": "suburban"},
    {"state_code": "WA", "name": "Washington",       "region": "West", "urban_rural": "suburban"},
    {"state_code": "WY", "name": "Wyoming",          "region": "West", "urban_rural": "rural"},
    # Southwest
    {"state_code": "AZ", "name": "Arizona",          "region": "Southwest", "urban_rural": "urban"},
    {"state_code": "AR", "name": "Arkansas",         "region": "Southwest", "urban_rural": "rural"},
    {"state_code": "NM", "name": "New Mexico",       "region": "Southwest", "urban_rural": "rural"},
    {"state_code": "OK", "name": "Oklahoma",         "region": "Southwest", "urban_rural": "suburban"},
    {"state_code": "TX", "name": "Texas",            "region": "Southwest", "urban_rural": "urban"},
]

# ── Diagnoses ────────────────────────────────────────────────────────────
# ICD-style grouping codes with severity weights that feed into loss models.

DIAGNOSES: List[Dict] = [
    {"diagnosis_id": "DX01", "code": "T81.4",  "category": "Surgical Site Infection",       "severity_weight": 1.0},
    {"diagnosis_id": "DX02", "code": "O75.4",  "category": "Birth Injury",                  "severity_weight": 1.8},
    {"diagnosis_id": "DX03", "code": "T80.1",  "category": "Medication Error",              "severity_weight": 1.2},
    {"diagnosis_id": "DX04", "code": "Y65.5",  "category": "Misdiagnosis",                  "severity_weight": 1.5},
    {"diagnosis_id": "DX05", "code": "T81.0",  "category": "Post-Operative Hemorrhage",     "severity_weight": 1.6},
    {"diagnosis_id": "DX06", "code": "T84.0",  "category": "Implant/Device Failure",        "severity_weight": 1.4},
    {"diagnosis_id": "DX07", "code": "G97.1",  "category": "Anesthesia Complication",       "severity_weight": 1.7},
    {"diagnosis_id": "DX08", "code": "T85.6",  "category": "Catheter-Related Infection",    "severity_weight": 0.8},
    {"diagnosis_id": "DX09", "code": "T81.1",  "category": "Surgical Shock",                "severity_weight": 2.0},
    {"diagnosis_id": "DX10", "code": "Y83.1",  "category": "Wrong-Site Surgery",            "severity_weight": 1.9},
    {"diagnosis_id": "DX11", "code": "T88.7",  "category": "Adverse Drug Reaction",         "severity_weight": 1.1},
    {"diagnosis_id": "DX12", "code": "W19",    "category": "Patient Fall",                  "severity_weight": 0.7},
    {"diagnosis_id": "DX13", "code": "L89.3",  "category": "Pressure Ulcer",                "severity_weight": 0.6},
    {"diagnosis_id": "DX14", "code": "T80.2",  "category": "Transfusion Reaction",          "severity_weight": 1.3},
    {"diagnosis_id": "DX15", "code": "J95.0",  "category": "Ventilator-Associated Pneumonia", "severity_weight": 1.2},
    {"diagnosis_id": "DX16", "code": "A41.9",  "category": "Sepsis",                        "severity_weight": 1.8},
    {"diagnosis_id": "DX17", "code": "I26.9",  "category": "Pulmonary Embolism",            "severity_weight": 1.6},
    {"diagnosis_id": "DX18", "code": "R57.0",  "category": "Cardiac Arrest - Iatrogenic",   "severity_weight": 2.0},
    {"diagnosis_id": "DX19", "code": "T86.1",  "category": "Organ Rejection",               "severity_weight": 1.5},
    {"diagnosis_id": "DX20", "code": "Y69",    "category": "Delayed Treatment",             "severity_weight": 0.9},
]

# ── Root Causes ──────────────────────────────────────────────────────────

ROOT_CAUSES: List[Dict] = [
    {"root_cause_id": "RC01", "category": "Communication Failure",       "preventability": "preventable"},
    {"root_cause_id": "RC02", "category": "Documentation Error",         "preventability": "preventable"},
    {"root_cause_id": "RC03", "category": "Inadequate Training",         "preventability": "preventable"},
    {"root_cause_id": "RC04", "category": "Protocol Non-Compliance",     "preventability": "preventable"},
    {"root_cause_id": "RC05", "category": "Equipment Malfunction",       "preventability": "partially"},
    {"root_cause_id": "RC06", "category": "Staffing Shortage",           "preventability": "partially"},
    {"root_cause_id": "RC07", "category": "Patient Non-Compliance",      "preventability": "partially"},
    {"root_cause_id": "RC08", "category": "Diagnostic Error",            "preventability": "partially"},
    {"root_cause_id": "RC09", "category": "System/IT Failure",           "preventability": "partially"},
    {"root_cause_id": "RC10", "category": "Unforeseen Complication",     "preventability": "non-preventable"},
    {"root_cause_id": "RC11", "category": "Patient Anatomy Variation",   "preventability": "non-preventable"},
    {"root_cause_id": "RC12", "category": "Disease Progression",         "preventability": "non-preventable"},
]

# ── Hospital Departments ─────────────────────────────────────────────────

DEPARTMENTS: List[str] = [
    "Emergency Department",
    "Operating Room",
    "Intensive Care Unit",
    "Labor & Delivery",
    "Medical/Surgical Floor",
    "Outpatient Clinic",
    "Radiology",
    "Pharmacy",
]

# ── Claim Types ──────────────────────────────────────────────────────────

CLAIM_TYPES: List[str] = [
    "medical_malpractice",
    "general_liability",
    "workers_comp",
    "product_liability",
    "professional_liability",
]

# Probability weights mirror typical healthcare insurer book of business.
CLAIM_TYPE_WEIGHTS: List[float] = [0.45, 0.20, 0.15, 0.08, 0.12]

# ── Procedure Categories ─────────────────────────────────────────────────

PROCEDURE_CATEGORIES: List[str] = [
    "Surgical – Major",
    "Surgical – Minor",
    "Diagnostic Imaging",
    "Laboratory/Pathology",
    "Medication Administration",
    "Therapeutic Procedure",
    "Obstetric Delivery",
    "Anesthesia",
    "Emergency Intervention",
    "Rehabilitation/Physical Therapy",
]

# ── Specialty → Severity Distribution ─────────────────────────────────────
# Values are (weights for severity 1‑5).  High-risk surgical specialties
# skew toward 4-5; primary-care / low-risk specialties skew toward 1-2.

SPECIALTY_SEVERITY_MAP: Dict[str, List[float]] = {
    # High-risk – skew 4-5
    "General Surgery":        [0.05, 0.10, 0.20, 0.35, 0.30],
    "Orthopedic Surgery":     [0.05, 0.10, 0.20, 0.35, 0.30],
    "Neurosurgery":           [0.03, 0.07, 0.15, 0.35, 0.40],
    "Obstetrics/Gynecology":  [0.05, 0.10, 0.15, 0.35, 0.35],
    "Anesthesiology":         [0.05, 0.10, 0.20, 0.30, 0.35],
    # Medium-risk – roughly uniform / mild skew
    "Emergency Medicine":     [0.10, 0.15, 0.30, 0.25, 0.20],
    "Cardiology":             [0.08, 0.12, 0.25, 0.30, 0.25],
    "Radiology":              [0.15, 0.25, 0.30, 0.20, 0.10],
    "Gastroenterology":       [0.12, 0.20, 0.30, 0.25, 0.13],
    "Oncology":               [0.10, 0.15, 0.25, 0.30, 0.20],
    # Low-risk – skew 1-2
    "Internal Medicine":      [0.35, 0.30, 0.20, 0.10, 0.05],
    "Family Medicine":        [0.40, 0.30, 0.18, 0.08, 0.04],
    "Pediatrics":             [0.35, 0.30, 0.20, 0.10, 0.05],
    "Psychiatry":             [0.40, 0.28, 0.18, 0.10, 0.04],
    "Dermatology":            [0.45, 0.30, 0.15, 0.07, 0.03],
}

# ── Region Multipliers ───────────────────────────────────────────────────
# volume_multiplier  – scales the base probability a claim originates here.
# severity_multiplier – scales the severity draw for claims in this region.

REGION_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "Northeast":  {"volume_multiplier": 1.15, "severity_multiplier": 1.10},
    "Southeast":  {"volume_multiplier": 1.05, "severity_multiplier": 1.00},
    "Midwest":    {"volume_multiplier": 0.90, "severity_multiplier": 0.95},
    "West":       {"volume_multiplier": 1.10, "severity_multiplier": 1.05},
    "Southwest":  {"volume_multiplier": 0.95, "severity_multiplier": 0.98},
}

# ── High-Litigation States ───────────────────────────────────────────────
# These states have elevated litigation rates and average indemnity.

HIGH_LITIGATION_STATES: List[str] = ["NY", "CA", "FL", "TX"]

# ── Convenience Indexes ──────────────────────────────────────────────────
# Pre-built dicts for O(1) lookup by ID / code used throughout generation.

_SPECIALTY_BY_ID: Dict[str, Dict] = {s["specialty_id"]: s for s in SPECIALTIES}
_STATE_BY_CODE: Dict[str, Dict] = {s["state_code"]: s for s in STATES}
_DIAGNOSIS_BY_ID: Dict[str, Dict] = {d["diagnosis_id"]: d for d in DIAGNOSES}
_ROOT_CAUSE_BY_ID: Dict[str, Dict] = {r["root_cause_id"]: r for r in ROOT_CAUSES}


def get_specialty(specialty_id: str) -> Dict:
    """Return the specialty dict for the given ID, or raise KeyError."""
    return _SPECIALTY_BY_ID[specialty_id]


def get_state(state_code: str) -> Dict:
    """Return the state dict for the given two-letter code."""
    return _STATE_BY_CODE[state_code]


def get_diagnosis(diagnosis_id: str) -> Dict:
    """Return the diagnosis dict for the given ID."""
    return _DIAGNOSIS_BY_ID[diagnosis_id]


def get_root_cause(root_cause_id: str) -> Dict:
    """Return the root-cause dict for the given ID."""
    return _ROOT_CAUSE_BY_ID[root_cause_id]

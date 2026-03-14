"""Main synthetic-data generator for the insurance-claims dashboard.

Usage
-----
    python -m src.generation.synthetic_claims          # from project root
    # or
    from src.generation.synthetic_claims import generate_claims
    generate_claims()

Outputs five CSV files into ``data/raw/``:
    claims.csv, providers.csv, regions.csv, diagnoses.csv, root_causes.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from config import (
    DATE_END,
    DATE_START,
    N_CLAIMS,
    N_PROVIDERS,
    RANDOM_SEED,
    RAW_DIR,
    STATUS_WEIGHTS,
)
from src.generation.reference_data import (
    CLAIM_TYPES,
    CLAIM_TYPE_WEIGHTS,
    DEPARTMENTS,
    DIAGNOSES,
    PROCEDURE_CATEGORIES,
    REGION_MULTIPLIERS,
    ROOT_CAUSES,
    SPECIALTIES,
    STATES,
)
from src.generation.distributions import (
    days_to_close,
    days_to_report,
    inject_quality_issues,
    litigation_flags,
    lognormal_losses,
    seasonal_modifier_array,
    severity_for_specialty_batch,
)

# ── Helpers ───────────────────────────────────────────────────────────────

def _age_band(age: int) -> str:
    if age < 18:
        return "0-17"
    if age < 35:
        return "18-34"
    if age < 50:
        return "35-49"
    if age < 65:
        return "50-64"
    return "65+"


def _risk_segment(age_band: str, severity_weight: float) -> str:
    """Derive patient risk segment from age band and diagnosis severity weight.

    Rules (intentionally simple so downstream analytics have a clean signal):
      - ``critical`` : 65+ *and* weight >= 1.5
      - ``high``     : weight >= 1.5 or age 65+
      - ``medium``   : weight >= 1.0
      - ``low``      : everything else
    """
    elderly = age_band == "65+"
    if elderly and severity_weight >= 1.5:
        return "critical"
    if severity_weight >= 1.5 or elderly:
        return "high"
    if severity_weight >= 1.0:
        return "medium"
    return "low"


# ── Provider Generation ──────────────────────────────────────────────────

def _generate_providers(
    n: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Create a provider dimension table."""
    specialty_choices = rng.choice(len(SPECIALTIES), size=n)
    dept_choices = rng.choice(len(DEPARTMENTS), size=n)

    providers = pd.DataFrame(
        {
            "provider_id": [f"PRV{i+1:04d}" for i in range(n)],
            "specialty_id": [SPECIALTIES[s]["specialty_id"] for s in specialty_choices],
            "specialty_name": [SPECIALTIES[s]["name"] for s in specialty_choices],
            "department": [DEPARTMENTS[d] for d in dept_choices],
            "risk_tier": [SPECIALTIES[s]["risk_tier"] for s in specialty_choices],
        }
    )
    return providers


# ── Region Dimension ─────────────────────────────────────────────────────

def _build_region_table() -> pd.DataFrame:
    """Build a region dimension from the STATES reference list."""
    records = []
    for st in STATES:
        region = st["region"]
        mults = REGION_MULTIPLIERS[region]
        records.append(
            {
                "state_code": st["state_code"],
                "state_name": st["name"],
                "region": region,
                "urban_rural": st["urban_rural"],
                "volume_multiplier": mults["volume_multiplier"],
                "severity_multiplier": mults["severity_multiplier"],
            }
        )
    return pd.DataFrame(records)


# ── State Sampling Weights ───────────────────────────────────────────────

def _state_weights() -> tuple[np.ndarray, np.ndarray]:
    """Return aligned arrays of state codes and normalised sampling weights."""
    codes: List[str] = []
    weights: List[float] = []
    for st in STATES:
        codes.append(st["state_code"])
        weights.append(REGION_MULTIPLIERS[st["region"]]["volume_multiplier"])
    w = np.array(weights, dtype=float)
    w /= w.sum()
    return np.array(codes), w


# ── Main Generator ────────────────────────────────────────────────────────

def generate_claims(
    n: int = N_CLAIMS,
    seed: int = RANDOM_SEED,
) -> Dict[str, pd.DataFrame]:
    """Generate the full synthetic dataset and persist to CSV.

    Parameters
    ----------
    n : int
        Number of claim records (default from config).
    seed : int
        Seed for the numpy Generator.

    Returns
    -------
    dict
        Mapping of table name to DataFrame.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # ── Dimension tables ────────────────────────────────────────────────
    providers_df = _generate_providers(N_PROVIDERS, rng)
    regions_df = _build_region_table()
    diagnoses_df = pd.DataFrame(DIAGNOSES)
    root_causes_df = pd.DataFrame(ROOT_CAUSES)

    # ── Pre-compute lookup structures ───────────────────────────────────
    diag_weight_map: Dict[str, float] = {
        d["diagnosis_id"]: d["severity_weight"] for d in DIAGNOSES
    }
    state_codes, state_weights = _state_weights()
    state_region_map: Dict[str, str] = {s["state_code"]: s["region"] for s in STATES}

    # ── Incident dates (uniform with seasonal modulation) ───────────────
    start_ts = pd.Timestamp(DATE_START)
    end_ts = pd.Timestamp(DATE_END)
    total_days = (end_ts - start_ts).days

    # Over-sample then accept/reject using seasonal modifier
    candidate_offsets = rng.integers(0, total_days, size=int(n * 1.5))
    candidate_dates = start_ts + pd.to_timedelta(candidate_offsets, unit="D")
    modifiers = seasonal_modifier_array(pd.Series(candidate_dates))
    accept_prob = modifiers / modifiers.max()
    accepted = rng.random(len(candidate_dates)) < accept_prob
    incident_dates = candidate_dates[accepted][:n]

    # If we didn't get enough (very unlikely), top up uniformly
    if len(incident_dates) < n:
        shortfall = n - len(incident_dates)
        extra_offsets = rng.integers(0, total_days, size=shortfall)
        extra_dates = start_ts + pd.to_timedelta(extra_offsets, unit="D")
        incident_dates = np.concatenate([incident_dates, extra_dates])

    incident_dates = pd.to_datetime(incident_dates[:n])

    # ── Provider assignment ─────────────────────────────────────────────
    provider_idx = rng.integers(0, N_PROVIDERS, size=n)
    provider_ids = providers_df["provider_id"].values[provider_idx]
    specialty_names = providers_df["specialty_name"].values[provider_idx]

    # ── Severity (specialty-driven) ─────────────────────────────────────
    severities = severity_for_specialty_batch(specialty_names, rng)

    # ── State / Region ──────────────────────────────────────────────────
    sampled_states = rng.choice(state_codes, size=n, p=state_weights)

    # ── Diagnosis ───────────────────────────────────────────────────────
    diag_ids = rng.choice(
        [d["diagnosis_id"] for d in DIAGNOSES], size=n
    )
    diag_severity_weights = np.array(
        [diag_weight_map[d] for d in diag_ids], dtype=float
    )

    # ── Root Cause ──────────────────────────────────────────────────────
    rc_ids = rng.choice(
        [r["root_cause_id"] for r in ROOT_CAUSES], size=n
    )

    # ── Claim Type ──────────────────────────────────────────────────────
    claim_type_arr = rng.choice(
        CLAIM_TYPES,
        size=n,
        p=np.array(CLAIM_TYPE_WEIGHTS) / sum(CLAIM_TYPE_WEIGHTS),
    )

    # ── Procedure Category ──────────────────────────────────────────────
    procedure_arr = rng.choice(PROCEDURE_CATEGORIES, size=n)

    # ── Loss amounts ────────────────────────────────────────────────────
    paid_amounts = lognormal_losses(n, severities, rng)

    # Reserve typically exceeds paid; add random headroom
    reserve_mult = 1.0 + rng.exponential(0.3, size=n)
    reserve_amounts = paid_amounts * reserve_mult
    incurred_amounts = paid_amounts + reserve_amounts

    # ── Dates: report and close ─────────────────────────────────────────
    report_lag = days_to_report(severities, rng)
    report_dates = incident_dates + pd.to_timedelta(report_lag, unit="D")

    close_lag = days_to_close(severities, rng)

    # ── Status ──────────────────────────────────────────────────────────
    statuses = rng.choice(
        list(STATUS_WEIGHTS.keys()),
        size=n,
        p=list(STATUS_WEIGHTS.values()),
    )
    # Open / reopened claims have no close date
    close_dates = report_dates + pd.to_timedelta(close_lag, unit="D")
    close_dates = pd.Series(close_dates)
    close_dates[statuses != "closed"] = pd.NaT

    # ── Year derivations ────────────────────────────────────────────────
    accident_years = incident_dates.year
    report_years = report_dates.year
    development_years = report_years - accident_years

    # ── Patient demographics ────────────────────────────────────────────
    patient_ages = rng.integers(0, 96, size=n)
    age_bands = np.array([_age_band(a) for a in patient_ages])
    risk_segments = np.array(
        [
            _risk_segment(ab, sw)
            for ab, sw in zip(age_bands, diag_severity_weights)
        ]
    )

    # ── Litigation flag ─────────────────────────────────────────────────
    lit_flags = litigation_flags(severities, sampled_states, rng)

    # ── Repeat event flag (~8%) ─────────────────────────────────────────
    repeat_flags = rng.random(n) < 0.08

    # ── Assemble the fact table ─────────────────────────────────────────
    claims_df = pd.DataFrame(
        {
            "claim_id": [f"CLM{i+1:06d}" for i in range(n)],
            "incident_date": incident_dates,
            "report_date": report_dates,
            "close_date": close_dates.values,
            "accident_year": accident_years,
            "report_year": report_years,
            "development_year": development_years,
            "provider_id": provider_ids,
            "state_code": sampled_states,
            "region": [state_region_map[sc] for sc in sampled_states],
            "diagnosis_id": diag_ids,
            "root_cause_id": rc_ids,
            "claim_type": claim_type_arr,
            "procedure_category": procedure_arr,
            "severity": severities,
            "status": statuses,
            "paid_amount": np.round(paid_amounts, 2),
            "reserve_amount": np.round(reserve_amounts, 2),
            "incurred_amount": np.round(incurred_amounts, 2),
            "patient_age": patient_ages,
            "patient_age_band": age_bands,
            "patient_risk_segment": risk_segments,
            "litigation_flag": lit_flags.astype(int),
            "repeat_event_flag": repeat_flags.astype(int),
            "report_lag_days": report_lag,
        }
    )

    # ── Inject data-quality issues ──────────────────────────────────────
    claims_df = inject_quality_issues(claims_df, rng)

    # ── Persist to CSV ──────────────────────────────────────────────────
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    output_map = {
        "claims": claims_df,
        "providers": providers_df,
        "regions": regions_df,
        "diagnoses": diagnoses_df,
        "root_causes": root_causes_df,
    }

    for name, frame in output_map.items():
        path = RAW_DIR / f"{name}.csv"
        frame.to_csv(path, index=False)

    elapsed = time.perf_counter() - t0

    # ── Summary ─────────────────────────────────────────────────────────
    _print_summary(output_map, elapsed)

    return output_map


# ── Pretty-print Summary ─────────────────────────────────────────────────

def _print_summary(tables: Dict[str, pd.DataFrame], elapsed: float) -> None:
    claims = tables["claims"]
    divider = "-" * 60

    print(f"\n{'=' * 60}")
    print("  SYNTHETIC CLAIMS GENERATION — SUMMARY")
    print(f"{'=' * 60}\n")

    for name, df in tables.items():
        print(f"  {name:<16s}  {len(df):>8,} rows  ->  data/raw/{name}.csv")
    print()

    print(divider)
    print("  Fact Table Quick Stats (claims.csv)")
    print(divider)

    print(f"  Date range        : {claims['incident_date'].min().date()}"
          f" -> {claims['incident_date'].max().date()}")

    if "severity" in claims.columns:
        # Severity may have text values after quality injection; coerce
        sev_numeric = pd.to_numeric(claims["severity"], errors="coerce")
        print(f"  Severity (mean)   : {sev_numeric.mean():.2f}")

    print(f"  Paid amount (med) : ${claims['paid_amount'].median():,.0f}")
    print(f"  Paid amount (p99) : ${claims['paid_amount'].quantile(0.99):,.0f}")

    status_pcts = claims["status"].value_counts(normalize=True).mul(100).round(1)
    status_str = "  /  ".join(
        f"{k}: {v}%" for k, v in status_pcts.items()
    )
    print(f"  Status dist       : {status_str}")
    print(f"  Litigation rate   : {claims['litigation_flag'].mean():.1%}")
    print(f"  Repeat events     : {claims['repeat_event_flag'].mean():.1%}")

    # Quality-issue counts
    neg_paid = (claims["paid_amount"] < 0).sum()
    null_diag = claims["diagnosis_id"].isna().sum()
    outliers = (claims["paid_amount"] > 10_000_000).sum()
    print(f"\n  Injected quality issues:")
    print(f"    Negative paid amounts   : {neg_paid}")
    print(f"    NULL diagnoses          : {null_diag}")
    print(f"    Outliers (>$10M)        : {outliers}")

    print(f"\n  Generation time   : {elapsed:.2f}s")
    print(f"{'=' * 60}\n")


# ── CLI entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_claims()

"""SQLAlchemy ORM models for the insurance claims star schema.

The warehouse follows a classic Kimball star design:

*   **FactClaim** -- one row per claim; contains all measures and
    foreign keys into the four dimension tables.
*   **DimProvider** -- provider / specialty reference data.
*   **DimRegion** -- geographic reference data.
*   **DimDiagnosis** -- diagnosis reference data.
*   **DimRootCause** -- root-cause / preventability reference data.

Usage
-----
    from src.database.models import Base, FactClaim, DimProvider, ...

    Base.metadata.create_all(engine)   # create all tables
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Date,
    Float,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


# ── Declarative Base ─────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """Shared declarative base for every ORM model in the warehouse."""
    pass


# ── Dimension Tables ─────────────────────────────────────────────────────

class DimProvider(Base):
    """Provider / specialty dimension."""

    __tablename__ = "dim_provider"

    provider_key = Column(Integer, primary_key=True, autoincrement=True)
    provider_id = Column(String(20), nullable=False, unique=True, index=True)
    provider_name = Column(String(100), nullable=True)
    specialty = Column(String(80), nullable=False)
    department = Column(String(80), nullable=False)
    risk_tier = Column(String(20), nullable=False)

    # Back-reference to fact table
    claims = relationship("FactClaim", back_populates="provider")

    def __repr__(self) -> str:
        return (
            f"<DimProvider(provider_key={self.provider_key}, "
            f"provider_id='{self.provider_id}', "
            f"specialty='{self.specialty}')>"
        )


class DimRegion(Base):
    """Geographic region dimension."""

    __tablename__ = "dim_region"

    region_key = Column(Integer, primary_key=True, autoincrement=True)
    state_code = Column(String(2), nullable=False, unique=True, index=True)
    state_name = Column(String(50), nullable=False)
    region = Column(String(30), nullable=False)
    urban_rural = Column(String(20), nullable=False)

    claims = relationship("FactClaim", back_populates="region_dim")

    def __repr__(self) -> str:
        return (
            f"<DimRegion(region_key={self.region_key}, "
            f"state_code='{self.state_code}', "
            f"region='{self.region}')>"
        )


class DimDiagnosis(Base):
    """Diagnosis dimension."""

    __tablename__ = "dim_diagnosis"

    diagnosis_key = Column(Integer, primary_key=True, autoincrement=True)
    diagnosis_code = Column(String(20), nullable=False, unique=True, index=True)
    diagnosis_category = Column(String(80), nullable=False)
    severity_weight = Column(Float, nullable=False, default=1.0)

    claims = relationship("FactClaim", back_populates="diagnosis")

    def __repr__(self) -> str:
        return (
            f"<DimDiagnosis(diagnosis_key={self.diagnosis_key}, "
            f"diagnosis_code='{self.diagnosis_code}')>"
        )


class DimRootCause(Base):
    """Root-cause / preventability dimension."""

    __tablename__ = "dim_root_cause"

    root_cause_key = Column(Integer, primary_key=True, autoincrement=True)
    root_cause_code = Column(String(20), nullable=False, unique=True, index=True)
    root_cause_category = Column(String(80), nullable=False)
    preventability = Column(String(30), nullable=False)

    claims = relationship("FactClaim", back_populates="root_cause")

    def __repr__(self) -> str:
        return (
            f"<DimRootCause(root_cause_key={self.root_cause_key}, "
            f"root_cause_code='{self.root_cause_code}')>"
        )


# ── Fact Table ───────────────────────────────────────────────────────────

class FactClaim(Base):
    """Central fact table -- one row per insurance claim."""

    __tablename__ = "fact_claim"
    __table_args__ = (
        CheckConstraint("severity_level BETWEEN 1 AND 5", name="ck_severity_range"),
        CheckConstraint("paid_amount >= 0", name="ck_paid_nonneg"),
        CheckConstraint("incurred_amount >= 0", name="ck_incurred_nonneg"),
    )

    # Primary key
    claim_id = Column(String(20), primary_key=True)

    # Dates
    incident_date = Column(Date, nullable=False)
    report_date = Column(Date, nullable=False)
    close_date = Column(Date, nullable=True)

    # Foreign keys to dimensions
    provider_key = Column(
        Integer, ForeignKey("dim_provider.provider_key"), nullable=False, index=True
    )
    region_key = Column(
        Integer, ForeignKey("dim_region.region_key"), nullable=False, index=True
    )
    diagnosis_key = Column(
        Integer, ForeignKey("dim_diagnosis.diagnosis_key"), nullable=False, index=True
    )
    root_cause_key = Column(
        Integer, ForeignKey("dim_root_cause.root_cause_key"), nullable=False, index=True
    )

    # Categorical attributes
    claim_type = Column(String(40), nullable=False)
    procedure_category = Column(String(80), nullable=False)
    severity_level = Column(SmallInteger, nullable=False)
    status = Column(String(20), nullable=False, index=True)

    # Monetary measures
    paid_amount = Column(Float, nullable=False, default=0.0)
    incurred_amount = Column(Float, nullable=False, default=0.0)
    reserved_amount = Column(Float, nullable=False, default=0.0)

    # Derived / pre-computed measures
    days_to_close = Column(Integer, nullable=True)
    days_to_report = Column(Integer, nullable=False)

    # Patient demographics
    patient_age_band = Column(String(20), nullable=False)
    patient_risk_segment = Column(String(20), nullable=False)

    # Flags
    repeat_event_flag = Column(Boolean, nullable=False, default=False)
    litigation_flag = Column(Boolean, nullable=False, default=False)

    # Time intelligence
    accident_year = Column(Integer, nullable=False, index=True)
    development_year = Column(Integer, nullable=False)

    # ── ORM relationships ────────────────────────────────────────────────
    provider = relationship("DimProvider", back_populates="claims")
    region_dim = relationship("DimRegion", back_populates="claims")
    diagnosis = relationship("DimDiagnosis", back_populates="claims")
    root_cause = relationship("DimRootCause", back_populates="claims")

    def __repr__(self) -> str:
        return (
            f"<FactClaim(claim_id='{self.claim_id}', "
            f"status='{self.status}', "
            f"paid_amount={self.paid_amount:.2f})>"
        )

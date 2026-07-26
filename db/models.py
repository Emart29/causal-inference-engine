import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, JSON,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class AnalysisStatus(str, enum.Enum):
    draft = "draft"
    running = "running"
    complete = "complete"
    failed = "failed"


class EstimateType(str, enum.Enum):
    ate = "ate"
    att = "att"
    cate = "cate"
    late = "late"


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    storage_path: Mapped[str] = mapped_column(String, nullable=False)
    n_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_cols: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    columns_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    description: Mapped[str | None] = mapped_column(String, nullable=True)


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    treatment: Mapped[str] = mapped_column(String, nullable=False)
    outcome: Mapped[str] = mapped_column(String, nullable=False)
    covariates: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    instrument: Mapped[str | None] = mapped_column(String, nullable=True)
    time_col: Mapped[str | None] = mapped_column(String, nullable=True)
    group_col: Mapped[str | None] = mapped_column(String, nullable=True)
    dag_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(
        Enum(AnalysisStatus, name="analysis_status_enum"), nullable=False, default=AnalysisStatus.draft
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)


class EffectEstimate(Base):
    __tablename__ = "effect_estimates"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("analyses.id"), nullable=False, index=True)
    method: Mapped[str] = mapped_column(String, nullable=False)
    estimate_type: Mapped[str] = mapped_column(
        Enum(EstimateType, name="estimate_type_enum"), nullable=False, default=EstimateType.ate
    )
    point_estimate: Mapped[float] = mapped_column(Float, nullable=False)
    ci_low: Mapped[float] = mapped_column(Float, nullable=False)
    ci_high: Mapped[float] = mapped_column(Float, nullable=False)
    std_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_treated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_control: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    diagnostics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RefutationResult(Base):
    __tablename__ = "refutation_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    estimate_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("effect_estimates.id"), nullable=False, index=True)
    test_name: Mapped[str] = mapped_column(String, nullable=False)
    original_effect: Mapped[float] = mapped_column(Float, nullable=False)
    new_effect: Mapped[float] = mapped_column(Float, nullable=False)
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    interpretation: Mapped[str] = mapped_column(String, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("analyses.id"), nullable=False, index=True)
    storage_path: Mapped[str] = mapped_column(String, nullable=False)
    summary_text: Mapped[str] = mapped_column(String, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

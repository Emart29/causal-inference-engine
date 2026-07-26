"""Persistence layer for analyses and their results.

Every method takes an active :class:`~sqlalchemy.ext.asyncio.AsyncSession` bound
at construction time, so the caller controls the transaction boundary and no
session is ever shared across event loops.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import (
    Analysis,
    AnalysisStatus,
    Dataset,
    EffectEstimate,
    RefutationResult,
    Report,
)


class MetadataStore:
    """Create and query the records that make up a causal analysis."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    async def create_dataset(
        self,
        name: str,
        storage_path: str,
        n_rows: int,
        n_cols: int,
        columns_json: dict,
        description: str | None = None,
    ) -> Dataset:
        """Register an uploaded dataset and return the stored record."""
        dataset = Dataset(
            name=name,
            storage_path=storage_path,
            n_rows=n_rows,
            n_cols=n_cols,
            columns_json=columns_json,
            description=description,
        )
        self._session.add(dataset)
        await self._session.commit()
        await self._session.refresh(dataset)
        return dataset

    async def get_dataset(self, dataset_id: uuid.UUID) -> Dataset | None:
        """Return a dataset by id, or ``None`` if it does not exist."""
        return await self._session.get(Dataset, dataset_id)

    async def list_datasets(self, limit: int = 100) -> list[Dataset]:
        """Return recently uploaded datasets, newest first."""
        result = await self._session.execute(
            select(Dataset).order_by(Dataset.uploaded_at.desc()).limit(limit)
        )
        return list(result.scalars().all())

    async def delete_dataset(self, dataset_id: uuid.UUID) -> bool:
        """Delete a dataset record. Returns whether a record was removed."""
        dataset = await self.get_dataset(dataset_id)
        if dataset is None:
            return False
        await self._session.delete(dataset)
        await self._session.commit()
        return True

    # ------------------------------------------------------------------
    # Analyses
    # ------------------------------------------------------------------

    async def create_analysis(
        self,
        dataset_id: uuid.UUID,
        name: str,
        treatment: str,
        outcome: str,
        covariates: list[str],
        instrument: str | None = None,
        time_col: str | None = None,
        group_col: str | None = None,
    ) -> Analysis:
        """Create a draft analysis describing the causal question to be answered."""
        analysis = Analysis(
            dataset_id=dataset_id,
            name=name,
            treatment=treatment,
            outcome=outcome,
            covariates=covariates,
            instrument=instrument,
            time_col=time_col,
            group_col=group_col,
            status=AnalysisStatus.draft,
        )
        self._session.add(analysis)
        await self._session.commit()
        await self._session.refresh(analysis)
        return analysis

    async def get_analysis(self, analysis_id: uuid.UUID) -> Analysis | None:
        """Return an analysis by id, or ``None`` if it does not exist."""
        return await self._session.get(Analysis, analysis_id)

    async def list_analyses(self, status: str | None = None, limit: int = 100) -> list[Analysis]:
        """Return analyses, newest first, optionally filtered by status."""
        query = select(Analysis).order_by(Analysis.created_at.desc()).limit(limit)
        if status is not None:
            query = query.where(Analysis.status == status)
        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def update_analysis(self, analysis_id: uuid.UUID, **fields) -> Analysis | None:
        """Apply field updates to an analysis and return the refreshed record."""
        if fields:
            await self._session.execute(
                update(Analysis).where(Analysis.id == analysis_id).values(**fields)
            )
            await self._session.commit()
        return await self.get_analysis(analysis_id)

    async def set_dag(self, analysis_id: uuid.UUID, dag_json: dict) -> Analysis | None:
        """Store the causal graph the analysis is identified under."""
        return await self.update_analysis(analysis_id, dag_json=dag_json)

    # ------------------------------------------------------------------
    # Effect estimates
    # ------------------------------------------------------------------

    async def add_estimate(
        self,
        analysis_id: uuid.UUID,
        method: str,
        estimate_type: str,
        point_estimate: float,
        ci_low: float,
        ci_high: float,
        n_treated: int,
        n_control: int,
        std_error: float | None = None,
        p_value: float | None = None,
        diagnostics: dict | None = None,
    ) -> EffectEstimate:
        """Record one estimator's result for an analysis."""
        estimate = EffectEstimate(
            analysis_id=analysis_id,
            method=method,
            estimate_type=estimate_type,
            point_estimate=point_estimate,
            ci_low=ci_low,
            ci_high=ci_high,
            std_error=std_error,
            p_value=p_value,
            n_treated=n_treated,
            n_control=n_control,
            diagnostics=diagnostics or {},
        )
        self._session.add(estimate)
        await self._session.commit()
        await self._session.refresh(estimate)
        return estimate

    async def get_estimates(self, analysis_id: uuid.UUID) -> list[EffectEstimate]:
        """Return every estimate recorded for an analysis, oldest first."""
        result = await self._session.execute(
            select(EffectEstimate)
            .where(EffectEstimate.analysis_id == analysis_id)
            .order_by(EffectEstimate.created_at.asc())
        )
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Refutations
    # ------------------------------------------------------------------

    async def add_refutation(
        self,
        estimate_id: uuid.UUID,
        test_name: str,
        original_effect: float,
        new_effect: float,
        passed: bool,
        interpretation: str,
        p_value: float | None = None,
    ) -> RefutationResult:
        """Record the outcome of one robustness check against an estimate."""
        refutation = RefutationResult(
            estimate_id=estimate_id,
            test_name=test_name,
            original_effect=original_effect,
            new_effect=new_effect,
            p_value=p_value,
            passed=passed,
            interpretation=interpretation,
        )
        self._session.add(refutation)
        await self._session.commit()
        await self._session.refresh(refutation)
        return refutation

    async def get_refutations(self, estimate_id: uuid.UUID) -> list[RefutationResult]:
        """Return every refutation result recorded against an estimate."""
        result = await self._session.execute(
            select(RefutationResult)
            .where(RefutationResult.estimate_id == estimate_id)
            .order_by(RefutationResult.created_at.asc())
        )
        return list(result.scalars().all())

    async def refutation_summary(self, analysis_id: uuid.UUID) -> dict:
        """Summarise how many robustness checks each estimate survived.

        Returns:
            Mapping of estimator method name to ``{"passed": int, "total": int,
            "verdict": str}``, where the verdict is ``robust`` when every check
            passed, ``fragile`` when some failed, and ``failed`` when none passed.
        """
        summary: dict[str, dict] = {}
        for estimate in await self.get_estimates(analysis_id):
            results = await self.get_refutations(estimate.id)
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            if total == 0:
                verdict = "untested"
            elif passed == total:
                verdict = "robust"
            elif passed == 0:
                verdict = "failed"
            else:
                verdict = "fragile"
            summary[estimate.method] = {"passed": passed, "total": total, "verdict": verdict}
        return summary

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    async def save_report(
        self, analysis_id: uuid.UUID, storage_path: str, summary_text: str
    ) -> Report:
        """Record a generated report and where its PDF is stored."""
        report = Report(
            analysis_id=analysis_id, storage_path=storage_path, summary_text=summary_text
        )
        self._session.add(report)
        await self._session.commit()
        await self._session.refresh(report)
        return report

    async def get_report(self, analysis_id: uuid.UUID) -> Report | None:
        """Return the most recent report for an analysis, if one exists."""
        result = await self._session.execute(
            select(Report)
            .where(Report.analysis_id == analysis_id)
            .order_by(Report.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

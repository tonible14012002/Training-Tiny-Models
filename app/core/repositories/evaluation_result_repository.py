from typing import Optional, List
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from app.core.models.models import EvaluationResult, utc_now


class EvaluationResultRepository:
    """Repository for EvaluationResult operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        trained_model_id: str,
        human_test_set_id: Optional[str] = None,
        dataset_file_id: Optional[str] = None,
        accuracy: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        f1_score: Optional[float] = None,
        label_metrics: Optional[str] = None,
        metrics: Optional[str] = None
    ) -> EvaluationResult:
        """Create a new evaluation result record"""
        evaluation_result = EvaluationResult(
            trained_model_id=trained_model_id,
            human_test_set_id=human_test_set_id,
            dataset_file_id=dataset_file_id,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            label_metrics=label_metrics,
            metrics=metrics
        )
        self.session.add(evaluation_result)
        await self.session.commit()
        await self.session.refresh(evaluation_result)
        return evaluation_result

    async def get_by_id(self, evaluation_id: str) -> Optional[EvaluationResult]:
        """Get evaluation result by ID"""
        return await self.session.get(EvaluationResult, evaluation_id)

    async def get_by_trained_model(self, trained_model_id: str) -> List[EvaluationResult]:
        """Get all evaluation results for a trained model"""
        statement = select(EvaluationResult).where(
            EvaluationResult.trained_model_id == trained_model_id
        ).order_by(EvaluationResult.evaluated_at.desc())
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_latest_by_trained_model(self, trained_model_id: str) -> Optional[EvaluationResult]:
        """Get the latest evaluation result for a trained model"""
        statement = select(EvaluationResult).where(
            EvaluationResult.trained_model_id == trained_model_id
        ).order_by(EvaluationResult.evaluated_at.desc())
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def get_by_human_test_set(self, human_test_set_id: str) -> List[EvaluationResult]:
        """Get all evaluation results for a human test set"""
        statement = select(EvaluationResult).where(
            EvaluationResult.human_test_set_id == human_test_set_id
        ).order_by(EvaluationResult.evaluated_at.desc())
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_by_dataset_file(self, dataset_file_id: str) -> List[EvaluationResult]:
        """Get all evaluation results for a dataset file"""
        statement = select(EvaluationResult).where(
            EvaluationResult.dataset_file_id == dataset_file_id
        ).order_by(EvaluationResult.evaluated_at.desc())
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_by_accuracy_threshold(
        self,
        min_accuracy: float,
        trained_model_id: Optional[str] = None
    ) -> List[EvaluationResult]:
        """Get evaluation results above a certain accuracy threshold"""
        statement = select(EvaluationResult).where(
            EvaluationResult.accuracy >= min_accuracy
        )
        if trained_model_id:
            statement = statement.where(
                EvaluationResult.trained_model_id == trained_model_id
            )
        statement = statement.order_by(EvaluationResult.accuracy.desc())
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_by_f1_threshold(
        self,
        min_f1_score: float,
        trained_model_id: Optional[str] = None
    ) -> List[EvaluationResult]:
        """Get evaluation results above a certain F1 score threshold"""
        statement = select(EvaluationResult).where(
            EvaluationResult.f1_score >= min_f1_score
        )
        if trained_model_id:
            statement = statement.where(
                EvaluationResult.trained_model_id == trained_model_id
            )
        statement = statement.order_by(EvaluationResult.f1_score.desc())
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_best_evaluation_by_metric(
        self,
        metric_name: str,
        trained_model_id: Optional[str] = None
    ) -> Optional[EvaluationResult]:
        """Get the evaluation with the best score for a specific metric

        Args:
            metric_name: One of 'accuracy', 'precision', 'recall', 'f1_score'
            trained_model_id: Optional filter by trained model
        """
        statement = select(EvaluationResult)

        if trained_model_id:
            statement = statement.where(
                EvaluationResult.trained_model_id == trained_model_id
            )

        # Order by the specified metric
        if metric_name == "accuracy":
            statement = statement.order_by(EvaluationResult.accuracy.desc())
        elif metric_name == "precision":
            statement = statement.order_by(EvaluationResult.precision.desc())
        elif metric_name == "recall":
            statement = statement.order_by(EvaluationResult.recall.desc())
        elif metric_name == "f1_score":
            statement = statement.order_by(EvaluationResult.f1_score.desc())
        else:
            raise ValueError(f"Invalid metric name: {metric_name}")

        result = await self.session.execute(statement)
        return result.scalars().first()

    async def get_evaluations_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        trained_model_id: Optional[str] = None
    ) -> List[EvaluationResult]:
        """Get evaluation results within a date range"""
        statement = select(EvaluationResult).where(
            EvaluationResult.evaluated_at >= start_date,
            EvaluationResult.evaluated_at <= end_date
        )
        if trained_model_id:
            statement = statement.where(
                EvaluationResult.trained_model_id == trained_model_id
            )
        statement = statement.order_by(EvaluationResult.evaluated_at.desc())
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def update(
        self,
        evaluation_id: str,
        accuracy: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        f1_score: Optional[float] = None,
        label_metrics: Optional[str] = None,
        metrics: Optional[str] = None
    ) -> Optional[EvaluationResult]:
        """Update evaluation result metrics"""
        evaluation_result = await self.get_by_id(evaluation_id)
        if not evaluation_result:
            return None

        if accuracy is not None:
            evaluation_result.accuracy = accuracy
        if precision is not None:
            evaluation_result.precision = precision
        if recall is not None:
            evaluation_result.recall = recall
        if f1_score is not None:
            evaluation_result.f1_score = f1_score
        if label_metrics is not None:
            evaluation_result.label_metrics = label_metrics
        if metrics is not None:
            evaluation_result.metrics = metrics

        self.session.add(evaluation_result)
        await self.session.commit()
        await self.session.refresh(evaluation_result)
        return evaluation_result

    async def delete(self, evaluation_id: str) -> bool:
        """Delete evaluation result"""
        evaluation_result = await self.get_by_id(evaluation_id)
        if not evaluation_result:
            return False

        self.session.delete(evaluation_result)
        await self.session.commit()
        return True

    async def get_all_evaluations(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[EvaluationResult]:
        """Get all evaluation results with optional pagination"""
        statement = select(EvaluationResult).order_by(
            EvaluationResult.evaluated_at.desc()
        )

        if offset:
            statement = statement.offset(offset)
        if limit:
            statement = statement.limit(limit)

        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def count_evaluations_by_trained_model(self, trained_model_id: str) -> int:
        """Count the number of evaluations for a trained model"""
        from sqlalchemy import func
        statement = select(func.count()).select_from(EvaluationResult).where(
            EvaluationResult.trained_model_id == trained_model_id
        )
        result = await self.session.execute(statement)
        return result.scalar_one()

    async def get_average_metrics_by_trained_model(self, trained_model_id: str) -> dict:
        """Get average metrics for all evaluations of a trained model"""
        from sqlalchemy import func
        statement = select(
            func.avg(EvaluationResult.accuracy).label("avg_accuracy"),
            func.avg(EvaluationResult.precision).label("avg_precision"),
            func.avg(EvaluationResult.recall).label("avg_recall"),
            func.avg(EvaluationResult.f1_score).label("avg_f1_score")
        ).where(
            EvaluationResult.trained_model_id == trained_model_id
        )
        result = await self.session.execute(statement)
        row = result.first()

        if row:
            return {
                "avg_accuracy": float(row.avg_accuracy) if row.avg_accuracy else None,
                "avg_precision": float(row.avg_precision) if row.avg_precision else None,
                "avg_recall": float(row.avg_recall) if row.avg_recall else None,
                "avg_f1_score": float(row.avg_f1_score) if row.avg_f1_score else None
            }
        return {}

    async def get_latest_by_trained_model_ids(self, trained_model_ids: List[str]) -> dict[str, Optional[EvaluationResult]]:
        """Get the latest evaluation result for each trained model

        Args:
            trained_model_ids: List of trained model IDs to get evaluations for

        Returns:
            Dictionary mapping trained_model_id to the latest EvaluationResult for that model
        """
        from sqlalchemy import func

        # Subquery to get the latest evaluated_at for each trained_model
        latest_subquery = (
            select(
                EvaluationResult.trained_model_id,
                func.max(EvaluationResult.evaluated_at).label("max_evaluated_at")
            )
            .where(EvaluationResult.trained_model_id.in_(trained_model_ids))
            .group_by(EvaluationResult.trained_model_id)
            .subquery()
        )

        # Get the evaluation results
        statement = (
            select(EvaluationResult)
            .join(
                latest_subquery,
                (EvaluationResult.trained_model_id == latest_subquery.c.trained_model_id) &
                (EvaluationResult.evaluated_at == latest_subquery.c.max_evaluated_at)
            )
        )
        result = await self.session.execute(statement)
        evaluations = list(result.scalars().all())

        # Build result dict
        result_dict: dict[str, Optional[EvaluationResult]] = {model_id: None for model_id in trained_model_ids}
        for evaluation in evaluations:
            result_dict[evaluation.trained_model_id] = evaluation

        return result_dict

    async def get_latest_by_phase(self, phase_id: str) -> Optional[EvaluationResult]:
        """Get the latest evaluation result for a given phase

        Args:
            phase_id: The phase ID to get evaluation for

        Returns:
            The latest EvaluationResult for the phase, or None if not found
        """
        from app.core.models.models import TrainedModel

        # Get trained models for this phase
        trained_models_statement = select(TrainedModel).where(
            TrainedModel.phase_id == phase_id
        ).order_by(TrainedModel.created_at.desc())
        trained_models_result = await self.session.execute(trained_models_statement)
        trained_models = list(trained_models_result.scalars().all())

        if not trained_models:
            return None

        # Get the latest trained model
        latest_trained_model = trained_models[0]

        # Get the latest evaluation for this model
        return await self.get_latest_by_trained_model(latest_trained_model.id)

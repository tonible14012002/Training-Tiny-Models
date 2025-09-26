import logging
import time
from typing import Dict, Optional

from app.core.services.data_generator.data_generator import DataGenerator
from app.core.services.trainer.trainer import TrainerService
from app.core.services.model_analyzer.model_analyzer import ModelAnalyzer
from app.core.services.data_manager.data_manager import DataManager
from app.core.services.eval_data_manager.eval_data_manager import EvalDataManager
from app.core.schemas.orchestrator import (
    IterationMetrics,
    PipelineConfig,
    PipelineStatus
)

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """
    Orchestrates the automated training pipeline that coordinates all services
    to iteratively improve the model through data generation, training, and evaluation.

    Pipeline: Generate Data -> Train Model -> Evaluate -> Check Metrics -> Repeat
    """

    def __init__(
        self,
        data_generator: DataGenerator,
        trainer_service: TrainerService,
        model_analyzer: ModelAnalyzer,
        data_manager: DataManager,
        eval_data_manager: EvalDataManager
    ):
        self.data_generator = data_generator
        self.trainer_service = trainer_service
        self.model_analyzer = model_analyzer
        self.data_manager = data_manager
        self.eval_data_manager = eval_data_manager

        self.status = PipelineStatus()
        self.logger = logging.getLogger(__name__)

    async def start_auto_training_pipeline(
        self,
        config: Optional[PipelineConfig] = None
    ) -> Dict:
        """
        Start the automated training pipeline.

        Args:
            config: Pipeline configuration (uses defaults if None)

        Returns:
            Dictionary with pipeline results and final status
        """
        if self.status.is_running:
            return {
                "error": "Pipeline is already running",
                "status": "error",
                "current_iteration": self.status.current_iteration
            }

        config = config or PipelineConfig()

        self.logger.info(f"Starting auto-training pipeline with max_iterations={config.max_iterations}")

        # Initialize pipeline status
        self.status = PipelineStatus(
            is_running=True,
            current_iteration=0,
            total_iterations=config.max_iterations,
            start_time=time.time(),
            metrics_history=[]
        )

        try:
            # Load evaluation dataset once at the beginning
            eval_samples = self.eval_data_manager.load()
            if not eval_samples or len(eval_samples) == 0:
                raise ValueError("No evaluation dataset available. Please generate evaluation data first.")

            # Convert to dataset format for model evaluation
            from datasets import Dataset
            from app.core.schemas import PAYMENT_LABEL

            eval_data = []
            for sample in eval_samples:
                eval_data.append({
                    "msg": sample.msg,
                    "label": PAYMENT_LABEL.from_str(sample.label)
                })

            eval_dataset = Dataset.from_list(eval_data)
            self.logger.info(f"Loaded evaluation dataset with {len(eval_dataset)} samples")

            # Run the iterative pipeline
            for iteration in range(1, config.max_iterations + 1):
                self.status.current_iteration = iteration
                self.logger.info(f"=== Starting Iteration {iteration}/{config.max_iterations} ===")

                # Step 1: Generate new training data
                self.logger.info("Step 1: Generating new training data...")

                # Load human seeds for data generation (optional)
                human_seeds = self._load_human_seeds()

                # Generate new training data using fresh_gen
                if iteration != 1:
                    await self.data_generator.fresh_gen(human_seeds)

                self.logger.info(f"Generated new training data using {len(human_seeds) if human_seeds else 0} human seeds")

                # Step 2: Train model with latest data
                self.logger.info("Step 2: Training model with updated dataset...")
                training_start = time.time()

                # Get dataset from data manager
                dataset = self.data_manager.to_datasets()
                checkpoint_num = await self.trainer_service.train(dataset)
                checkpoint_path = f"{self.trainer_service.CHECKPOINT_DIR}/{checkpoint_num}"

                training_time = time.time() - training_start
                self.logger.info(f"Training completed in {training_time:.2f}s. Checkpoint: {checkpoint_path}")

                # Step 3: Evaluate model
                self.logger.info("Step 3: Evaluating model performance...")
                evaluation_start = time.time()

                # Load the trained model for evaluation
                self.model_analyzer.load_model(checkpoint_path)
                eval_result = self.model_analyzer.analyze_model(eval_dataset)

                evaluation_time = time.time() - evaluation_start

                # Step 4: Record metrics
                iteration_metrics = IterationMetrics(
                    iteration=iteration,
                    accuracy=eval_result.overall.accuracy,
                    macro_f1=eval_result.overall.macro_f1,
                    coverage=eval_result.overall.coverage,
                    unknown_rate=eval_result.overall.unknown_rate,
                    total_samples=eval_result.overall.total_samples,
                    checkpoint_path=checkpoint_path,
                    timestamp=time.time(),
                    training_time=training_time,
                    evaluation_time=evaluation_time
                )

                self.status.metrics_history.append(iteration_metrics)
                self.status.last_metrics = iteration_metrics

                # Update best metrics
                if (self.status.best_metrics is None or
                    iteration_metrics.macro_f1 > self.status.best_metrics.macro_f1):
                    self.status.best_metrics = iteration_metrics

                self.logger.info(
                    f"Iteration {iteration} Results: "
                    f"Accuracy={iteration_metrics.accuracy:.3f}, "
                    f"Macro-F1={iteration_metrics.macro_f1:.3f}, "
                    f"Coverage={iteration_metrics.coverage:.3f}"
                )

                # Step 5: Check termination conditions
                termination_reason = self._check_termination_conditions(config, iteration_metrics)
                if termination_reason:
                    self.status.termination_reason = termination_reason
                    self.logger.info(f"Pipeline terminated: {termination_reason}")
                    break

            # Pipeline completed
            self.status.is_running = False
            total_time = time.time() - self.status.start_time

            if not self.status.termination_reason:
                self.status.termination_reason = f"Completed maximum iterations ({config.max_iterations})"

            result = {
                "success": True,
                "status": "completed",
                "termination_reason": self.status.termination_reason,
                "total_iterations": len(self.status.metrics_history),
                "total_time": total_time,
                "best_metrics": {
                    "iteration": self.status.best_metrics.iteration,
                    "accuracy": self.status.best_metrics.accuracy,
                    "macro_f1": self.status.best_metrics.macro_f1,
                    "coverage": self.status.best_metrics.coverage,
                    "checkpoint_path": self.status.best_metrics.checkpoint_path
                } if self.status.best_metrics else None,
                "final_metrics": {
                    "accuracy": self.status.last_metrics.accuracy,
                    "macro_f1": self.status.last_metrics.macro_f1,
                    "coverage": self.status.last_metrics.coverage,
                } if self.status.last_metrics else None,
                "metrics_history": [
                    {
                        "iteration": m.iteration,
                        "accuracy": m.accuracy,
                        "macro_f1": m.macro_f1,
                        "coverage": m.coverage,
                        "unknown_rate": m.unknown_rate,
                        "checkpoint_path": m.checkpoint_path
                    } for m in self.status.metrics_history
                ]
            }

            self.logger.info(f"Pipeline completed successfully in {total_time:.2f}s")
            return result

        except Exception as e:
            self.status.is_running = False
            self.status.termination_reason = f"Error: {str(e)}"
            self.logger.error(f"Pipeline failed: {e}")

            return {
                "success": False,
                "status": "error",
                "error": str(e),
                "termination_reason": self.status.termination_reason,
                "iterations_completed": len(self.status.metrics_history)
            }

    def _check_termination_conditions(
        self,
        config: PipelineConfig,
        current_metrics: IterationMetrics
    ) -> Optional[str]:
        """
        Check if the pipeline should terminate based on various conditions.

        Returns:
            Termination reason string if should terminate, None otherwise
        """
        # Check target achievement
        if (current_metrics.accuracy >= config.target_accuracy and
            current_metrics.macro_f1 >= config.target_macro_f1):
            return f"Target metrics achieved (Acc: {current_metrics.accuracy:.3f} >= {config.target_accuracy}, F1: {current_metrics.macro_f1:.3f} >= {config.target_macro_f1})"

        # Check early termination based on lack of improvement
        if len(self.status.metrics_history) >= config.min_improvement_iterations + 1:
            recent_metrics = self.status.metrics_history[-(config.min_improvement_iterations + 1):]

            # Check if there's been no significant improvement in macro_f1
            baseline_f1 = recent_metrics[0].macro_f1
            recent_improvements = [m.macro_f1 - baseline_f1 for m in recent_metrics[1:]]

            max_improvement = max(recent_improvements) if recent_improvements else 0

            if max_improvement < config.early_termination_threshold:
                return f"Early termination: No significant improvement (max: {max_improvement:.3f} < {config.early_termination_threshold}) over last {config.min_improvement_iterations} iterations"

        return None

    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status"""
        return {
            "is_running": self.status.is_running,
            "current_iteration": self.status.current_iteration,
            "total_iterations": self.status.total_iterations,
            "termination_reason": self.status.termination_reason,
            "last_metrics": {
                "accuracy": self.status.last_metrics.accuracy,
                "macro_f1": self.status.last_metrics.macro_f1,
                "coverage": self.status.last_metrics.coverage,
                "iteration": self.status.last_metrics.iteration
            } if self.status.last_metrics else None,
            "best_metrics": {
                "accuracy": self.status.best_metrics.accuracy,
                "macro_f1": self.status.best_metrics.macro_f1,
                "coverage": self.status.best_metrics.coverage,
                "iteration": self.status.best_metrics.iteration
            } if self.status.best_metrics else None,
            "metrics_history": [
                {
                    "iteration": m.iteration,
                    "accuracy": m.accuracy,
                    "macro_f1": m.macro_f1,
                    "coverage": m.coverage
                } for m in self.status.metrics_history
            ] if self.status.metrics_history else []
        }

    def stop_pipeline(self) -> Dict:
        """Stop the currently running pipeline"""
        if not self.status.is_running:
            return {"success": False, "message": "No pipeline is currently running"}

        self.status.is_running = False
        self.status.termination_reason = "Manual termination requested"

        self.logger.info("Pipeline manually stopped")

        return {
            "success": True,
            "message": "Pipeline stopped successfully",
            "iterations_completed": len(self.status.metrics_history)
        }

    def reset_pipeline(self) -> Dict:
        """Reset pipeline status and history"""
        if self.status.is_running:
            return {"success": False, "message": "Cannot reset while pipeline is running"}

        self.status = PipelineStatus()

        self.logger.info("Pipeline status reset")

        return {"success": True, "message": "Pipeline reset successfully"}

    def _load_human_seeds(self):
        """Load human seeds from the cache"""
        import json
        import os
        from app.core.schemas.workflow import Sample

        seed_file = ".cache/human_seed.json"
        if not os.path.exists(seed_file):
            return []

        try:
            with open(seed_file, 'r') as f:
                data = json.load(f)
                return [Sample(**item) for item in data]
        except Exception as e:
            self.logger.warning(f"Failed to load human seeds: {e}")
            return []
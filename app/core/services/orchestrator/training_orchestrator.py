import logging
import time
from typing import Dict, Optional, Type, List

from app.core.services.data_generator.data_generator_v2 import DataGeneratorV2
from app.core.services.trainer.trainer import TrainerService
from app.core.services.model_analyzer.model_analyzer import ModelAnalyzer
from app.core.services.data_manager.data_manager import DataManager
from app.core.services.eval_data_manager.eval_data_manager import EvalDataManager
from app.core.services.error_pattern_analyzer.error_pattern_analyzer import ErrorPatternAnalysisService
from app.core.services.prompt_builder.prompt_builder import PromptBuilder
from app.core.schemas.workflow import BaseLabelConfig
from app.core.schemas.orchestrator import (
    IterationMetrics,
    PipelineStatus
)
from src.payment_classifier.inference.prob_inference import ProbModelInference

logger = logging.getLogger(__name__)

class TrainingOrchestrator:
    """
    Orchestrates the automated training pipeline that coordinates all services
    to iteratively improve the model through data generation, training, and evaluation.

    Pipeline: Generate Data -> Train Model -> Evaluate -> Check Metrics -> Repeat
    """

    def __init__(
        self,
        data_generator: DataGeneratorV2,
        trainer_service: TrainerService,
        model_analyzer: ModelAnalyzer,
        data_manager: DataManager,
        eval_data_manager: EvalDataManager,
        error_pattern_analyzer: ErrorPatternAnalysisService,
        prompt_builder: PromptBuilder,
        label_config: Type[BaseLabelConfig]
    ):
        self.data_generator = data_generator
        self.trainer_service = trainer_service
        self.model_analyzer = model_analyzer
        self.data_manager = data_manager
        self.eval_data_manager = eval_data_manager
        self.error_pattern_analyzer = error_pattern_analyzer
        self.prompt_builder = prompt_builder
        self.label_config = label_config
        self.status = PipelineStatus()
        self.logger = logging.getLogger(__name__)

    async def run(
        self,
        initial_checkpoint_id: str = "11.7",
        max_iterations: int = 20,
        target_f1_per_label: float = 0.7,
        samples_per_action: int = 500,
        iteration_number: Optional[int] = None
    ) -> Dict:
        """
        Run the complete training orchestration pipeline:
        1. Load model from checkpoint (e.g., "11.7")
        2. Evaluate on eval dataset
        3. Analyze errors using LLM
        4. For each data action, generate training data
        5. Continue training from previous checkpoint
        6. Loop until all labels achieve target F1

        Args:
            initial_checkpoint_id: Starting checkpoint (e.g., "11.7")
            max_iterations: Maximum number of iterations to prevent infinite loops
            target_f1_per_label: Target F1 score for all labels (default: 0.7)
            samples_per_action: Number of samples to generate per action (default: 500)
            iteration_number: Evaluation dataset iteration to use (None = latest)

        Returns:
            Dictionary with pipeline results
        """
        if self.status.is_running:
            return {
                "error": "Pipeline is already running",
                "status": "error"
            }

        self.logger.info(f"Starting orchestrator run from checkpoint {initial_checkpoint_id}")

        # Initialize status
        self.status = PipelineStatus(
            is_running=True,
            current_iteration=0,
            total_iterations=max_iterations,
            start_time=time.time(),
            metrics_history=[]
        )

        try:
            # Load human seeds for data generation
            human_seeds = self._load_human_seeds()
            if not human_seeds:
                raise ValueError("No human seed data found. Please ensure .cache/human_seed.json exists.")

            # Get evaluation dataset
            eval_samples = self.eval_data_manager.load(iteration_number)
            if not eval_samples:
                raise ValueError("No evaluation dataset available. Please generate evaluation data first.")

            eval_dataset = self.eval_data_manager.to_datasets(iteration_number)
            open_intent_samples = self.eval_data_manager.load_open_intent(iteration_number)
            eval_iter = iteration_number if iteration_number is not None else self.eval_data_manager.get_latest_item_number()
            self.logger.info(f"Using evaluation dataset iteration: {eval_iter}")
            self.logger.info(f"Loaded {len(eval_dataset)} eval samples and {len(open_intent_samples) if open_intent_samples else 0} open intent samples")

            current_checkpoint_id = initial_checkpoint_id
            iteration = 0

            while iteration < max_iterations:
                iteration += 1
                self.status.current_iteration = iteration
                self.logger.info(f"\n{'='*60}\nIteration {iteration}/{max_iterations}\n{'='*60}")

                # Step 1: Load model and evaluate
                self.logger.info(f"Step 1: Loading checkpoint {current_checkpoint_id} and running evaluation...")
                checkpoint_path = self.trainer_service._file_helper.get_item_path_by_id(current_checkpoint_id)
                if checkpoint_path is None:
                    raise ValueError(f"Checkpoint {current_checkpoint_id} not found")

                # Load model with prob inferencer
                prob_inferencer = ProbModelInference(
                    peft_path=str(checkpoint_path),
                    label_config=self.label_config
                )
                self.model_analyzer.load_model(str(checkpoint_path), inferencer=prob_inferencer)

                # Evaluate model
                eval_start = time.time()
                evaluation_result = self.model_analyzer.analyze_model(
                    evaluation_dataset=eval_dataset,
                    open_intent_samples=open_intent_samples,
                    include_test_cases=False
                )
                eval_time = time.time() - eval_start

                # Log per-label metrics
                self.logger.info("Per-label F1 scores:")
                for label, metrics in evaluation_result.per_label.items():
                    self.logger.info(f"  {label}: F1={metrics.f1_score:.3f}")

                # Step 2: Check convergence - all labels must meet target F1
                labels_below_target = self._get_labels_below_target(evaluation_result.per_label, target_f1_per_label)

                if not labels_below_target:
                    self.logger.info(f"✓ All labels achieved F1 >= {target_f1_per_label}")
                    self.status.termination_reason = f"All labels achieved target F1 >= {target_f1_per_label}"
                    self.status.is_running = False

                    return {
                        "success": True,
                        "status": "completed",
                        "termination_reason": self.status.termination_reason,
                        "iterations_completed": iteration,
                        "final_checkpoint": current_checkpoint_id,
                        "final_metrics": {
                            "overall_accuracy": evaluation_result.overall.accuracy,
                            "macro_f1": evaluation_result.overall.macro_f1,
                            "per_label_f1": {k: v.f1_score for k, v in evaluation_result.per_label.items()}
                        },
                        "config": {
                            "target_f1_per_label": target_f1_per_label,
                            "samples_per_action": samples_per_action,
                            "initial_checkpoint_id": initial_checkpoint_id
                        }
                    }

                self.logger.info(f"Labels below target F1 ({target_f1_per_label}): {labels_below_target}")

                # Step 3: Analyze error patterns using LLM
                self.logger.info("Step 2: Analyzing error patterns with LLM...")
                if not evaluation_result.errors_by_label:
                    self.logger.warning("No errors found to analyze. Stopping pipeline.")
                    break

                label_explanations = self.label_config.get_label_explanation()
                error_analyses = await self.error_pattern_analyzer.analyze_error_patterns(
                    errors_by_label=evaluation_result.errors_by_label,
                    label_explanations=label_explanations
                )

                if not error_analyses:
                    self.logger.warning("No error patterns identified. Stopping pipeline.")
                    break

                self.logger.info(f"Found {len(error_analyses)} error patterns to address")

                # Step 4: Generate data for each error pattern
                self.logger.info(f"Step 3: Generating {samples_per_action} samples per error pattern...")

                total_samples_generated = 0
                versioned_files = []  # Track versioned files for this iteration

                for idx, error_analysis in enumerate(error_analyses):
                    self.logger.info(f"  Pattern {idx+1}/{len(error_analyses)}: {error_analysis.expected_label} -> {error_analysis.predicted_label}")

                    # Get data actions from the analysis
                    if not error_analysis.data_actions:
                        self.logger.warning(f"  No data actions for pattern {idx+1}, skipping")
                        continue

                    for action_idx, data_action in enumerate(error_analysis.data_actions):
                        self.logger.info(f"    Action {action_idx+1}: Generating {samples_per_action} samples for label '{data_action.label}'")

                        # Build prompt using PromptBuilder
                        prompt = self.prompt_builder.build_generate_data_prompt([data_action])

                        if not prompt:
                            self.logger.warning(f"    Failed to build prompt for action {action_idx+1}")
                            continue

                        # Generate data using fix_gen
                        all_results, versioned_file_path, saved_count = await self.data_generator.fix_gen(
                            human_seeds=human_seeds,
                            prompt=prompt,
                            amount=samples_per_action
                        )

                        total_samples_generated += saved_count
                        if versioned_file_path:
                            versioned_files.append(versioned_file_path)

                        self.logger.info(f"    Generated {len(all_results)} samples, {saved_count} saved after deduplication")

                if total_samples_generated == 0:
                    self.logger.warning("No new samples generated. Stopping pipeline.")
                    break

                self.logger.info(f"Total new samples generated: {total_samples_generated}")

                # Step 5: Combine all versioned files into one dataset for continual training
                self.logger.info(f"Step 4: Preparing training dataset from {len(versioned_files)} versioned files...")
                combined_dataset = self.data_manager.combine_versioned_files_to_dataset(versioned_files)

                if len(combined_dataset) == 0:
                    self.logger.warning("No samples in combined dataset. Stopping pipeline.")
                    break

                self.logger.info(f"Combined dataset has {len(combined_dataset)} samples")

                # Step 6: Continue training from current checkpoint on NEW data only
                self.logger.info(f"Step 5: Continuing training from checkpoint {current_checkpoint_id} on NEW data only...")
                train_start = time.time()

                # Continue training with sub-versioning on the combined NEW data
                new_checkpoint_id = await self.trainer_service.continual_train(
                    checkpoint_id=current_checkpoint_id,
                    dataset=combined_dataset,
                    inference_type="prob"
                )

                train_time = time.time() - train_start
                self.logger.info(f"Training completed in {train_time:.2f}s. New checkpoint: {new_checkpoint_id}")

                # Update checkpoint for next iteration
                current_checkpoint_id = new_checkpoint_id

                # Record metrics for this iteration
                iteration_metrics = IterationMetrics(
                    iteration=iteration,
                    accuracy=evaluation_result.overall.accuracy,
                    macro_f1=evaluation_result.overall.macro_f1,
                    unknown_rate=evaluation_result.overall.unknown_rate,
                    total_samples=evaluation_result.overall.total_samples,
                    checkpoint_path=str(checkpoint_path),
                    timestamp=time.time(),
                    training_time=train_time,
                    evaluation_time=eval_time
                )
                self.status.metrics_history.append(iteration_metrics)

            # Pipeline completed max iterations without convergence
            self.status.is_running = False
            self.status.termination_reason = f"Reached max iterations ({max_iterations})"

            return {
                "success": True,
                "status": "completed",
                "termination_reason": self.status.termination_reason,
                "iterations_completed": iteration,
                "final_checkpoint": current_checkpoint_id,
                "note": "Pipeline stopped before all labels reached target F1"
            }

        except Exception as e:
            self.status.is_running = False
            self.status.termination_reason = f"Error: {str(e)}"
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)

            return {
                "success": False,
                "status": "error",
                "error": str(e),
                "iterations_completed": iteration if 'iteration' in locals() else 0
            }

    def _get_labels_below_target(self, per_label_metrics: Dict, target_f1: float) -> List[str]:
        """Get list of labels that are below target F1 score"""
        labels_below = []
        for label, metrics in per_label_metrics.items():
            if metrics.f1_score < target_f1:
                labels_below.append(f"{label} (F1={metrics.f1_score:.3f})")
        return labels_below

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
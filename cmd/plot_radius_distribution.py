#!/usr/bin/env python3
"""
Script to load ADBInference model and plot radius distribution to center point for each label.
Uses the training dataset to analyze the distribution of distances from data points to their respective centers.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset
from collections import defaultdict
import json
from pathlib import Path

from src.payment_classifier.inference.adb_inference import ADBModelInference
from app.core.mixins.numerical_file_access import NumericalFileAccessHelper


class CheckpointManager:
    """Helper class to manage checkpoint loading"""

    def __init__(self):
        self._file_helper = NumericalFileAccessHelper(".checkpoints")

    def get_latest_item_path(self):
        return self._file_helper.get_latest_item_path()


def load_training_dataset(data_path: str) -> Dataset:
    """Load training dataset from JSONL file"""
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    return Dataset.from_list(data)


def calculate_distances_to_centers(model: ADBModelInference, dataset: Dataset):
    """Calculate distances from each data point to their respective label centers"""

    # Ensure ADB data is calculated
    if model.intent_centers is None or model.intent_radii is None:
        print("Calculating ADB centers and radii...")
        model.post_train(dataset)

    # Group data by labels
    label_distances = defaultdict(list)
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            texts = batch['msg']
            labels = batch['label']

            # Get embeddings
            inputs = model.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(model.device)

            outputs = model.peft_model(**inputs, output_hidden_states=True)
            cls_embeddings = outputs.hidden_states[-1][:, 0, :]  # CLS token

            # Calculate distances to respective centers
            for emb, label in zip(cls_embeddings, labels):
                if label in model.intent_centers:
                    center = model.intent_centers[label]
                    distance = torch.norm(emb - center).item()
                    label_distances[label].append(distance)

    return label_distances, model.intent_radii


def plot_radius_distributions(label_distances, intent_radii, id2label):
    """Plot histogram distributions of distances for each label"""

    num_labels = len(label_distances)
    fig, axes = plt.subplots(1, num_labels, figsize=(5*num_labels, 5))

    if num_labels == 1:
        axes = [axes]

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for idx, (label, distances) in enumerate(label_distances.items()):
        ax = axes[idx]

        # Plot histogram
        ax.hist(distances, bins=30, alpha=0.7, color=colors[idx % len(colors)],
                edgecolor='black', density=True)

        # Add vertical line for radius
        radius = intent_radii[label]
        ax.axvline(x=radius, color='red', linestyle='--', linewidth=2,
                  label=f'Radius: {radius:.3f}')

        # Add vertical line for mean distance
        mean_dist = np.mean(distances)
        ax.axvline(x=mean_dist, color='green', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_dist:.3f}')

        # Formatting
        label_name = id2label.get(label, f"Label {label}")
        ax.set_title(f'{label_name}\n(n={len(distances)})')
        ax.set_xlabel('Distance to Center')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    # Configuration
    data_path = ".cache/.data.jsonl"

    # Auto-load latest checkpoint
    checkpoint_manager = CheckpointManager()
    latest_checkpoint_path = checkpoint_manager.get_latest_item_path()

    if latest_checkpoint_path is None:
        print("Error: No checkpoints found in .checkpoints/ directory")
        return

    checkpoint_path = str(latest_checkpoint_path)
    print(f"Loading model from: {checkpoint_path}")
    print(f"Loading dataset from: {data_path}")

    # Load model
    model = ADBModelInference(peft_path=checkpoint_path)

    # Load training dataset
    dataset = load_training_dataset(data_path)
    print(f"Loaded {len(dataset)} training samples")

    # Calculate distances
    print("Calculating distances to centers...")
    label_distances, intent_radii = calculate_distances_to_centers(model, dataset)

    # Print statistics
    print("\nDistance Statistics:")
    print("-" * 50)
    for label, distances in label_distances.items():
        label_name = model.id2label.get(label, f"Label {label}")
        radius = intent_radii[label]
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        max_dist = np.max(distances)
        within_radius = sum(1 for d in distances if d <= radius)

        print(f"{label_name}:")
        print(f"  Samples: {len(distances)}")
        print(f"  Radius: {radius:.4f}")
        print(f"  Mean distance: {mean_dist:.4f}")
        print(f"  Std distance: {std_dist:.4f}")
        print(f"  Max distance: {max_dist:.4f}")
        print(f"  Within radius: {within_radius}/{len(distances)} ({100*within_radius/len(distances):.1f}%)")

    # Create and save plot
    print("Creating distribution plots...")
    fig = plot_radius_distributions(label_distances, intent_radii, model.id2label)

    output_path = "radius_distribution_plot.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_path}")

    # Show plot
    plt.show()


if __name__ == "__main__":
    main()
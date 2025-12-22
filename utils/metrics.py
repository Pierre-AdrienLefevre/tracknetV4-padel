"""TrackNet Metrics for training validation.

Calculates accuracy, precision, recall, F1 score, and detection rate.
"""

import numpy as np
import torch

GROUND_TRUTH_THRESHOLD = 0.1


class TrackNetMetrics:
    """Calculate tracking metrics during validation."""

    def __init__(self, threshold=0.5, tolerance=4):
        """
        Args:
            threshold: Detection threshold for predictions (default 0.5)
            tolerance: Distance tolerance in pixels for TP classification (default 4)
        """
        self.threshold = threshold
        self.tolerance = tolerance
        self.reset()

    def reset(self):
        """Reset all counters."""
        self.tp = 0  # True Positive: detected and within tolerance
        self.tn = 0  # True Negative: no ball predicted, no ball present
        self.fp1 = 0  # False Positive: detected but outside tolerance
        self.fp2 = 0  # False Positive: detected but no ball present
        self.fn = 0  # False Negative: ball present but not detected
        self.total_frames = 0
        self.detected_frames = 0

    def _extract_coordinates(self, heatmap):
        """Extract (x, y) coordinates from heatmap prediction."""
        if heatmap.max() < self.threshold:
            return None
        max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        return (max_pos[1], max_pos[0])  # (x, y)

    def _extract_gt_coordinates(self, heatmap):
        """Extract (x, y) coordinates from ground truth heatmap."""
        if heatmap.max() < GROUND_TRUTH_THRESHOLD:
            return None
        max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        return (max_pos[1], max_pos[0])  # (x, y)

    def _calculate_distance(self, pred_coord, gt_coord):
        """Calculate Euclidean distance between two coordinates."""
        if pred_coord is None or gt_coord is None:
            return float('inf')
        return np.sqrt((pred_coord[0] - gt_coord[0]) ** 2 + (pred_coord[1] - gt_coord[1]) ** 2)

    def _classify(self, pred_coord, gt_coord):
        """Classify prediction into TP, TN, FP1, FP2, or FN."""
        has_pred = pred_coord is not None
        has_gt = gt_coord is not None

        if not has_pred and not has_gt:
            return 'tn'
        elif not has_pred and has_gt:
            return 'fn'
        elif has_pred and not has_gt:
            return 'fp2'
        else:
            distance = self._calculate_distance(pred_coord, gt_coord)
            return 'tp' if distance <= self.tolerance else 'fp1'

    def update(self, outputs, targets):
        """
        Update metrics with a batch of predictions.

        Args:
            outputs: Model predictions [B, 3, H, W] (after sigmoid)
            targets: Ground truth heatmaps [B, 3, H, W]
        """
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        batch_size = outputs.shape[0]
        for b in range(batch_size):
            # Only use center frame (index 1) for metrics
            pred_heatmap = outputs[b, 1]
            gt_heatmap = targets[b, 1]

            pred_coord = self._extract_coordinates(pred_heatmap)
            gt_coord = self._extract_gt_coordinates(gt_heatmap)

            classification = self._classify(pred_coord, gt_coord)

            if classification == 'tp':
                self.tp += 1
            elif classification == 'tn':
                self.tn += 1
            elif classification == 'fp1':
                self.fp1 += 1
            elif classification == 'fp2':
                self.fp2 += 1
            elif classification == 'fn':
                self.fn += 1

            self.total_frames += 1
            if pred_coord is not None:
                self.detected_frames += 1

    def compute(self):
        """
        Compute all metrics.

        Returns:
            dict with accuracy, precision, recall, f1_score, detection_rate
        """
        total_fp = self.fp1 + self.fp2
        total_predictions = self.tp + total_fp
        total_positives = self.tp + self.fn
        total_samples = self.tp + self.tn + total_fp + self.fn

        accuracy = (self.tp + self.tn) / total_samples if total_samples > 0 else 0
        precision = self.tp / total_predictions if total_predictions > 0 else 0
        recall = self.tp / total_positives if total_positives > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        detection_rate = self.detected_frames / self.total_frames if self.total_frames > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'detection_rate': detection_rate,
            'tp': self.tp,
            'tn': self.tn,
            'fp1': self.fp1,
            'fp2': self.fp2,
            'fn': self.fn
        }

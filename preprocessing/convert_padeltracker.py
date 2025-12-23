"""
Convert PadelTracker100 dataset to TrackNet format

PadelTracker100 format (COCO):
- videos/*.mp4 - Video files
- labels/*_ball.json - COCO format annotations with ball bboxes

TrackNet format:
- match{N}/inputs/frame{M}/0.jpg, 1.jpg, 2.jpg... (512x288 RGB)
- match{N}/heatmaps/frame{M}/0.jpg, 1.jpg, 2.jpg... (512x288 grayscale gaussian)

Usage:
    python preprocessing/convert_padeltracker.py --config config.yaml
    python preprocessing/convert_padeltracker.py --source dataset --output dataset/preprocessed/train
"""

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# TrackNet expected dimensions
TARGET_WIDTH = 512
TARGET_HEIGHT = 288

# Video-specific sync corrections for PadelTracker100
# The masculine match (FinalM) has replays that are not in the annotations
VIDEO_SYNC_CONFIG = {
    '2022_BCN_FinalF_1': {
        'replays': [],  # No replays in feminine match
        'max_annotation_id': 45000,  # Stop at 45000 to be safe (last ~900 frames may have issues)
    },
    '2022_BCN_FinalM_1': {
        'replays': [
            {'start': 325, 'end': 389},  # 65 frames of replay
        ],
        'max_annotation_id': 21408,  # Stop at this annotation (only ~40% of video is annotated)
    },
}


def load_config(config_path: str) -> dict:
    """Load preprocessing config from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get('preprocess', {})


def generate_gaussian_heatmap(height: int, width: int, center_x: float, center_y: float, sigma: float = 3.0) -> np.ndarray:
    """
    Generate 2D Gaussian heatmap.

    Args:
        height: Heatmap height
        width: Heatmap width
        center_x: Ball center x coordinate (in heatmap coords)
        center_y: Ball center y coordinate (in heatmap coords)
        sigma: Gaussian standard deviation

    Returns:
        heatmap: [H, W] uint8 image (0-255)
    """
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    heatmap = np.exp(-((X - center_x) ** 2 + (Y - center_y) ** 2) / (2 * sigma ** 2))
    heatmap = (heatmap * 255).astype(np.uint8)

    return heatmap


def generate_empty_heatmap(height: int, width: int) -> np.ndarray:
    """Generate empty (black) heatmap when ball is not visible."""
    return np.zeros((height, width), dtype=np.uint8)


def load_coco_annotations(json_path: str) -> tuple[dict, dict]:
    """
    Load COCO format annotations.

    Returns:
        images_dict: {image_id: {"file_name": str, "width": int, "height": int}}
        annotations_dict: {image_id: {"x": float, "y": float, "w": float, "h": float}}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Build image lookup
    images_dict = {}
    for img in data['images']:
        images_dict[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }

    # Build annotations lookup (category_id=1 is Ball)
    annotations_dict = {}
    for ann in data['annotations']:
        if ann['category_id'] == 1:  # Ball
            image_id = ann['image_id']
            bbox = ann['bbox']  # [x, y, width, height]
            annotations_dict[image_id] = {
                'x': bbox[0],
                'y': bbox[1],
                'w': bbox[2],
                'h': bbox[3]
            }

    return images_dict, annotations_dict


def extract_frames_from_video(video_path: str, output_dir: Path, max_frames: int = None) -> int:
    """
    Extract frames from video file.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        max_frames: Maximum number of frames to extract (None for all)

    Returns:
        Number of frames extracted
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    pbar = tqdm(total=total_frames, desc=f"Extracting {Path(video_path).stem}", leave=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames and frame_count >= max_frames:
            break

        # Resize to TrackNet dimensions
        frame_resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

        # Save frame
        frame_path = output_dir / f"frame_{frame_count:06d}.jpg"
        cv2.imwrite(str(frame_path), frame_resized)

        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    return frame_count


def get_sync_config(video_stem: str) -> dict:
    """Get sync configuration for a video."""
    return VIDEO_SYNC_CONFIG.get(video_stem, {'replays': [], 'max_annotation_id': None})


def is_in_replay(frame_idx: int, replays: list) -> bool:
    """Check if frame is within a replay segment."""
    for replay in replays:
        if replay['start'] <= frame_idx <= replay['end']:
            return True
    return False


def get_cumulative_offset(frame_idx: int, replays: list) -> int:
    """Calculate cumulative offset from all replays before this frame."""
    offset = 0
    for replay in replays:
        if frame_idx > replay['end']:
            offset += replay['end'] - replay['start'] + 1
    return offset


def process_match(
    video_path: str,
    annotations_path: str,
    output_dir: Path,
    match_name: str,
    sigma: float = 3.0,
    frames_per_group: int = 100,
    max_frames: int = None
) -> dict:
    """
    Process a single match: extract frames and generate heatmaps.

    Args:
        video_path: Path to video file
        annotations_path: Path to COCO annotations JSON
        output_dir: Output directory
        match_name: Match identifier (e.g., "match1")
        sigma: Gaussian sigma for heatmaps
        frames_per_group: Number of frames per frame group folder
        max_frames: Maximum frames to process (None for all)

    Returns:
        Statistics dict
    """
    match_dir = output_dir / match_name
    inputs_dir = match_dir / "inputs"
    heatmaps_dir = match_dir / "heatmaps"

    # Load annotations
    print(f"\nLoading annotations from {annotations_path}...")
    images_dict, annotations_dict = load_coco_annotations(annotations_path)

    # Get video info and sync config
    video_stem = Path(video_path).stem
    sync_config = get_sync_config(video_stem)
    replays = sync_config['replays']
    max_annotation_id = sync_config['max_annotation_id']

    cap = cv2.VideoCapture(video_path)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print(f"Video: {orig_width}x{orig_height}, {total_frames} frames, {fps:.2f} FPS")
    print(f"Annotations: {len(images_dict)} images, {len(annotations_dict)} ball positions")

    if replays:
        total_replay_frames = sum(r['end'] - r['start'] + 1 for r in replays)
        print(f"Sync config: {len(replays)} replay(s) ({total_replay_frames} frames to skip)")
    if max_annotation_id:
        print(f"Max annotation ID: {max_annotation_id}")

    if max_frames:
        total_frames = min(total_frames, max_frames)

    # Scale factors for coordinate conversion
    scale_x = TARGET_WIDTH / orig_width
    scale_y = TARGET_HEIGHT / orig_height

    # Process video frame by frame
    cap = cv2.VideoCapture(video_path)

    stats = {
        'total_frames': 0,
        'frames_with_ball': 0,
        'frames_without_ball': 0,
        'frames_skipped_replay': 0
    }

    pbar = tqdm(total=total_frames, desc=f"Processing {match_name}")
    frame_idx = 0
    output_idx = 0  # Separate counter for output frames (excludes replays)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames and frame_idx >= max_frames:
            break

        # Skip replay frames
        if is_in_replay(frame_idx, replays):
            stats['frames_skipped_replay'] += 1
            frame_idx += 1
            pbar.update(1)
            continue

        # Calculate image_id with offset correction
        offset = get_cumulative_offset(frame_idx, replays)
        image_id = frame_idx - offset + 1

        # Stop if we've reached the max annotation ID
        if max_annotation_id and image_id > max_annotation_id:
            print(f"\nReached max annotation ID {max_annotation_id}, stopping.")
            break

        # Determine frame group (based on output_idx, not frame_idx)
        group_idx = output_idx // frames_per_group
        group_name = f"frame{group_idx}"
        local_idx = output_idx % frames_per_group

        # Create directories
        input_group_dir = inputs_dir / group_name
        heatmap_group_dir = heatmaps_dir / group_name
        input_group_dir.mkdir(parents=True, exist_ok=True)
        heatmap_group_dir.mkdir(parents=True, exist_ok=True)

        # Resize and save input frame
        frame_resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        input_path = input_group_dir / f"{local_idx}.jpg"
        cv2.imwrite(str(input_path), frame_resized)

        # Generate heatmap
        if image_id in annotations_dict:
            ann = annotations_dict[image_id]
            # Calculate ball center in original coords
            center_x_orig = ann['x'] + ann['w'] / 2
            center_y_orig = ann['y'] + ann['h'] / 2

            # Scale to target coords
            center_x = center_x_orig * scale_x
            center_y = center_y_orig * scale_y

            heatmap = generate_gaussian_heatmap(
                TARGET_HEIGHT, TARGET_WIDTH, center_x, center_y, sigma
            )
            stats['frames_with_ball'] += 1
        else:
            heatmap = generate_empty_heatmap(TARGET_HEIGHT, TARGET_WIDTH)
            stats['frames_without_ball'] += 1

        # Save heatmap
        heatmap_path = heatmap_group_dir / f"{local_idx}.jpg"
        cv2.imwrite(str(heatmap_path), heatmap)

        stats['total_frames'] += 1
        frame_idx += 1
        output_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    return stats


def find_video_annotation_pairs(source_dir: Path) -> list[tuple[Path, Path]]:
    """
    Find matching video and annotation file pairs.

    Expected structure:
    source_dir/
    ├── video1.mp4
    ├── labels/
    │   └── video1_ball.json

    Returns:
        List of (video_path, annotation_path) tuples
    """
    pairs = []
    labels_dir = source_dir / "labels"

    # Find all MP4 files
    video_files = list(source_dir.glob("*.mp4"))

    for video_path in video_files:
        video_stem = video_path.stem  # e.g., "2022_BCN_FinalF_1"
        ann_path = labels_dir / f"{video_stem}_ball.json"

        if ann_path.exists():
            pairs.append((video_path, ann_path))
            print(f"Found pair: {video_path.name} <-> {ann_path.name}")
        else:
            print(f"Warning: No annotations found for {video_path.name}")

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Convert PadelTracker100 to TrackNet format")
    parser.add_argument("--config", type=str, help="Path to config.yaml")
    parser.add_argument("--source", type=str, help="Source dataset directory (overrides config)")
    parser.add_argument("--output", type=str, help="Output directory (overrides config)")
    parser.add_argument("--sigma", type=float, default=3.0, help="Gaussian sigma for heatmaps")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames per video (for testing)")
    parser.add_argument("--frames-per-group", type=int, default=100, help="Frames per group folder")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    # Load config
    if args.config:
        cfg = load_config(args.config)
        source_dir = Path(args.source or cfg.get('source', 'dataset'))
        output_dir = Path(args.output or cfg.get('output', 'dataset/preprocessed/train'))
        sigma = args.sigma or cfg.get('sigma', 3.0)
        force = args.force or cfg.get('force', False)
    else:
        source_dir = Path(args.source or 'dataset')
        output_dir = Path(args.output or 'dataset/preprocessed/train')
        sigma = args.sigma
        force = args.force

    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Sigma: {sigma}")

    # Check source exists
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        sys.exit(1)

    # Check output
    if output_dir.exists() and not force:
        print(f"Error: Output directory exists: {output_dir}")
        print("Use --force to overwrite")
        sys.exit(1)

    # Find video-annotation pairs
    pairs = find_video_annotation_pairs(source_dir)

    if not pairs:
        print("Error: No video-annotation pairs found")
        sys.exit(1)

    print(f"\nFound {len(pairs)} video-annotation pairs")

    # Process each match
    all_stats = []
    for i, (video_path, ann_path) in enumerate(pairs):
        match_name = f"match{i + 1}"
        print(f"\n{'='*60}")
        print(f"Processing {match_name}: {video_path.name}")
        print(f"{'='*60}")

        stats = process_match(
            str(video_path),
            str(ann_path),
            output_dir,
            match_name,
            sigma=sigma,
            frames_per_group=args.frames_per_group,
            max_frames=args.max_frames
        )
        all_stats.append((match_name, stats))

        print(f"\n{match_name} stats:")
        print(f"  Total frames: {stats['total_frames']}")
        print(f"  With ball: {stats['frames_with_ball']} ({100*stats['frames_with_ball']/stats['total_frames']:.1f}%)")
        print(f"  Without ball: {stats['frames_without_ball']} ({100*stats['frames_without_ball']/stats['total_frames']:.1f}%)")
        if stats.get('frames_skipped_replay', 0) > 0:
            print(f"  Skipped (replay): {stats['frames_skipped_replay']}")

    # Final summary
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print(f"{'='*60}")

    total_frames = sum(s['total_frames'] for _, s in all_stats)
    total_with_ball = sum(s['frames_with_ball'] for _, s in all_stats)

    print(f"Total matches: {len(all_stats)}")
    print(f"Total frames: {total_frames}")
    print(f"Frames with ball: {total_with_ball} ({100*total_with_ball/total_frames:.1f}%)")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
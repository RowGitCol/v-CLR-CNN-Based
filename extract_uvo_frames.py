"""
Extract frames from UVO video dataset for object detection evaluation.

This script:
1. Reads the UVO annotation JSON files
2. Extracts the specific frames referenced in annotations from MP4 videos
3. Saves frames as PNG images in the expected directory structure

Usage:
    python extract_uvo_frames.py

The script expects:
    datasets/uvo_videos_dense/uvo_videos_dense/*.mp4  (video files)
    datasets/UVO_video_val_dense.json  (annotations)

Output:
    datasets/uvo_frames/<video_id>/<frame_number>.png
"""

import json
import cv2
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def extract_frames_from_videos(
    annotation_file: Path, video_dir: Path, output_dir: Path, split: str = "val"
):
    """
    Extract frames referenced in annotations from video files.

    Args:
        annotation_file: Path to UVO annotation JSON
        video_dir: Directory containing .mp4 video files
        output_dir: Directory to save extracted frames
        split: Dataset split name (for logging)
    """
    print(f"Loading annotations from: {annotation_file}")
    with open(annotation_file, "r") as f:
        data = json.load(f)

    # Build mapping: video_id -> video info
    videos = {v["id"]: v for v in data["videos"]}
    print(f"Found {len(videos)} videos in annotations")

    # Find which frames we need from each video
    # The annotations have 'start_idx' which tells us the frame offset
    video_frames = defaultdict(set)

    for ann in data["annotations"]:
        video_id = ann["video_id"]
        start_idx = ann["start_idx"]
        length = ann["length"]

        # Add all frames in this annotation's range
        for i in range(length):
            if (
                ann["segmentations"][i] is not None
            ):  # Only extract frames with annotations
                video_frames[video_id].add(start_idx + i)

    print(f"Need to extract frames from {len(video_frames)} videos")

    # Extract frames
    total_extracted = 0
    failed_videos = []

    for video_id, frame_indices in tqdm(
        video_frames.items(), desc=f"Extracting {split} frames"
    ):
        video_info = videos[video_id]
        # Use 'ytid' field (YouTube ID) as video name
        video_name = video_info.get("ytid") or video_info["file_names"][0].split("/")[0]

        # Find video file
        video_path = video_dir / f"{video_name}.mp4"
        if not video_path.exists():
            failed_videos.append(video_name)
            continue

        # Create output directory for this video
        video_output_dir = output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            failed_videos.append(video_name)
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Extract only the frames we need
        sorted_frames = sorted(frame_indices)
        for frame_idx in sorted_frames:
            if frame_idx >= total_frames:
                continue

            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                output_path = video_output_dir / f"{frame_idx}.png"
                cv2.imwrite(str(output_path), frame)
                total_extracted += 1

        cap.release()

    print(f"\nExtraction complete!")
    print(f"  Total frames extracted: {total_extracted}")
    print(f"  Failed videos: {len(failed_videos)}")
    if failed_videos[:5]:
        print(f"  First few failed: {failed_videos[:5]}")

    return total_extracted, failed_videos


def create_coco_format_annotations(
    uvo_annotation_file: Path, output_file: Path, frames_dir: Path
):
    """
    Convert UVO video annotations to standard COCO image format.
    This creates a single JSON file with image-level annotations.

    Args:
        uvo_annotation_file: Original UVO annotation JSON
        output_file: Output COCO-format JSON path
        frames_dir: Directory containing extracted frames
    """
    print(f"\nConverting annotations to COCO format...")

    with open(uvo_annotation_file, "r") as f:
        data = json.load(f)

    videos = {v["id"]: v for v in data["videos"]}

    coco_output = {
        "info": {
            "description": "UVO Dense Dataset - Converted to COCO format",
            "version": "1.0",
            "year": 2021,
            "contributor": "UVO Team",
            "date_created": "2021",
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "object", "supercategory": "object"}
        ],  # Class-agnostic
    }

    image_id = 0
    ann_id = 0

    for ann in tqdm(data["annotations"], desc="Converting annotations"):
        video_id = ann["video_id"]
        video_info = videos[video_id]
        # Use 'ytid' field (YouTube ID) as video name
        video_name = video_info.get("ytid") or video_info["file_names"][0].split("/")[0]
        start_idx = ann["start_idx"]

        for i, (bbox, segm, area) in enumerate(
            zip(ann["bboxes"], ann["segmentations"], ann["areas"])
        ):
            if bbox is None or segm is None:
                continue

            frame_idx = start_idx + i
            frame_path = frames_dir / video_name / f"{frame_idx}.png"

            if not frame_path.exists():
                continue

            # Create image entry (avoid duplicates)
            file_name = f"{video_name}/{frame_idx}.png"

            # Check if image already exists
            existing_img = None
            for img in coco_output["images"]:
                if img["file_name"] == file_name:
                    existing_img = img
                    break

            if existing_img is None:
                image_id += 1
                img_entry = {
                    "id": image_id,
                    "file_name": file_name,
                    "height": ann["height"],
                    "width": ann["width"],
                }
                coco_output["images"].append(img_entry)
                current_image_id = image_id
            else:
                current_image_id = existing_img["id"]

            # Create annotation entry
            ann_id += 1
            x, y, w, h = bbox
            ann_entry = {
                "id": ann_id,
                "image_id": current_image_id,
                "category_id": 1,  # Class-agnostic
                "bbox": [x, y, w, h],
                "area": area if area else w * h,
                "iscrowd": ann.get("iscrowd", 0),
                "segmentation": segm,
            }
            coco_output["annotations"].append(ann_entry)

    print(f"  Total images: {len(coco_output['images'])}")
    print(f"  Total annotations: {len(coco_output['annotations'])}")

    # Save output
    with open(output_file, "w") as f:
        json.dump(coco_output, f)

    print(f"  Saved to: {output_file}")

    return coco_output


def main():
    # Paths
    base_dir = Path(__file__).parent
    datasets_dir = base_dir / "datasets"

    video_dir = datasets_dir / "uvo_videos_dense" / "uvo_videos_dense"
    frames_dir = datasets_dir / "uvo_frames"

    val_ann_file = datasets_dir / "UVO_video_val_dense.json"
    train_ann_file = datasets_dir / "UVO_video_train_dense.json"

    # Check paths exist
    if not video_dir.exists():
        print(f"ERROR: Video directory not found: {video_dir}")
        return

    # Process validation set
    if val_ann_file.exists():
        print("=" * 60)
        print("Processing VALIDATION set")
        print("=" * 60)

        extract_frames_from_videos(val_ann_file, video_dir, frames_dir, split="val")

        # Convert to COCO format
        coco_val_file = datasets_dir / "uvo_val_coco.json"
        create_coco_format_annotations(val_ann_file, coco_val_file, frames_dir)
    else:
        print(f"Validation annotations not found: {val_ann_file}")

    # Process training set (optional)
    if train_ann_file.exists():
        print("\n" + "=" * 60)
        print("Processing TRAINING set")
        print("=" * 60)

        extract_frames_from_videos(train_ann_file, video_dir, frames_dir, split="train")

        # Convert to COCO format
        coco_train_file = datasets_dir / "uvo_train_coco.json"
        create_coco_format_annotations(train_ann_file, coco_train_file, frames_dir)
    else:
        print(f"\nTraining annotations not found: {train_ann_file}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nExtracted frames are in: {frames_dir}")
    print(f"COCO-format annotations are in: {datasets_dir}")
    print("\nYou can now run the YOLO baseline evaluation notebook!")


if __name__ == "__main__":
    main()

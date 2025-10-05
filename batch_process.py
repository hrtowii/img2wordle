#!/usr/bin/env python3
"""
Batch process images through yellow_optimized.py using subprocess calls and create video from frames.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json
import time
import tempfile
import shutil
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import math


def get_image_files(input_dir):
    """Get list of image files from input directory"""
    input_path = Path(input_dir)
    if not input_path.exists():
        return []

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = []

    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))

    return sorted(image_files)


def split_images_into_chunks(image_files, num_threads):
    """Split image files into chunks for each thread"""
    chunk_size = math.ceil(len(image_files) / num_threads)
    chunks = []

    for i in range(0, len(image_files), chunk_size):
        chunk = image_files[i : i + chunk_size]
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)

    return chunks


def run_yellow_optimized_subprocess(
    image_chunk,
    thread_id,
    wordlist_path,
    min_cols,
    target_word,
    output_dir,
    max_workers_per_image,
    batch_size_per_image,
    screenshot_workers,
):
    """Run yellow_optimized.py as subprocess for a chunk of images"""
    results = []

    for i, image_path in enumerate(image_chunk):
        # Create unique output prefix with proper sequential ordering
        image_stem = Path(image_path).stem
        # Use image filename for consistent ordering across runs
        image_number = int("".join(filter(str.isdigit, image_stem)) or "0")
        global_index = image_number
        output_prefix = f"{global_index:04d}_{image_stem}"

        # Build command
        cmd = [
            sys.executable,
            "yellow_optimized.py",
            str(image_path),
            str(wordlist_path),
            str(min_cols),
        ]

        # Add optional arguments
        if target_word:
            cmd.extend(["--target-word", target_word])

        cmd.extend(
            [
                "--output-dir",
                str(output_dir),
                "--output-prefix",
                output_prefix,
            ]
        )

        if max_workers_per_image:
            cmd.extend(["--max-workers", str(max_workers_per_image)])

        if batch_size_per_image:
            cmd.extend(["--batch-size", str(batch_size_per_image)])

        if screenshot_workers:
            cmd.extend(["--screenshot-workers", str(screenshot_workers)])

        # Note: yellow_optimized.py doesn't support --screenshot-wait
        # Screenshots are taken by default unless --no-screenshot is used

        try:
            # Run subprocess
            print(f"Thread {thread_id}: Processing {image_path.name}...")
            start_time = time.time()

            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout per image
            )

            processing_time = time.time() - start_time

            if result.returncode == 0:
                # Try to parse any JSON output for stats
                stats = {"solved_count": "?", "total_blocks": "?", "from_cache": False}

                # Look for JSON output file
                json_path = Path(output_dir) / f"{output_prefix}.json"
                if json_path.exists():
                    try:
                        with open(json_path) as f:
                            data = json.load(f)
                            stats["solved_count"] = len(
                                [b for b in data.get("blocks", []) if b.get("solved")]
                            )
                            stats["total_blocks"] = len(data.get("blocks", []))
                    except:
                        pass

                # Look for screenshot
                screenshot_path = None
                for ext in [".png", ".jpg"]:
                    potential_screenshot = (
                        Path(output_dir) / f"{output_prefix}_screenshot{ext}"
                    )
                    if potential_screenshot.exists():
                        screenshot_path = potential_screenshot
                        break

                results.append(
                    {
                        "success": True,
                        "image_path": str(image_path),
                        "output_prefix": output_prefix,
                        "html_path": str(Path(output_dir) / f"{output_prefix}.html"),
                        "json_path": str(Path(output_dir) / f"{output_prefix}.json"),
                        "screenshot_path": str(screenshot_path)
                        if screenshot_path
                        else None,
                        "processing_time": processing_time,
                        "thread_id": thread_id,
                        **stats,
                    }
                )

                print(
                    f"Thread {thread_id}: ✓ Completed {image_path.name} "
                    f"({stats['solved_count']}/{stats['total_blocks']} blocks, "
                    f"{processing_time:.1f}s)"
                )
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                results.append(
                    {
                        "success": False,
                        "image_path": str(image_path),
                        "output_prefix": output_prefix,
                        "error": error_msg,
                        "thread_id": thread_id,
                        "processing_time": processing_time,
                    }
                )

                print(f"Thread {thread_id}: ✗ Failed {image_path.name}: {error_msg}")

        except subprocess.TimeoutExpired:
            results.append(
                {
                    "success": False,
                    "image_path": str(image_path),
                    "output_prefix": output_prefix,
                    "error": "Processing timeout (>10 minutes)",
                    "thread_id": thread_id,
                    "processing_time": 600,
                }
            )
            print(f"Thread {thread_id}: ✗ Timeout {image_path.name}")

        except Exception as e:
            results.append(
                {
                    "success": False,
                    "image_path": str(image_path),
                    "output_prefix": output_prefix,
                    "error": str(e),
                    "thread_id": thread_id,
                    "processing_time": 0,
                }
            )
            print(f"Thread {thread_id}: ✗ Exception {image_path.name}: {e}")

    return results


def create_video_with_ffmpeg(
    screenshot_dir,
    output_video_path,
    fps=2.0,
    duration_per_frame=None,
    video_quality="high",
):
    """Create video from screenshot frames using ffmpeg"""
    screenshot_files = []

    # Collect all screenshot files
    for ext in [".png", ".jpg", ".jpeg"]:
        screenshot_files.extend(screenshot_dir.glob(f"*{ext}"))

    if not screenshot_files:
        print("No screenshot files found for video creation")
        return False

    # Sort by filename to ensure proper order
    screenshot_files = sorted(screenshot_files, key=lambda x: x.name)
    print(f"Found {len(screenshot_files)} screenshot files for video")

    # Create temporary directory for symlinks with sequential names
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create symlinks with sequential names for ffmpeg
        for i, screenshot_path in enumerate(screenshot_files):
            ext = screenshot_path.suffix
            symlink_path = temp_path / f"frame_{i:06d}{ext}"
            symlink_path.symlink_to(screenshot_path.absolute())

        # Determine frame rate and input pattern
        if duration_per_frame:
            actual_fps = 1.0 / duration_per_frame
        else:
            actual_fps = fps

        # Build ffmpeg command
        input_pattern = temp_path / "frame_%06d.png"
        # Try .png first, if no .png files exist, try other extensions
        if not any(f.suffix == ".png" for f in screenshot_files):
            if any(f.suffix == ".jpg" for f in screenshot_files):
                input_pattern = temp_path / "frame_%06d.jpg"
            elif any(f.suffix == ".jpeg" for f in screenshot_files):
                input_pattern = temp_path / "frame_%06d.jpeg"

        # Set quality parameters
        if video_quality == "high":
            quality_params = ["-crf", "18", "-preset", "slow"]
        elif video_quality == "medium":
            quality_params = ["-crf", "23", "-preset", "medium"]
        else:  # low
            quality_params = ["-crf", "28", "-preset", "fast"]

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-framerate",
            str(actual_fps),
            "-i",
            str(input_pattern),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            *quality_params,
            "-movflags",
            "+faststart",  # Optimize for web streaming
            str(output_video_path),
        ]

        try:
            print(
                f"Creating video with {len(screenshot_files)} frames at {actual_fps} fps..."
            )
            print(f"Command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0:
                print(f"✓ Video created successfully: {output_video_path}")
                return True
            else:
                print(f"✗ FFmpeg failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("✗ Video creation timed out")
            return False
        except FileNotFoundError:
            print("✗ FFmpeg not found. Please install ffmpeg to create videos.")
            return False
        except Exception as e:
            print(f"✗ Error creating video: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch process images through yellow_optimized.py subprocesses and create video"
    )
    parser.add_argument("input_dir", help="Directory containing input images")
    parser.add_argument("wordlist", help="Path to wordlist file")
    parser.add_argument("min_cols", type=int, help="Minimum number of columns")
    parser.add_argument(
        "--target-word",
        default="APPLE",
        help="Target word for all images (default: APPLE)",
    )
    parser.add_argument("--output-dir", default="batch_output", help="Output directory")
    parser.add_argument(
        "--threads", type=int, default=4, help="Number of parallel threads/processes"
    )
    parser.add_argument(
        "--max-workers-per-image",
        type=int,
        help="Maximum workers per image (default: all CPU cores for yellow_optimized.py)",
    )
    parser.add_argument(
        "--batch-size-per-image",
        type=int,
        help="Batch size per image for optimized processing (default: adaptive)",
    )
    parser.add_argument(
        "--screenshot-workers",
        type=int,
        default=2,
        help="Number of parallel screenshot workers per image (default: 2)",
    )
    parser.add_argument(
        "--video-fps", type=float, default=2.0, help="Video framerate (fps)"
    )
    parser.add_argument(
        "--video-duration-per-frame",
        type=float,
        help="Duration per frame in seconds (overrides fps)",
    )
    parser.add_argument(
        "--video-output", default="wordle_art_video.mp4", help="Output video filename"
    )
    parser.add_argument(
        "--video-quality",
        choices=["low", "medium", "high"],
        default="high",
        help="Video quality (default: high)",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep intermediate files (HTML, JSON, screenshots)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video creation, only process images",
    )

    args = parser.parse_args()

    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Get input images
    image_files = get_image_files(args.input_dir)
    if not image_files:
        print(f"No image files found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(image_files)} images to process")
    print(f"Using {args.threads} parallel threads")

    # Split images into chunks for each thread
    image_chunks = split_images_into_chunks(image_files, args.threads)
    print(
        f"Split into {len(image_chunks)} chunks: {[len(chunk) for chunk in image_chunks]}"
    )

    # Process images using ThreadPoolExecutor
    print("Starting parallel processing...")
    start_time = time.time()
    all_results = []

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # Submit all chunks
        future_to_thread = {}
        for thread_id, chunk in enumerate(image_chunks):
            future = executor.submit(
                run_yellow_optimized_subprocess,
                chunk,
                thread_id,
                args.wordlist,
                args.min_cols,
                args.target_word,
                output_dir,
                args.max_workers_per_image,
                args.batch_size_per_image,
                args.screenshot_workers,
            )
            future_to_thread[future] = thread_id

        # Collect results as they complete
        for future in as_completed(future_to_thread):
            thread_id = future_to_thread[future]
            try:
                thread_results = future.result()
                all_results.extend(thread_results)
                print(
                    f"Thread {thread_id} completed with {len(thread_results)} results"
                )
            except Exception as e:
                print(f"Thread {thread_id} failed with exception: {e}")

    total_time = time.time() - start_time

    # Summary
    successful_results = [r for r in all_results if r["success"]]
    failed_results = [r for r in all_results if not r["success"]]

    print(f"\n=== BATCH PROCESSING COMPLETE ===")
    print(f"Total time: {total_time:.1f}s")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")

    if failed_results:
        print("\nFailed images:")
        for result in failed_results:
            print(f"  - {Path(result['image_path']).name}: {result['error']}")

    # Create video if requested and we have successful results
    if not args.no_video and successful_results:
        print(f"\n=== CREATING VIDEO ===")

        # Find all screenshots and sort them properly
        screenshot_results = []
        for result in successful_results:
            if result.get("screenshot_path"):
                screenshot_path = Path(result["screenshot_path"])
                if screenshot_path.exists():
                    screenshot_results.append(result)

        # Sort by output_prefix to ensure correct order
        screenshot_results.sort(key=lambda x: x["output_prefix"])

        if screenshot_results:
            video_success = create_video_with_ffmpeg(
                output_dir,  # Screenshots are in the output directory
                args.video_output,
                args.video_fps,
                args.video_duration_per_frame,
                args.video_quality,
            )

            if video_success:
                print(f"Video saved as: {args.video_output}")
            else:
                print("Video creation failed")
        else:
            print("No screenshots found for video creation")

    # Cleanup intermediates if requested
    if not args.keep_intermediates:
        print("\nCleaning up intermediate files...")
        for result in successful_results:
            # Remove HTML and JSON files, keep screenshots for video
            for path_key in ["html_path", "json_path"]:
                if result.get(path_key):
                    path = Path(result[path_key])
                    if path.exists():
                        path.unlink()
        print("Intermediate files cleaned up")

    print(f"\nBatch processing complete! Output in: {output_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

import make_scathach_wallpaper_two_source_rife_interpolated as base
import make_scathach_wallpaper_two_source_rife_interpolated_v2 as v2
import make_scathach_wallpaper_two_source_rife_interpolated_v4 as v4
import make_scathach_wallpaper_two_source_rife_interpolated_v6 as v6


ROOT = base.ROOT
OUT_DIR = base.OUT_DIR
PROJECT_DIR = base.PROJECT_DIR
REVIEW_DIR = OUT_DIR / "two_source_rife_interpolated_v8_source_locked_review"
ALL_FRAME_REVIEW_DIR = OUT_DIR / "v8_source_locked_all_frame_review"
OUTPUT_VIDEO = OUT_DIR / "scathach_two_source_rife_interpolated_2160p60_4s_v8_source_locked.mp4"
PROJECT_VIDEO = PROJECT_DIR / OUTPUT_VIDEO.name

FPS = v6.FPS
TARGET_FRAMES = v6.TARGET_FRAMES
OUT_W = v6.OUT_W
OUT_H = v6.OUT_H
CRF = v6.CRF
PRESET = v6.PRESET


def luma_values(source: np.ndarray) -> np.ndarray:
    return source[:, :, 0] * 0.299 + source[:, :, 1] * 0.587 + source[:, :, 2] * 0.114


def build_source_lock_mask(size: tuple[int, int] = (OUT_W, OUT_H)) -> np.ndarray:
    chest = v4.ellipse_mask(size, (510, 170, 1425, 735), 20.0)
    abdomen = v4.ellipse_mask(size, (585, 470, 1345, 1080), 22.0)
    lap = v4.ellipse_mask(size, (435, 690, 1515, 1080), 22.0)
    weapon = v4.rectangle_mask(size, (0, 540, 1785, 925), 10.0)
    left_hair = v4.rectangle_mask(size, (330, 300, 820, 1080), 18.0) * 0.42
    right_hair = v4.rectangle_mask(size, (1110, 345, 1675, 1080), 18.0) * 0.42
    face_guard = v4.ellipse_mask(size, (680, 0, 1280, 430), 24.0)
    far_right_guard = v4.rectangle_mask(size, (1510, 0, 1920, 1080), 20.0)

    mask = np.maximum.reduce([chest, abdomen, lap, weapon, left_hair, right_hair])
    mask = mask * (1.0 - face_guard * 0.98)
    mask = mask * (1.0 - far_right_guard * 0.88)
    image = Image.fromarray(np.clip(mask * 255, 0, 255).astype(np.uint8), "L")
    image = image.filter(ImageFilter.GaussianBlur(0.55 * size[0] / OUT_W))
    return np.asarray(image, dtype=np.float32) / 255.0


def locked_reference(source_a: np.ndarray, source_b: np.ndarray, alpha: float) -> np.ndarray:
    peak = np.sin(np.pi * np.clip(alpha, 0.0, 1.0))
    b_weight = 0.16 * peak * peak
    return source_a * (1.0 - b_weight) + source_b * b_weight


def apply_source_locked_region(
    frame: np.ndarray,
    source_a: np.ndarray,
    source_b: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    strength: float = 0.96,
) -> np.ndarray:
    reference = locked_reference(source_a, source_b, alpha)
    frame_image = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), "RGB")
    ref_image = Image.fromarray(np.clip(reference, 0, 255).astype(np.uint8), "RGB")
    blur_radius = max(4.0, 18.0 * frame.shape[1] / OUT_W)
    frame_wide = np.asarray(frame_image.filter(ImageFilter.GaussianBlur(blur_radius)), dtype=np.float32)
    ref_wide = np.asarray(ref_image.filter(ImageFilter.GaussianBlur(blur_radius)), dtype=np.float32)
    coarse_light = (frame_wide - ref_wide) * 0.16
    locked = np.clip(reference + coarse_light, 0, 255)
    gate = np.clip(mask[:, :, None] * strength, 0.0, 0.98)
    return np.clip(frame * (1.0 - gate) + locked * gate, 0, 255)


def prepared_context(overwrite_sources: bool = False) -> tuple:
    context = v6.prepared_context(overwrite_sources=overwrite_sources)
    source_lock_mask_4k = build_source_lock_mask((OUT_W, OUT_H))
    return (*context, source_lock_mask_4k)


def render_frame_4k(
    motion_path: Path,
    index: int,
    source_a_1080: np.ndarray,
    source_b_1080: np.ndarray,
    source_a_4k: np.ndarray,
    source_b_4k: np.ndarray,
    high_a_4k: np.ndarray,
    high_b_4k: np.ndarray,
    detail_mask_4k: np.ndarray,
    anchor_mask_4k: np.ndarray,
    flower_mask_4k: np.ndarray,
    highpass_cleanup_mask_4k: np.ndarray,
    source_lock_mask_4k: np.ndarray,
    motion_frames: list[Path],
) -> np.ndarray:
    frame = v4.upscaled_v2_frame(motion_path, index, source_a_1080, source_b_1080, motion_frames)
    alpha = v2.timeline_alpha(index, TARGET_FRAMES)
    frame = v4.apply_4k_detail_plate(
        frame,
        high_a_4k,
        high_b_4k,
        detail_mask_4k,
        alpha=alpha,
        amount=0.76,
        anchor_mask=anchor_mask_4k,
    )
    frame = v6.apply_highpass_deghost(frame, source_a_4k, source_b_4k, highpass_cleanup_mask_4k, alpha, strength=0.94)
    frame = apply_source_locked_region(frame, source_a_4k, source_b_4k, source_lock_mask_4k, alpha, strength=0.97)
    return v4.apply_flower_lighting(frame, flower_mask_4k, index, TARGET_FRAMES)


def render_context_frame(context: tuple, motion_frames: list[Path], index: int) -> np.ndarray:
    return render_frame_4k(motion_frames[index], index, *context, motion_frames)


def write_review_artifacts(overwrite_sources: bool = False, review_dir: Path = REVIEW_DIR) -> list[Path]:
    v4.ensure_motion_frames(overwrite_sources=overwrite_sources, overwrite_rife=False)
    motion_frames = sorted(v2.MOTION_FRAME_DIR.glob("*.png"))
    context = prepared_context(overwrite_sources=overwrite_sources)
    review_dir.mkdir(parents=True, exist_ok=True)

    indices = sorted({round(i * (TARGET_FRAMES - 1) / 8) for i in range(9)})
    frame_paths: list[Path] = []
    for index in indices:
        frame = render_context_frame(context, motion_frames, index).astype(np.uint8)
        out = review_dir / f"rife_interpolated_v8_sample_{index:06d}.jpg"
        Image.fromarray(frame, "RGB").save(out, quality=94)
        frame_paths.append(out)

    outputs: list[Path] = list(frame_paths)
    overview = review_dir / "rife_interpolated_v8_overview.jpg"
    v6.make_contact_sheet(frame_paths, overview)
    outputs.append(overview)
    torso = review_dir / "rife_interpolated_v8_torso_weapon_crops.jpg"
    v6.make_crop_sheet(frame_paths, (1160, 430, 2550, 1780), torso)
    outputs.append(torso)
    hair = review_dir / "rife_interpolated_v8_right_hair_field_crops.jpg"
    v6.make_crop_sheet(frame_paths, (1500, 500, 3840, 2040), hair)
    outputs.append(hair)
    loop = review_dir / "rife_interpolated_v8_loop_check.jpg"
    v6.write_loop_check(Image.open(frame_paths[0]), Image.open(frame_paths[-1]), loop)
    outputs.append(loop)
    preview = review_dir / "rife_interpolated_v8_preview.jpg"
    shutil.copy2(frame_paths[0], preview)
    outputs.append(preview)
    return outputs


def write_all_frame_review(overwrite_sources: bool = False, review_dir: Path = ALL_FRAME_REVIEW_DIR) -> list[Path]:
    v4.ensure_motion_frames(overwrite_sources=overwrite_sources, overwrite_rife=False)
    motion_frames = sorted(v2.MOTION_FRAME_DIR.glob("*.png"))
    context = prepared_context(overwrite_sources=overwrite_sources)
    review_dir.mkdir(parents=True, exist_ok=True)
    for existing in review_dir.glob("*.jpg"):
        existing.unlink()

    outputs: list[Path] = []
    torso_box = (1160, 430, 2550, 1780)
    hair_box = (1500, 500, 3840, 2040)
    for index in range(TARGET_FRAMES):
        frame = render_context_frame(context, motion_frames, index).astype(np.uint8)
        image = Image.fromarray(frame, "RGB")
        torso = review_dir / f"torso_weapon_{index:03d}.jpg"
        image.crop(torso_box).resize((420, 408), Image.Resampling.LANCZOS).save(torso, quality=92)
        hair = review_dir / f"right_hair_field_{index:03d}.jpg"
        image.crop(hair_box).resize((420, 276), Image.Resampling.LANCZOS).save(hair, quality=92)
        outputs.extend([torso, hair])
        if (index + 1) % 20 == 0 or index == TARGET_FRAMES - 1:
            print(f"reviewed v8 crop frame {index + 1}/{TARGET_FRAMES}", flush=True)
    return outputs


def build_ffmpeg_command(output: Path) -> list[str]:
    return [
        str(base.find_ffmpeg()),
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{OUT_W}x{OUT_H}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-frames:v",
        str(TARGET_FRAMES),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        PRESET,
        "-crf",
        CRF,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


def encode_video(overwrite_sources: bool = False, output: Path = OUTPUT_VIDEO) -> Path:
    v4.ensure_motion_frames(overwrite_sources=overwrite_sources, overwrite_rife=False)
    motion_frames = sorted(v2.MOTION_FRAME_DIR.glob("*.png"))
    if len(motion_frames) < TARGET_FRAMES:
        raise RuntimeError(f"motion frame count {len(motion_frames)} < {TARGET_FRAMES}")
    context = prepared_context(overwrite_sources=overwrite_sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_ffmpeg_command(output)
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for index, motion_path in enumerate(motion_frames[:TARGET_FRAMES]):
            frame = render_frame_4k(motion_path, index, *context, motion_frames).astype(np.uint8)
            process.stdin.write(frame.tobytes())
            if (index + 1) % 20 == 0 or index == TARGET_FRAMES - 1:
                print(f"streamed v8 4k frame {index + 1}/{TARGET_FRAMES}", flush=True)
    finally:
        if process.stdin is not None:
            process.stdin.close()
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, command)
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, PROJECT_VIDEO)
    return output


def write_project(video_path: Path, preview_path: Path = REVIEW_DIR / "rife_interpolated_v8_preview.jpg") -> None:
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    project = {
        "file": PROJECT_VIDEO.name,
        "title": "Scathach two-source source-locked 4K RIFE loop",
        "type": "video",
        "visibility": "private",
        "description": "4K 60 fps two-source loop with source-locked body/weapon reconstruction to avoid interpolation trails.",
        "preview": "preview.jpg",
    }
    (PROJECT_DIR / "project.json").write_text(json.dumps(project, indent=2), encoding="utf-8")
    if preview_path.exists():
        shutil.copy2(preview_path, PROJECT_DIR / "preview.jpg")
    elif video_path.exists():
        first_review = write_review_artifacts(review_dir=REVIEW_DIR)[0]
        shutil.copy2(first_review, PROJECT_DIR / "preview.jpg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["review", "all-frame-review", "video"], default="review")
    parser.add_argument("--overwrite-sources", action="store_true")
    parser.add_argument("--no-project", action="store_true")
    args = parser.parse_args()

    if args.mode == "review":
        review = write_review_artifacts(overwrite_sources=args.overwrite_sources, review_dir=REVIEW_DIR)
        print(f"review artifacts: {REVIEW_DIR} ({len(review)})")
    elif args.mode == "all-frame-review":
        outputs = write_all_frame_review(overwrite_sources=args.overwrite_sources, review_dir=ALL_FRAME_REVIEW_DIR)
        print(f"all-frame review artifacts: {ALL_FRAME_REVIEW_DIR} ({len(outputs)})")
    else:
        video = encode_video(overwrite_sources=args.overwrite_sources, output=OUTPUT_VIDEO)
        if not args.no_project:
            preview = REVIEW_DIR / "rife_interpolated_v8_preview.jpg"
            if not preview.exists():
                write_review_artifacts(overwrite_sources=args.overwrite_sources, review_dir=REVIEW_DIR)
            write_project(video, preview)
        print(f"video: {video}")


if __name__ == "__main__":
    main()

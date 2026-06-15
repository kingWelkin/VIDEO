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


ROOT = base.ROOT
OUT_DIR = base.OUT_DIR
PROJECT_DIR = base.PROJECT_DIR
REVIEW_DIR = OUT_DIR / "two_source_rife_interpolated_v3_review"
OUTPUT_VIDEO = OUT_DIR / "scathach_two_source_rife_interpolated_2160p60_4s_v3.mp4"
PROJECT_VIDEO = PROJECT_DIR / OUTPUT_VIDEO.name
EFFECT_FRAME_DIR = v2.WORK_DIR / "effect_frames_v3_1080p60"

FPS = v2.FPS
TARGET_FRAMES = v2.TARGET_FRAMES
OUT_W = v2.OUT_W
OUT_H = v2.OUT_H
CRF = v2.CRF
PRESET = v2.PRESET


def build_problem_detail_mask(size: tuple[int, int] = (base.KEY_W, base.KEY_H)) -> np.ndarray:
    right_hair = v2.rectangle_mask(size, (900, 250, 1920, 940), 22.0)
    lower_flowers = v2.rectangle_mask(size, (600, 500, 1920, 1080), 18.0)
    face_guard = v2.ellipse_mask(size, (720, 30, 1240, 420), 22.0)
    torso_guard = v2.ellipse_mask(size, (560, 390, 1080, 1080), 26.0)
    mask = np.maximum(right_hair, lower_flowers)
    mask = mask * (1.0 - face_guard * 0.96)
    mask = mask * (1.0 - torso_guard * 0.45)
    return np.clip(mask, 0.0, 1.0)


def enhance_problem_detail(frame: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> np.ndarray:
    frame_image = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), "RGB")
    reference_image = Image.fromarray(np.clip(reference, 0, 255).astype(np.uint8), "RGB")
    frame_soft = np.asarray(frame_image.filter(ImageFilter.GaussianBlur(0.75)), dtype=np.float32)
    reference_soft = np.asarray(reference_image.filter(ImageFilter.GaussianBlur(0.85)), dtype=np.float32)
    current_high = frame - frame_soft
    reference_high = reference - reference_soft
    gate = np.clip(mask[:, :, None], 0.0, 1.0)
    boosted = frame + current_high * gate * 0.28 + reference_high * gate * 0.62
    return np.clip(boosted, 0, 255)


def postprocess_v3_frame(
    frame: np.ndarray,
    index: int,
    source_a: np.ndarray,
    source_b: np.ndarray,
    previous: np.ndarray | None = None,
    following: np.ndarray | None = None,
) -> np.ndarray:
    result = v2.postprocess_v2_frame(frame, index, source_a, source_b, previous, following)
    problem_mask = build_problem_detail_mask((result.shape[1], result.shape[0]))
    reference = v2.reference_for_detail(source_a, source_b, index)
    result = enhance_problem_detail(result, reference, problem_mask)
    result = v2.restore_local_detail(result, reference, problem_mask, amount=0.22)
    return np.clip(result, 0, 255)


def ensure_motion_frames(overwrite_sources: bool = False, overwrite_rife: bool = False) -> None:
    motion_frames = sorted(v2.MOTION_FRAME_DIR.glob("*.png")) if v2.MOTION_FRAME_DIR.exists() else []
    if len(motion_frames) >= TARGET_FRAMES and not overwrite_rife:
        return
    v2.write_segment_keyframes(overwrite_sources=overwrite_sources)
    v2.run_rife_segment(v2.KEYFRAME_AB_DIR, v2.RIFE_AB_DIR, overwrite=overwrite_rife)
    v2.run_rife_segment(v2.KEYFRAME_BA_DIR, v2.RIFE_BA_DIR, overwrite=overwrite_rife)
    v2.combine_motion_segments(overwrite=True)


def write_effect_frames(overwrite: bool = False, overwrite_sources: bool = False) -> list[Path]:
    motion_frames = sorted(v2.MOTION_FRAME_DIR.glob("*.png"))
    if len(motion_frames) < TARGET_FRAMES:
        raise RuntimeError(f"motion frame count {len(motion_frames)} < {TARGET_FRAMES}")
    if overwrite:
        base.safe_reset_dir(EFFECT_FRAME_DIR, v2.WORK_DIR)
    else:
        EFFECT_FRAME_DIR.mkdir(parents=True, exist_ok=True)

    source_a, source_b = v2.load_prepared_sources(overwrite_sources=overwrite_sources)
    outputs: list[Path] = []
    for index, path in enumerate(motion_frames[:TARGET_FRAMES]):
        out_path = EFFECT_FRAME_DIR / f"{index:08d}.png"
        if out_path.exists() and not overwrite:
            outputs.append(out_path)
            continue
        current = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        previous = np.asarray(Image.open(motion_frames[max(0, index - 1)]).convert("RGB"), dtype=np.float32)
        following = np.asarray(Image.open(motion_frames[min(TARGET_FRAMES - 1, index + 1)]).convert("RGB"), dtype=np.float32)
        processed = postprocess_v3_frame(current, index, source_a, source_b, previous, following)
        Image.fromarray(processed.astype(np.uint8), "RGB").save(out_path, compress_level=1)
        outputs.append(out_path)
        if index % 20 == 0:
            print(f"postprocessed v3 frame {index + 1}/{TARGET_FRAMES}", flush=True)
    return outputs


def build_ffmpeg_command(output: Path, input_dir: Path = EFFECT_FRAME_DIR) -> list[str]:
    return [
        base.find_ffmpeg(),
        "-y",
        "-framerate",
        str(FPS),
        "-start_number",
        "0",
        "-i",
        str(input_dir / "%08d.png"),
        "-frames:v",
        str(TARGET_FRAMES),
        "-vf",
        f"scale={OUT_W}:{OUT_H}:flags=lanczos,unsharp=5:5:0.22:3:3:0.06,format=yuv420p",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        PRESET,
        "-tune",
        "animation",
        "-crf",
        CRF,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


def encode_video(output: Path = OUTPUT_VIDEO) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(build_ffmpeg_command(output, EFFECT_FRAME_DIR), check=True)
    return output


def make_contact_sheet(paths: list[Path], out_path: Path, thumb_size: tuple[int, int] = (320, 180)) -> None:
    thumbs: list[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_size[0], thumb_size[1] + 24), (18, 18, 22))
        canvas.paste(image, ((thumb_size[0] - image.width) // 2, 0))
        ImageDraw.Draw(canvas).text((8, thumb_size[1] + 5), path.stem, fill=(235, 235, 235))
        thumbs.append(canvas)
    sheet = Image.new("RGB", (thumbs[0].width * len(thumbs), thumbs[0].height), (18, 18, 22))
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, (index * thumbs[0].width, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=94)


def make_crop_sheet(paths: list[Path], crop_box: tuple[int, int, int, int], out_path: Path) -> None:
    crops: list[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB").crop(crop_box)
        image.thumbnail((360, 252), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (360, 276), (18, 18, 22))
        canvas.paste(image, ((360 - image.width) // 2, 0))
        ImageDraw.Draw(canvas).text((8, 257), path.stem, fill=(235, 235, 235))
        crops.append(canvas)
    sheet = Image.new("RGB", (crops[0].width * len(crops), crops[0].height), (18, 18, 22))
    for index, crop in enumerate(crops):
        sheet.paste(crop, (index * crops[0].width, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=95)


def write_loop_check(first: Image.Image, last: Image.Image, out_path: Path) -> None:
    diff = ImageChops.difference(first.convert("RGB"), last.convert("RGB")).convert("L")
    heat = ImageOps.colorize(ImageOps.autocontrast(diff), black="#101018", white="#ff4058", mid="#6837c7")
    images = [first.convert("RGB"), last.convert("RGB"), heat]
    thumbs: list[Image.Image] = []
    for image in images:
        image.thumbnail((360, 202), Image.Resampling.LANCZOS)
        thumbs.append(image)
    sheet = Image.new("RGB", (360 * 3, 226), (18, 18, 22))
    for index, (label, image) in enumerate(zip(("first", "last", "difference"), thumbs)):
        x = index * 360
        sheet.paste(image, (x + (360 - image.width) // 2, 0))
        ImageDraw.Draw(sheet).text((x + 8, 207), label, fill=(235, 235, 235))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=94)


def write_review_artifacts(review_dir: Path = REVIEW_DIR) -> list[Path]:
    review_dir.mkdir(parents=True, exist_ok=True)
    indices = sorted({round(i * (TARGET_FRAMES - 1) / 8) for i in range(9)})
    frame_paths: list[Path] = []
    for index in indices:
        src = EFFECT_FRAME_DIR / f"{index:08d}.png"
        if not src.exists():
            continue
        out = review_dir / f"rife_interpolated_v3_sample_{index:06d}.jpg"
        Image.open(src).convert("RGB").save(out, quality=95)
        frame_paths.append(out)
    outputs: list[Path] = list(frame_paths)
    if frame_paths:
        overview = review_dir / "rife_interpolated_v3_overview.jpg"
        make_contact_sheet(frame_paths, overview)
        outputs.append(overview)
        problem = review_dir / "rife_interpolated_v3_problem_area_crops.jpg"
        make_crop_sheet(frame_paths, (600, 500, 1920, 1080), problem)
        outputs.append(problem)
        right_hair = review_dir / "rife_interpolated_v3_right_hair_crops.jpg"
        make_crop_sheet(frame_paths, (900, 250, 1920, 940), right_hair)
        outputs.append(right_hair)
        loop = review_dir / "rife_interpolated_v3_loop_check.jpg"
        write_loop_check(Image.open(frame_paths[0]), Image.open(frame_paths[-1]), loop)
        outputs.append(loop)
        preview = review_dir / "rife_interpolated_v3_preview.jpg"
        shutil.copy2(frame_paths[0], preview)
        outputs.append(preview)
    return outputs


def write_project(video_path: Path, preview_path: Path = REVIEW_DIR / "rife_interpolated_v3_preview.jpg") -> None:
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_path, PROJECT_VIDEO)
    if preview_path.exists():
        shutil.copy2(preview_path, PROJECT_DIR / "preview.jpg")
    project = {
        "title": "Scathach Two Source RIFE Interpolated Loop V3",
        "description": "Looping 2160p60 4s wallpaper using v2 trimmed motion plus stronger lower-right hair and flower-field detail restoration.",
        "type": "video",
        "file": PROJECT_VIDEO.name,
        "preview": "preview.jpg",
        "tags": ["Anime"],
        "visibility": "private",
    }
    (PROJECT_DIR / "project.json").write_text(json.dumps(project, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_pipeline(overwrite_sources: bool = False, overwrite_rife: bool = False, overwrite_effects: bool = False) -> None:
    ensure_motion_frames(overwrite_sources=overwrite_sources, overwrite_rife=overwrite_rife)
    write_effect_frames(overwrite=overwrite_effects, overwrite_sources=overwrite_sources)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["effects", "review", "video", "all"], default="all")
    parser.add_argument("--overwrite-sources", action="store_true")
    parser.add_argument("--overwrite-rife", action="store_true")
    parser.add_argument("--overwrite-effects", action="store_true")
    parser.add_argument("--no-project", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        overwrite_sources=args.overwrite_sources,
        overwrite_rife=args.overwrite_rife,
        overwrite_effects=args.overwrite_effects,
    )
    print(f"effect frames: {EFFECT_FRAME_DIR}")
    if args.mode in {"review", "all"}:
        review = write_review_artifacts(REVIEW_DIR)
        print(f"review artifacts: {REVIEW_DIR} ({len(review)})")
    if args.mode in {"video", "all"}:
        video = encode_video(OUTPUT_VIDEO)
        if not args.no_project:
            preview = REVIEW_DIR / "rife_interpolated_v3_preview.jpg"
            if not preview.exists():
                write_review_artifacts(REVIEW_DIR)
            write_project(video, preview)
        print(video.resolve())


if __name__ == "__main__":
    main()

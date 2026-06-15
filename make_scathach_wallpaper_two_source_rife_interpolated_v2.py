from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

import make_scathach_wallpaper_two_source_rife_interpolated as base


ROOT = base.ROOT
OUT_DIR = base.OUT_DIR
PROJECT_DIR = base.PROJECT_DIR
SOURCE_DIR = base.SOURCE_DIR
REVIEW_DIR = OUT_DIR / "two_source_rife_interpolated_v2_review"
OUTPUT_VIDEO = OUT_DIR / "scathach_two_source_rife_interpolated_2160p60_4s_v2.mp4"
PROJECT_VIDEO = PROJECT_DIR / OUTPUT_VIDEO.name

TOOLS_DIR = base.TOOLS_DIR
RIFE_EXE = base.RIFE_EXE
RIFE_MODEL = base.RIFE_MODEL
WORK_DIR = TOOLS_DIR / "scathach_two_source_rife_interpolated_v2_work"
KEYFRAME_AB_DIR = WORK_DIR / "keyframes_ab_1080p"
KEYFRAME_BA_DIR = WORK_DIR / "keyframes_ba_1080p"
RIFE_AB_DIR = WORK_DIR / "rife_ab_raw"
RIFE_BA_DIR = WORK_DIR / "rife_ba_raw"
MOTION_FRAME_DIR = WORK_DIR / "motion_frames_1080p60"
EFFECT_FRAME_DIR = WORK_DIR / "effect_frames_1080p60"

FPS = 60
SECONDS = 4
TARGET_FRAMES = FPS * SECONDS
SEGMENT_FRAMES = TARGET_FRAMES // 2
SEGMENT_RIFE_FRAMES = SEGMENT_FRAMES * 2 - 2
OUT_W = base.OUT_W
OUT_H = base.OUT_H
CRF = base.CRF
PRESET = base.PRESET


def build_rife_command(input_dir: Path, output_dir: Path, frames: int = SEGMENT_RIFE_FRAMES) -> list[str]:
    return [
        str(RIFE_EXE),
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-m",
        str(RIFE_MODEL),
        "-n",
        str(frames),
        "-g",
        "0",
        "-j",
        "1:2:2",
    ]


def write_segment_keyframes_from_images(source_a: Image.Image, source_b: Image.Image, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, frame in enumerate((base.prepare_keyframe(source_a), base.prepare_keyframe(source_b))):
        path = output_dir / f"{index:08d}.png"
        frame.save(path, compress_level=1)
        paths.append(path)
    return paths


def write_segment_keyframes(overwrite_sources: bool = False) -> None:
    base.safe_reset_dir(KEYFRAME_AB_DIR, WORK_DIR)
    base.safe_reset_dir(KEYFRAME_BA_DIR, WORK_DIR)
    source_a, source_b = base.load_source_images(overwrite=overwrite_sources)
    write_segment_keyframes_from_images(source_a, source_b, KEYFRAME_AB_DIR)
    write_segment_keyframes_from_images(source_b, source_a, KEYFRAME_BA_DIR)


def run_rife_segment(input_dir: Path, output_dir: Path, overwrite: bool = False) -> None:
    if not RIFE_EXE.exists():
        raise FileNotFoundError(f"RIFE executable not found: {RIFE_EXE}")
    if not RIFE_MODEL.exists():
        raise FileNotFoundError(f"RIFE model not found: {RIFE_MODEL}")
    existing = sorted(output_dir.glob("*.png")) if output_dir.exists() else []
    if not overwrite and len(existing) >= SEGMENT_RIFE_FRAMES:
        print(f"reuse RIFE segment: {output_dir}", flush=True)
        return
    base.safe_reset_dir(output_dir, WORK_DIR)
    subprocess.run(build_rife_command(input_dir, output_dir, SEGMENT_RIFE_FRAMES), check=True)


def copy_motion_half(source_dir: Path, target_dir: Path, start_index: int, segment_frames: int = SEGMENT_FRAMES) -> list[Path]:
    source_paths = sorted(source_dir.glob("*.png"))
    if len(source_paths) < segment_frames:
        raise RuntimeError(f"RIFE segment count {len(source_paths)} < requested motion half {segment_frames}")
    target_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for offset, source_path in enumerate(source_paths[:segment_frames]):
        target_path = target_dir / f"{start_index + offset:08d}.png"
        shutil.copy2(source_path, target_path)
        outputs.append(target_path)
    return outputs


def combine_motion_segments(overwrite: bool = False) -> list[Path]:
    if overwrite:
        base.safe_reset_dir(MOTION_FRAME_DIR, WORK_DIR)
    else:
        MOTION_FRAME_DIR.mkdir(parents=True, exist_ok=True)
    outputs = copy_motion_half(RIFE_AB_DIR, MOTION_FRAME_DIR, 0, SEGMENT_FRAMES)
    outputs.extend(copy_motion_half(RIFE_BA_DIR, MOTION_FRAME_DIR, SEGMENT_FRAMES, SEGMENT_FRAMES))
    return outputs


def rectangle_mask(size: tuple[int, int], box: tuple[int, int, int, int], blur: float) -> np.ndarray:
    image = Image.new("L", size, 0)
    ImageDraw.Draw(image).rectangle(box, fill=255)
    if blur > 0:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(image, dtype=np.float32) / 255.0


def ellipse_mask(size: tuple[int, int], box: tuple[int, int, int, int], blur: float) -> np.ndarray:
    image = Image.new("L", size, 0)
    ImageDraw.Draw(image).ellipse(box, fill=255)
    if blur > 0:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(image, dtype=np.float32) / 255.0


def build_headscarf_mask(size: tuple[int, int] = (base.KEY_W, base.KEY_H)) -> np.ndarray:
    return ellipse_mask(size, (710, 22, 1240, 278), 14.0)


def build_right_hair_detail_mask(size: tuple[int, int] = (base.KEY_W, base.KEY_H)) -> np.ndarray:
    broad = rectangle_mask(size, (1040, 205, 1860, 920), 24.0)
    center_softener = ellipse_mask(size, (720, 0, 1270, 560), 18.0)
    return np.clip(broad * (1.0 - center_softener * 0.55), 0.0, 1.0)


def build_face_settle_mask(size: tuple[int, int] = (base.KEY_W, base.KEY_H)) -> np.ndarray:
    return ellipse_mask(size, (760, 55, 1190, 390), 18.0)


def temporal_smooth_region(
    previous: np.ndarray,
    current: np.ndarray,
    following: np.ndarray,
    mask: np.ndarray,
    strength: float,
) -> np.ndarray:
    local_average = (previous + current * 2.0 + following) * 0.25
    gate = np.clip(mask[:, :, None] * strength, 0.0, 1.0)
    return current * (1.0 - gate) + local_average * gate


def restore_local_detail(frame: np.ndarray, reference: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    reference_image = Image.fromarray(np.clip(reference, 0, 255).astype(np.uint8), "RGB")
    reference_soft = np.asarray(reference_image.filter(ImageFilter.GaussianBlur(1.1)), dtype=np.float32)
    high_frequency = reference - reference_soft
    gate = np.clip(mask[:, :, None] * amount, 0.0, 1.0)
    return np.clip(frame + high_frequency * gate, 0, 255)


def timeline_alpha(index: int, total_frames: int = TARGET_FRAMES) -> float:
    if index < SEGMENT_FRAMES:
        return index / max(1, SEGMENT_FRAMES - 1)
    return 1.0 - ((index - SEGMENT_FRAMES) / max(1, SEGMENT_FRAMES - 1))


def reference_for_detail(source_a: np.ndarray, source_b: np.ndarray, index: int) -> np.ndarray:
    return source_b if timeline_alpha(index) >= 0.5 else source_a


def endpoint_settle_weight(index: int, tail_frames: int = 24) -> float:
    start = TARGET_FRAMES - tail_frames
    if index <= start:
        return 0.0
    phase = (index - start) / max(1, tail_frames - 1)
    phase = min(1.0, max(0.0, phase))
    return phase * phase * (3.0 - 2.0 * phase)


def postprocess_v2_frame(
    frame: np.ndarray,
    index: int,
    source_a: np.ndarray,
    source_b: np.ndarray,
    previous: np.ndarray | None = None,
    following: np.ndarray | None = None,
) -> np.ndarray:
    result = base.sharpen_frame(frame.astype(np.float32))
    red_gate = base.source_red_mask(result)
    red_pulse = 0.03 + 0.08 * base.pulse(index, TARGET_FRAMES, 2.0, -0.10)
    result[:, :, 0] += red_gate * red_pulse * 7.0
    result[:, :, 1] += red_gate * red_pulse * 0.6
    result[:, :, 2] += red_gate * red_pulse * 1.6

    upper = np.zeros((result.shape[0], result.shape[1]), dtype=np.float32)
    upper[: round(result.shape[0] * 0.55), :] = 1.0
    upper = base.mask_float(Image.fromarray((upper * 255).astype(np.uint8), "L"), 22.0)[:, :, None]
    bg_breath = 0.982 + 0.024 * base.pulse(index, TARGET_FRAMES, 1.0, -0.18)
    result = result * (1.0 - upper) + result * bg_breath * upper

    headscarf = build_headscarf_mask((result.shape[1], result.shape[0]))
    right_hair = build_right_hair_detail_mask((result.shape[1], result.shape[0]))
    face_settle = build_face_settle_mask((result.shape[1], result.shape[0]))
    if previous is not None and following is not None:
        result = temporal_smooth_region(previous, result, following, headscarf, strength=0.42)

    settle = endpoint_settle_weight(index)
    if settle > 0.0:
        settle_mask = np.maximum.reduce([headscarf * 0.92, right_hair * 0.58, face_settle * 0.42])
        gate = np.clip(settle_mask[:, :, None] * settle, 0.0, 1.0)
        result = result * (1.0 - gate) + source_a * gate

    reference = reference_for_detail(source_a, source_b, index)
    result = restore_local_detail(result, reference, headscarf, amount=0.18)
    result = restore_local_detail(result, reference, right_hair, amount=0.34)
    return np.clip(result, 0, 255)


def load_prepared_sources(overwrite_sources: bool = False) -> tuple[np.ndarray, np.ndarray]:
    source_a, source_b = base.load_source_images(overwrite=overwrite_sources)
    return (
        np.asarray(base.prepare_keyframe(source_a), dtype=np.float32),
        np.asarray(base.prepare_keyframe(source_b), dtype=np.float32),
    )


def write_effect_frames(overwrite: bool = False, overwrite_sources: bool = False) -> list[Path]:
    motion_frames = sorted(MOTION_FRAME_DIR.glob("*.png"))
    if len(motion_frames) < TARGET_FRAMES:
        raise RuntimeError(f"motion frame count {len(motion_frames)} < {TARGET_FRAMES}")
    if overwrite:
        base.safe_reset_dir(EFFECT_FRAME_DIR, WORK_DIR)
    else:
        EFFECT_FRAME_DIR.mkdir(parents=True, exist_ok=True)

    source_a, source_b = load_prepared_sources(overwrite_sources=overwrite_sources)
    outputs: list[Path] = []
    for index, path in enumerate(motion_frames[:TARGET_FRAMES]):
        out_path = EFFECT_FRAME_DIR / f"{index:08d}.png"
        if out_path.exists() and not overwrite:
            outputs.append(out_path)
            continue
        current = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        previous_path = motion_frames[max(0, index - 1)]
        following_path = motion_frames[min(TARGET_FRAMES - 1, index + 1)]
        previous = np.asarray(Image.open(previous_path).convert("RGB"), dtype=np.float32)
        following = np.asarray(Image.open(following_path).convert("RGB"), dtype=np.float32)
        processed = postprocess_v2_frame(current, index, source_a, source_b, previous, following)
        Image.fromarray(processed.astype(np.uint8), "RGB").save(out_path, compress_level=1)
        outputs.append(out_path)
        if index % 20 == 0:
            print(f"postprocessed v2 frame {index + 1}/{TARGET_FRAMES}", flush=True)
    return outputs


def static_tail_count(frames: list[np.ndarray], threshold: float = 0.08) -> int:
    count = 0
    for index in range(len(frames) - 1, 0, -1):
        if float(np.abs(frames[index] - frames[index - 1]).mean()) < threshold:
            count += 1
        else:
            break
    return count


def build_ffmpeg_command(output: Path, input_dir: Path = EFFECT_FRAME_DIR, frames: int = TARGET_FRAMES) -> list[str]:
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
        str(frames),
        "-vf",
        f"scale={OUT_W}:{OUT_H}:flags=lanczos,unsharp=5:5:0.18:3:3:0.04,format=yuv420p",
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
    subprocess.run(build_ffmpeg_command(output, EFFECT_FRAME_DIR, TARGET_FRAMES), check=True)
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
    sheet.save(out_path, quality=93)


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
    sheet.save(out_path, quality=94)


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
    sheet.save(out_path, quality=93)


def write_review_artifacts(review_dir: Path = REVIEW_DIR) -> list[Path]:
    review_dir.mkdir(parents=True, exist_ok=True)
    indices = sorted({round(i * (TARGET_FRAMES - 1) / 8) for i in range(9)})
    frame_paths: list[Path] = []
    for index in indices:
        src = EFFECT_FRAME_DIR / f"{index:08d}.png"
        if not src.exists():
            continue
        out = review_dir / f"rife_interpolated_v2_sample_{index:06d}.jpg"
        Image.open(src).convert("RGB").save(out, quality=94)
        frame_paths.append(out)
    outputs: list[Path] = list(frame_paths)
    if frame_paths:
        overview = review_dir / "rife_interpolated_v2_overview.jpg"
        make_contact_sheet(frame_paths, overview)
        outputs.append(overview)
        face = review_dir / "rife_interpolated_v2_face_crops.jpg"
        make_crop_sheet(frame_paths, (780, 85, 1168, 355), face)
        outputs.append(face)
        headscarf = review_dir / "rife_interpolated_v2_headscarf_crops.jpg"
        make_crop_sheet(frame_paths, (710, 22, 1240, 278), headscarf)
        outputs.append(headscarf)
        right_hair = review_dir / "rife_interpolated_v2_right_hair_crops.jpg"
        make_crop_sheet(frame_paths, (1040, 205, 1860, 920), right_hair)
        outputs.append(right_hair)
        loop = review_dir / "rife_interpolated_v2_loop_check.jpg"
        write_loop_check(Image.open(frame_paths[0]), Image.open(frame_paths[-1]), loop)
        outputs.append(loop)
        preview = review_dir / "rife_interpolated_v2_preview.jpg"
        shutil.copy2(frame_paths[0], preview)
        outputs.append(preview)
    return outputs


def write_project(video_path: Path, preview_path: Path = REVIEW_DIR / "rife_interpolated_v2_preview.jpg") -> None:
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_path, PROJECT_VIDEO)
    if preview_path.exists():
        shutil.copy2(preview_path, PROJECT_DIR / "preview.jpg")
    project = {
        "title": "Scathach Two Source RIFE Interpolated Loop V2",
        "description": "Looping 2160p60 4s wallpaper using trimmed two-segment RIFE motion, headscarf temporal smoothing, right hair detail restoration, and source-gated red glow.",
        "type": "video",
        "file": PROJECT_VIDEO.name,
        "preview": "preview.jpg",
        "tags": ["Anime"],
        "visibility": "private",
    }
    (PROJECT_DIR / "project.json").write_text(json.dumps(project, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_pipeline(overwrite_sources: bool = False, overwrite_rife: bool = False, overwrite_effects: bool = False) -> None:
    write_segment_keyframes(overwrite_sources=overwrite_sources)
    run_rife_segment(KEYFRAME_AB_DIR, RIFE_AB_DIR, overwrite=overwrite_rife)
    run_rife_segment(KEYFRAME_BA_DIR, RIFE_BA_DIR, overwrite=overwrite_rife)
    combine_motion_segments(overwrite=True)
    write_effect_frames(overwrite=overwrite_effects, overwrite_sources=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["keyframes", "rife", "effects", "review", "video", "all"], default="all")
    parser.add_argument("--overwrite-sources", action="store_true")
    parser.add_argument("--overwrite-rife", action="store_true")
    parser.add_argument("--overwrite-effects", action="store_true")
    parser.add_argument("--no-project", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "keyframes":
        write_segment_keyframes(overwrite_sources=args.overwrite_sources)
        print(f"keyframes: {KEYFRAME_AB_DIR}, {KEYFRAME_BA_DIR}")
        return
    if args.mode == "rife":
        write_segment_keyframes(overwrite_sources=args.overwrite_sources)
        run_rife_segment(KEYFRAME_AB_DIR, RIFE_AB_DIR, overwrite=args.overwrite_rife)
        run_rife_segment(KEYFRAME_BA_DIR, RIFE_BA_DIR, overwrite=args.overwrite_rife)
        combine_motion_segments(overwrite=True)
        print(f"motion frames: {MOTION_FRAME_DIR}")
        return
    if args.mode in {"effects", "review", "video", "all"}:
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
            preview = REVIEW_DIR / "rife_interpolated_v2_preview.jpg"
            if not preview.exists():
                write_review_artifacts(REVIEW_DIR)
            write_project(video, preview)
        print(video.resolve())


if __name__ == "__main__":
    main()

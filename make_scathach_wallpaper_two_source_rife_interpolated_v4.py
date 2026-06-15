from __future__ import annotations

import argparse
import json
import math
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
REVIEW_DIR = OUT_DIR / "two_source_rife_interpolated_v4_review"
OUTPUT_VIDEO = OUT_DIR / "scathach_two_source_rife_interpolated_2160p60_4s_v4_plate.mp4"
PROJECT_VIDEO = PROJECT_DIR / OUTPUT_VIDEO.name

FPS = v2.FPS
TARGET_FRAMES = v2.TARGET_FRAMES
OUT_W = base.OUT_W
OUT_H = base.OUT_H
CRF = base.CRF
PRESET = base.PRESET


def scaled_box(box: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    sx = width / base.KEY_W
    sy = height / base.KEY_H
    return (
        round(box[0] * sx),
        round(box[1] * sy),
        round(box[2] * sx),
        round(box[3] * sy),
    )


def rectangle_mask(size: tuple[int, int], box: tuple[int, int, int, int], blur: float) -> np.ndarray:
    image = Image.new("L", size, 0)
    ImageDraw.Draw(image).rectangle(scaled_box(box, size), fill=255)
    if blur > 0:
        width, _height = size
        image = image.filter(ImageFilter.GaussianBlur(blur * width / base.KEY_W))
    return np.asarray(image, dtype=np.float32) / 255.0


def ellipse_mask(size: tuple[int, int], box: tuple[int, int, int, int], blur: float) -> np.ndarray:
    image = Image.new("L", size, 0)
    ImageDraw.Draw(image).ellipse(scaled_box(box, size), fill=255)
    if blur > 0:
        width, _height = size
        image = image.filter(ImageFilter.GaussianBlur(blur * width / base.KEY_W))
    return np.asarray(image, dtype=np.float32) / 255.0


def build_problem_detail_mask_4k(size: tuple[int, int] = (OUT_W, OUT_H)) -> np.ndarray:
    right_hair = rectangle_mask(size, (820, 235, 1920, 980), 18.0)
    lower_flowers = rectangle_mask(size, (520, 485, 1920, 1080), 16.0)
    face_guard = ellipse_mask(size, (720, 35, 1240, 420), 24.0)
    torso_guard = ellipse_mask(size, (520, 365, 1090, 1080), 30.0)
    hand_guard = ellipse_mask(size, (430, 255, 650, 540), 20.0)
    mask = np.maximum(right_hair, lower_flowers)
    mask = mask * (1.0 - face_guard * 0.98)
    mask = mask * (1.0 - torso_guard * 0.42)
    mask = mask * (1.0 - hand_guard * 0.65)
    return np.clip(mask, 0.0, 1.0)


def edge_strength(source: np.ndarray) -> np.ndarray:
    gray = source[:, :, 0] * 0.299 + source[:, :, 1] * 0.587 + source[:, :, 2] * 0.114
    image = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), "L")
    edge = image.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
    width = source.shape[1]
    edge = edge.filter(ImageFilter.GaussianBlur(max(0.35, 0.55 * width / OUT_W)))
    values = np.asarray(edge, dtype=np.float32) / 255.0
    return np.clip((values - 0.025) / 0.18, 0.0, 1.0)


def build_source_edge_mask(source_a: np.ndarray, source_b: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
    edges = np.maximum(edge_strength(source_a), edge_strength(source_b))
    edges = np.asarray(
        Image.fromarray((edges * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(0.35)),
        dtype=np.float32,
    ) / 255.0
    return np.clip(np.maximum(base_mask, edges * 0.78), 0.0, 1.0)


def build_deghost_anchor_mask(detail_mask: np.ndarray, size: tuple[int, int] = (OUT_W, OUT_H)) -> np.ndarray:
    torso = ellipse_mask(size, (580, 170, 1395, 1080), 20.0)
    weapon = rectangle_mask(size, (500, 585, 1710, 875), 16.0)
    face_guard = ellipse_mask(size, (710, 35, 1240, 420), 24.0)
    anchor_area = np.clip(np.maximum(torso * 0.78, weapon), 0.0, 1.0)
    anchor_area = anchor_area * (1.0 - face_guard * 0.75)
    return np.clip(detail_mask * anchor_area, 0.0, 1.0)


def build_flower_light_mask(source_a: np.ndarray, source_b: np.ndarray) -> np.ndarray:
    size = (source_a.shape[1], source_a.shape[0])
    strongest = np.maximum(source_a, source_b)
    red = strongest[:, :, 0]
    green = strongest[:, :, 1]
    blue = strongest[:, :, 2]
    red_pixels = ((red > 62.0) & (red - green > 16.0) & (red - blue > 4.0)).astype(np.float32)

    lower_field = rectangle_mask(size, (0, 430, 1920, 1080), 26.0)
    right_field = rectangle_mask(size, (1140, 95, 1920, 700), 24.0)
    left_field = rectangle_mask(size, (0, 80, 625, 700), 24.0)
    flower_area = np.maximum.reduce([lower_field, right_field, left_field])

    figure_guard = ellipse_mask(size, (515, 0, 1380, 1080), 36.0)
    weapon_guard = rectangle_mask(size, (450, 560, 1720, 900), 24.0)
    guard = np.maximum(figure_guard * 0.96, weapon_guard * 0.92)

    mask = red_pixels * flower_area * (1.0 - guard)
    mask_image = Image.fromarray(np.clip(mask * 255, 0, 255).astype(np.uint8), "L")
    mask_image = mask_image.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(8.0 * size[0] / OUT_W))
    return np.asarray(mask_image, dtype=np.float32) / 255.0


def flower_light_weight(index: int, total_frames: int = TARGET_FRAMES) -> float:
    phase = index / max(1, total_frames)
    return -math.cos(4.0 * math.pi * phase)


def apply_flower_lighting(
    frame: np.ndarray,
    mask: np.ndarray,
    index: int,
    total_frames: int = TARGET_FRAMES,
) -> np.ndarray:
    weight = flower_light_weight(index, total_frames)
    gate = np.clip(mask[:, :, None], 0.0, 1.0)
    if weight > 0.0:
        lit = frame * (1.0 + gate * weight * 0.22)
        glow = np.array([22.0, 3.4, 1.6], dtype=np.float32) * weight
    else:
        lit = frame * (1.0 + gate * weight * 0.30)
        glow = np.array([-8.0, -1.8, -0.8], dtype=np.float32) * (-weight)
    return np.clip(lit + gate * glow, 0, 255)


def source_high_frequency(source: np.ndarray, radius: float = 1.25) -> np.ndarray:
    image = Image.fromarray(np.clip(source, 0, 255).astype(np.uint8), "RGB")
    soft = np.asarray(image.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    return source - soft


def detail_blend_weight(alpha: float) -> float:
    if alpha <= 0.44:
        return 0.0
    if alpha >= 0.56:
        return 1.0
    phase = (alpha - 0.44) / 0.12
    return phase * phase * (3.0 - 2.0 * phase)


def apply_4k_detail_plate(
    frame: np.ndarray,
    high_a: np.ndarray,
    high_b: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    amount: float = 0.88,
    anchor_mask: np.ndarray | None = None,
) -> np.ndarray:
    detail_alpha = detail_blend_weight(alpha)
    detail = high_a * (1.0 - detail_alpha) + high_b * detail_alpha
    if anchor_mask is not None:
        nearest = high_b if alpha >= 0.5 else high_a
        anchor_gate = np.clip(anchor_mask[:, :, None] * 0.85, 0.0, 1.0)
        detail = detail * (1.0 - anchor_gate) + nearest * anchor_gate
    frame_image = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), "RGB")
    current_soft = np.asarray(frame_image.filter(ImageFilter.GaussianBlur(0.65)), dtype=np.float32)
    current_high = frame - current_soft
    gate = np.clip(mask[:, :, None], 0.0, 1.0)
    boosted = frame + detail * gate * amount + current_high * gate * 0.28
    return np.clip(boosted, 0, 255)


def apply_edge_deghost(
    frame: np.ndarray,
    source_a: np.ndarray,
    source_b: np.ndarray,
    anchor_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    reference = source_b if alpha >= 0.5 else source_a
    gate = np.clip(anchor_mask[:, :, None], 0.0, 1.0)
    frame_image = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), "RGB")
    ref_image = Image.fromarray(np.clip(reference, 0, 255).astype(np.uint8), "RGB")
    frame_high = frame - np.asarray(frame_image.filter(ImageFilter.GaussianBlur(0.7)), dtype=np.float32)
    ref_high = reference - np.asarray(ref_image.filter(ImageFilter.GaussianBlur(0.95)), dtype=np.float32)
    sharpened = frame + frame_high * gate * 0.24 + ref_high * gate * 0.42
    anchored = sharpened * (1.0 - gate * 0.055) + reference * (gate * 0.055)
    return np.clip(anchored, 0, 255)


def load_4k_sources(overwrite_sources: bool = False) -> tuple[np.ndarray, np.ndarray]:
    source_a, source_b = base.load_source_images(overwrite=overwrite_sources)
    return np.asarray(source_a, dtype=np.float32), np.asarray(source_b, dtype=np.float32)


def ensure_motion_frames(overwrite_sources: bool = False, overwrite_rife: bool = False) -> None:
    motion_frames = sorted(v2.MOTION_FRAME_DIR.glob("*.png")) if v2.MOTION_FRAME_DIR.exists() else []
    if len(motion_frames) >= TARGET_FRAMES and not overwrite_rife:
        return
    v2.write_segment_keyframes(overwrite_sources=overwrite_sources)
    v2.run_rife_segment(v2.KEYFRAME_AB_DIR, v2.RIFE_AB_DIR, overwrite=overwrite_rife)
    v2.run_rife_segment(v2.KEYFRAME_BA_DIR, v2.RIFE_BA_DIR, overwrite=overwrite_rife)
    v2.combine_motion_segments(overwrite=True)


def upscaled_v2_frame(
    motion_path: Path,
    index: int,
    source_a_1080: np.ndarray,
    source_b_1080: np.ndarray,
    motion_frames: list[Path],
) -> np.ndarray:
    current = np.asarray(Image.open(motion_path).convert("RGB"), dtype=np.float32)
    previous = np.asarray(Image.open(motion_frames[max(0, index - 1)]).convert("RGB"), dtype=np.float32)
    following = np.asarray(Image.open(motion_frames[min(TARGET_FRAMES - 1, index + 1)]).convert("RGB"), dtype=np.float32)
    processed_1080 = v2.postprocess_v2_frame(current, index, source_a_1080, source_b_1080, previous, following)
    image = Image.fromarray(processed_1080.astype(np.uint8), "RGB").resize((OUT_W, OUT_H), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32)


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
    motion_frames: list[Path],
) -> np.ndarray:
    frame = upscaled_v2_frame(motion_path, index, source_a_1080, source_b_1080, motion_frames)
    alpha = v2.timeline_alpha(index, TARGET_FRAMES)
    frame = apply_4k_detail_plate(
        frame,
        high_a_4k,
        high_b_4k,
        detail_mask_4k,
        alpha=alpha,
        amount=0.82,
        anchor_mask=anchor_mask_4k,
    )
    frame = apply_edge_deghost(frame, source_a_4k, source_b_4k, anchor_mask_4k, alpha)
    return apply_flower_lighting(frame, flower_mask_4k, index, TARGET_FRAMES)


def prepared_context(
    overwrite_sources: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_a_1080, source_b_1080 = v2.load_prepared_sources(overwrite_sources=overwrite_sources)
    source_a_4k, source_b_4k = load_4k_sources(overwrite_sources=False)
    high_a_4k = source_high_frequency(source_a_4k)
    high_b_4k = source_high_frequency(source_b_4k)
    problem_mask_4k = build_problem_detail_mask_4k((OUT_W, OUT_H))
    detail_mask_4k = build_source_edge_mask(source_a_4k, source_b_4k, problem_mask_4k)
    anchor_mask_4k = build_deghost_anchor_mask(detail_mask_4k, (OUT_W, OUT_H))
    flower_mask_4k = build_flower_light_mask(source_a_4k, source_b_4k)
    return (
        source_a_1080,
        source_b_1080,
        source_a_4k,
        source_b_4k,
        high_a_4k,
        high_b_4k,
        detail_mask_4k,
        anchor_mask_4k,
        flower_mask_4k,
    )


def write_review_artifacts(overwrite_sources: bool = False, review_dir: Path = REVIEW_DIR) -> list[Path]:
    ensure_motion_frames(overwrite_sources=overwrite_sources, overwrite_rife=False)
    motion_frames = sorted(v2.MOTION_FRAME_DIR.glob("*.png"))
    (
        source_a_1080,
        source_b_1080,
        source_a_4k,
        source_b_4k,
        high_a_4k,
        high_b_4k,
        detail_mask_4k,
        anchor_mask_4k,
        flower_mask_4k,
    ) = prepared_context(overwrite_sources=overwrite_sources)
    review_dir.mkdir(parents=True, exist_ok=True)

    indices = sorted({round(i * (TARGET_FRAMES - 1) / 8) for i in range(9)})
    frame_paths: list[Path] = []
    for index in indices:
        frame = render_frame_4k(
            motion_frames[index],
            index,
            source_a_1080,
            source_b_1080,
            source_a_4k,
            source_b_4k,
            high_a_4k,
            high_b_4k,
            detail_mask_4k,
            anchor_mask_4k,
            flower_mask_4k,
            motion_frames,
        ).astype(np.uint8)
        out = review_dir / f"rife_interpolated_v4_sample_{index:06d}.jpg"
        Image.fromarray(frame, "RGB").save(out, quality=94)
        frame_paths.append(out)

    outputs: list[Path] = list(frame_paths)
    overview = review_dir / "rife_interpolated_v4_overview.jpg"
    make_contact_sheet(frame_paths, overview)
    outputs.append(overview)
    problem = review_dir / "rife_interpolated_v4_problem_area_crops.jpg"
    make_crop_sheet(frame_paths, (1200, 970, 3840, 2160), problem)
    outputs.append(problem)
    torso = review_dir / "rife_interpolated_v4_torso_weapon_crops.jpg"
    make_crop_sheet(frame_paths, (920, 410, 2600, 1770), torso)
    outputs.append(torso)
    flower = review_dir / "rife_interpolated_v4_flower_light_crops.jpg"
    make_crop_sheet(frame_paths, (0, 760, 3840, 2160), flower)
    outputs.append(flower)
    loop = review_dir / "rife_interpolated_v4_loop_check.jpg"
    write_loop_check(Image.open(frame_paths[0]), Image.open(frame_paths[-1]), loop)
    outputs.append(loop)
    preview = review_dir / "rife_interpolated_v4_preview.jpg"
    shutil.copy2(frame_paths[0], preview)
    outputs.append(preview)
    return outputs


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
        image.thumbnail((420, 300), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (420, 324), (18, 18, 22))
        canvas.paste(image, ((420 - image.width) // 2, 0))
        ImageDraw.Draw(canvas).text((8, 305), path.stem, fill=(235, 235, 235))
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


def encode_video(overwrite_sources: bool = False, output: Path = OUTPUT_VIDEO) -> Path:
    ensure_motion_frames(overwrite_sources=overwrite_sources, overwrite_rife=False)
    motion_frames = sorted(v2.MOTION_FRAME_DIR.glob("*.png"))
    (
        source_a_1080,
        source_b_1080,
        source_a_4k,
        source_b_4k,
        high_a_4k,
        high_b_4k,
        detail_mask_4k,
        anchor_mask_4k,
        flower_mask_4k,
    ) = prepared_context(overwrite_sources=overwrite_sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        base.find_ffmpeg(),
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
        "pipe:0",
        "-frames:v",
        str(TARGET_FRAMES),
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
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for index, motion_path in enumerate(motion_frames[:TARGET_FRAMES]):
            frame = render_frame_4k(
                motion_path,
                index,
                source_a_1080,
                source_b_1080,
                source_a_4k,
                source_b_4k,
                high_a_4k,
                high_b_4k,
                detail_mask_4k,
                anchor_mask_4k,
                flower_mask_4k,
                motion_frames,
            ).astype(np.uint8)
            proc.stdin.write(frame.tobytes())
            if index % 20 == 0:
                print(f"streamed v4 4k frame {index + 1}/{TARGET_FRAMES}", flush=True)
    finally:
        proc.stdin.close()
    return_code = proc.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return output


def write_project(video_path: Path, preview_path: Path = REVIEW_DIR / "rife_interpolated_v4_preview.jpg") -> None:
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_path, PROJECT_VIDEO)
    if preview_path.exists():
        shutil.copy2(preview_path, PROJECT_DIR / "preview.jpg")
    project = {
        "title": "Scathach Two Source RIFE Interpolated Loop V4 Plate",
        "description": "Looping 2160p60 4s wallpaper using v2 motion plus original 4K source detail plates, edge deghosting, and flower light pulsing.",
        "type": "video",
        "file": PROJECT_VIDEO.name,
        "preview": "preview.jpg",
        "tags": ["Anime"],
        "visibility": "private",
    }
    (PROJECT_DIR / "project.json").write_text(json.dumps(project, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["review", "video", "all"], default="all")
    parser.add_argument("--overwrite-sources", action="store_true")
    parser.add_argument("--no-project", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {"review", "all"}:
        review = write_review_artifacts(overwrite_sources=args.overwrite_sources, review_dir=REVIEW_DIR)
        print(f"review artifacts: {REVIEW_DIR} ({len(review)})")
    if args.mode in {"video", "all"}:
        video = encode_video(overwrite_sources=args.overwrite_sources, output=OUTPUT_VIDEO)
        if not args.no_project:
            preview = REVIEW_DIR / "rife_interpolated_v4_preview.jpg"
            if not preview.exists():
                write_review_artifacts(overwrite_sources=args.overwrite_sources, review_dir=REVIEW_DIR)
            write_project(video, preview)
        print(video.resolve())


if __name__ == "__main__":
    main()

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


ROOT = base.ROOT
OUT_DIR = base.OUT_DIR
PROJECT_DIR = base.PROJECT_DIR
REVIEW_DIR = OUT_DIR / "two_source_rife_interpolated_v5_review"
ALL_FRAME_REVIEW_DIR = OUT_DIR / "v5_all_frame_ghost_review"
OUTPUT_VIDEO = OUT_DIR / "scathach_two_source_rife_interpolated_2160p60_4s_v5_deghost.mp4"
PROJECT_VIDEO = PROJECT_DIR / OUTPUT_VIDEO.name

FPS = v4.FPS
TARGET_FRAMES = v4.TARGET_FRAMES
OUT_W = v4.OUT_W
OUT_H = v4.OUT_H
CRF = v4.CRF
PRESET = v4.PRESET


def build_ghost_risk_mask(size: tuple[int, int] = (OUT_W, OUT_H)) -> np.ndarray:
    torso = v4.ellipse_mask(size, (500, 125, 1460, 1080), 18.0)
    weapon = v4.rectangle_mask(size, (0, 520, 1920, 930), 12.0)
    right_hair = v4.rectangle_mask(size, (760, 190, 1920, 1010), 18.0)
    flower_field = v4.rectangle_mask(size, (700, 360, 1920, 1080), 20.0) * 0.55
    face_guard = v4.ellipse_mask(size, (700, 20, 1255, 420), 22.0)
    risk = np.maximum.reduce([torso * 0.92, weapon, right_hair, flower_field])
    risk = risk * (1.0 - face_guard * 0.88)
    return np.clip(risk, 0.0, 1.0)


def build_ghost_cleanup_mask_from_risk(source_a: np.ndarray, source_b: np.ndarray, risk_mask: np.ndarray) -> np.ndarray:
    edges = np.maximum(v4.edge_strength(source_a), v4.edge_strength(source_b))
    line_mask = np.clip((edges - 0.10) / 0.32, 0.0, 1.0)
    darkest = np.minimum(source_a, source_b)
    strongest = np.maximum(source_a, source_b)
    luma = darkest[:, :, 0] * 0.299 + darkest[:, :, 1] * 0.587 + darkest[:, :, 2] * 0.114
    red = strongest[:, :, 0]
    green = strongest[:, :, 1]
    blue = strongest[:, :, 2]
    dark_line = luma < 112.0
    cold_red_line = (red > 92.0) & (green < 108.0) & (blue < 118.0) & (red - green > 34.0)
    content_gate = np.where(dark_line | cold_red_line, 1.0, 0.0).astype(np.float32)
    mask = np.clip(line_mask * risk_mask * content_gate, 0.0, 1.0)
    image = Image.fromarray(np.clip(mask * 255, 0, 255).astype(np.uint8), "L")
    image = image.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(max(0.25, source_a.shape[1] / OUT_W * 0.45)))
    return np.asarray(image, dtype=np.float32) / 255.0


def build_ghost_cleanup_mask(source_a: np.ndarray, source_b: np.ndarray) -> np.ndarray:
    return build_ghost_cleanup_mask_from_risk(source_a, source_b, build_ghost_risk_mask((source_a.shape[1], source_a.shape[0])))


def cleanup_reference(source_a: np.ndarray, source_b: np.ndarray, alpha: float) -> np.ndarray:
    return source_b if alpha >= 0.5 else source_a


def apply_phase_source_cleanup(
    frame: np.ndarray,
    source_a: np.ndarray,
    source_b: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    strength: float = 0.48,
) -> np.ndarray:
    reference = cleanup_reference(source_a, source_b, alpha)
    gate = np.clip(mask[:, :, None] * strength, 0.0, 0.85)
    ref_image = Image.fromarray(np.clip(reference, 0, 255).astype(np.uint8), "RGB")
    ref_soft = np.asarray(ref_image.filter(ImageFilter.GaussianBlur(0.85)), dtype=np.float32)
    ref_high = reference - ref_soft
    anchored = frame * (1.0 - gate) + reference * gate
    anchored = anchored + ref_high * gate * 0.18
    return np.clip(anchored, 0, 255)


def prepared_context(
    overwrite_sources: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
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
    ) = v4.prepared_context(overwrite_sources=overwrite_sources)
    ghost_cleanup_mask_4k = build_ghost_cleanup_mask(source_a_4k, source_b_4k)
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
        ghost_cleanup_mask_4k,
    )


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
    ghost_cleanup_mask_4k: np.ndarray,
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
    frame = v4.apply_edge_deghost(frame, source_a_4k, source_b_4k, anchor_mask_4k, alpha)
    frame = apply_phase_source_cleanup(frame, source_a_4k, source_b_4k, ghost_cleanup_mask_4k, alpha, strength=0.54)
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
        out = review_dir / f"rife_interpolated_v5_sample_{index:06d}.jpg"
        Image.fromarray(frame, "RGB").save(out, quality=94)
        frame_paths.append(out)

    outputs: list[Path] = list(frame_paths)
    overview = review_dir / "rife_interpolated_v5_overview.jpg"
    make_contact_sheet(frame_paths, overview)
    outputs.append(overview)
    torso = review_dir / "rife_interpolated_v5_torso_weapon_crops.jpg"
    make_crop_sheet(frame_paths, (1160, 430, 2550, 1780), torso)
    outputs.append(torso)
    hair = review_dir / "rife_interpolated_v5_right_hair_field_crops.jpg"
    make_crop_sheet(frame_paths, (1810, 260, 3840, 1800), hair)
    outputs.append(hair)
    loop = review_dir / "rife_interpolated_v5_loop_check.jpg"
    write_loop_check(Image.open(frame_paths[0]), Image.open(frame_paths[-1]), loop)
    outputs.append(loop)
    preview = review_dir / "rife_interpolated_v5_preview.jpg"
    shutil.copy2(frame_paths[0], preview)
    outputs.append(preview)
    return outputs


def write_all_frame_review(overwrite_sources: bool = False, review_dir: Path = ALL_FRAME_REVIEW_DIR) -> list[Path]:
    v4.ensure_motion_frames(overwrite_sources=overwrite_sources, overwrite_rife=False)
    motion_frames = sorted(v2.MOTION_FRAME_DIR.glob("*.png"))
    context = prepared_context(overwrite_sources=overwrite_sources)
    review_dir.mkdir(parents=True, exist_ok=True)
    crops = {
        "torso_weapon": (1160, 430, 2550, 1780),
        "right_hair_field": (1810, 260, 3840, 1800),
    }
    crop_frames: dict[str, list[Path]] = {key: [] for key in crops}

    for index in range(TARGET_FRAMES):
        frame = Image.fromarray(render_context_frame(context, motion_frames, index).astype(np.uint8), "RGB")
        for name, box in crops.items():
            out = review_dir / f"{name}_{index:03d}.jpg"
            crop = frame.crop(box)
            crop.thumbnail((220, 200), Image.Resampling.LANCZOS)
            crop.save(out, quality=94)
            crop_frames[name].append(out)
        if index % 20 == 0:
            print(f"reviewed v5 crop frame {index + 1}/{TARGET_FRAMES}", flush=True)

    outputs: list[Path] = []
    for name, paths in crop_frames.items():
        outputs.extend(make_all_frame_pages(paths, review_dir, name))
    return outputs


def make_all_frame_pages(paths: list[Path], review_dir: Path, name: str) -> list[Path]:
    outputs: list[Path] = []
    for page in range(12):
        subset = paths[page * 20 : (page + 1) * 20]
        thumbs: list[Image.Image] = []
        for path in subset:
            index = int(path.stem.rsplit("_", 1)[1])
            image = Image.open(path).convert("RGB")
            canvas = Image.new("RGB", (230, 230), (16, 16, 20))
            canvas.paste(image, ((230 - image.width) // 2, 2))
            ImageDraw.Draw(canvas).text((8, 208), f"{index:03d}", fill=(235, 235, 235))
            thumbs.append(canvas)
        sheet = Image.new("RGB", (5 * 230, 4 * 230), (16, 16, 20))
        for index, thumb in enumerate(thumbs):
            sheet.paste(thumb, ((index % 5) * 230, (index // 5) * 230))
        out = review_dir / f"{name}_page_{page + 1:02d}.jpg"
        sheet.save(out, quality=95)
        outputs.append(out)
    return outputs


def make_contact_sheet(paths: list[Path], out_path: Path, thumb_size: tuple[int, int] = (320, 180)) -> None:
    v4.make_contact_sheet(paths, out_path, thumb_size)


def make_crop_sheet(paths: list[Path], crop_box: tuple[int, int, int, int], out_path: Path) -> None:
    v4.make_crop_sheet(paths, crop_box, out_path)


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
    v4.ensure_motion_frames(overwrite_sources=overwrite_sources, overwrite_rife=False)
    motion_frames = sorted(v2.MOTION_FRAME_DIR.glob("*.png"))
    context = prepared_context(overwrite_sources=overwrite_sources)
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
            frame = render_context_frame(context, motion_frames, index).astype(np.uint8)
            proc.stdin.write(frame.tobytes())
            if index % 20 == 0:
                print(f"streamed v5 4k frame {index + 1}/{TARGET_FRAMES}", flush=True)
    finally:
        proc.stdin.close()
    return_code = proc.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return output


def write_project(video_path: Path, preview_path: Path = REVIEW_DIR / "rife_interpolated_v5_preview.jpg") -> None:
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_path, PROJECT_VIDEO)
    if preview_path.exists():
        shutil.copy2(preview_path, PROJECT_DIR / "preview.jpg")
    project = {
        "title": "Scathach Two Source RIFE Interpolated Loop V5 Deghost",
        "description": "Looping 2160p60 4s wallpaper using v4 4K detail plates plus phase-source cleanup for faint edge trails.",
        "type": "video",
        "file": PROJECT_VIDEO.name,
        "preview": "preview.jpg",
        "tags": ["Anime"],
        "visibility": "private",
    }
    (PROJECT_DIR / "project.json").write_text(json.dumps(project, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["review", "all-frame-review", "video", "all"], default="all")
    parser.add_argument("--overwrite-sources", action="store_true")
    parser.add_argument("--no-project", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {"review", "all"}:
        review = write_review_artifacts(overwrite_sources=args.overwrite_sources, review_dir=REVIEW_DIR)
        print(f"review artifacts: {REVIEW_DIR} ({len(review)})")
    if args.mode == "all-frame-review":
        outputs = write_all_frame_review(overwrite_sources=args.overwrite_sources, review_dir=ALL_FRAME_REVIEW_DIR)
        print(f"all-frame review artifacts: {ALL_FRAME_REVIEW_DIR} ({len(outputs)})")
    if args.mode in {"video", "all"}:
        video = encode_video(overwrite_sources=args.overwrite_sources, output=OUTPUT_VIDEO)
        if not args.no_project:
            preview = REVIEW_DIR / "rife_interpolated_v5_preview.jpg"
            if not preview.exists():
                write_review_artifacts(overwrite_sources=args.overwrite_sources, review_dir=REVIEW_DIR)
            write_project(video, preview)
        print(video.resolve())


if __name__ == "__main__":
    main()

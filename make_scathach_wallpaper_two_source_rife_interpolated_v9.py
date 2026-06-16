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
REVIEW_DIR = OUT_DIR / "two_source_rife_interpolated_v9_review"
ALL_FRAME_REVIEW_DIR = OUT_DIR / "v9_all_frame_strap_overlay_review"
OUTPUT_VIDEO = OUT_DIR / "scathach_two_source_rife_interpolated_2160p60_4s_v9_strap_overlay.mp4"
PROJECT_VIDEO = PROJECT_DIR / OUTPUT_VIDEO.name

FPS = v6.FPS
TARGET_FRAMES = v6.TARGET_FRAMES
OUT_W = v6.OUT_W
OUT_H = v6.OUT_H
CRF = v6.CRF
PRESET = v6.PRESET


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if value <= edge0:
        return 0.0
    if value >= edge1:
        return 1.0
    t = (value - edge0) / (edge1 - edge0)
    return t * t * (3.0 - 2.0 * t)


def strap_layer_weights(alpha: float) -> tuple[float, float, float]:
    weight_b = smoothstep(0.42, 0.58, alpha)
    weight_a = 1.0 - weight_b
    crossfade = 4.0 * weight_a * weight_b
    opacity = 0.94 - 0.18 * crossfade
    return weight_a, weight_b, opacity


def build_strap_area_mask(size: tuple[int, int] = (OUT_W, OUT_H)) -> np.ndarray:
    torso = v4.ellipse_mask(size, (500, 120, 1485, 1080), 16.0)
    lower_torso = v4.rectangle_mask(size, (430, 390, 1220, 1080), 16.0) * 0.82
    weapon_band = v4.rectangle_mask(size, (0, 515, 1920, 930), 10.0) * 0.58
    left_sash = v4.rectangle_mask(size, (0, 255, 720, 950), 16.0) * 0.52
    face_guard = v4.ellipse_mask(size, (680, 0, 1270, 430), 26.0)
    far_background_guard = v4.rectangle_mask(size, (1320, 0, 1920, 520), 18.0)
    area = np.maximum.reduce([torso, lower_torso, weapon_band, left_sash])
    area = area * (1.0 - face_guard * 0.98)
    area = area * (1.0 - far_background_guard * 0.70)
    return np.clip(area, 0.0, 1.0)


def build_clean_strap_alpha(source: np.ndarray, area_mask: np.ndarray) -> np.ndarray:
    red = source[:, :, 0]
    green = source[:, :, 1]
    blue = source[:, :, 2]
    luma = red * 0.299 + green * 0.587 + blue * 0.114
    channel_max = np.maximum.reduce([red, green, blue])
    channel_min = np.minimum.reduce([red, green, blue])
    chroma = channel_max - channel_min

    edges = v4.edge_strength(source)
    dark_candidate = (luma < 96.0) & (red < 122.0) & (green < 98.0) & (blue < 136.0)
    hard_black_edge = (luma < 58.0) & (edges > 0.07)
    decor_edge = dark_candidate & (edges > 0.15)
    red_guard = (red > 132.0) & (red - green > 52.0) & (red - blue > 34.0)
    gold_guard = (red > 150.0) & (green > 96.0) & (blue < 128.0) & (red - blue > 42.0)
    skin_guard = (red > 82.0) & (green > 58.0) & (blue > 66.0) & (red - green < 52.0) & (luma > 66.0) & (edges < 0.22)

    edge_gate = np.clip((edges - 0.06) / 0.30, 0.0, 1.0)
    content = np.where((hard_black_edge | decor_edge) & ~(red_guard | gold_guard | skin_guard), 1.0, 0.0)
    mask = np.clip(content.astype(np.float32) * area_mask * edge_gate, 0.0, 1.0)

    image = Image.fromarray(np.clip(mask * 255, 0, 255).astype(np.uint8), "L")
    width = source.shape[1]
    image = image.filter(ImageFilter.MaxFilter(5))
    image = image.filter(ImageFilter.GaussianBlur(max(0.20, 0.35 * width / OUT_W)))
    return np.asarray(image, dtype=np.float32) / 255.0


def build_strap_halo(alpha_a: np.ndarray, alpha_b: np.ndarray) -> np.ndarray:
    mask = np.maximum(alpha_a, alpha_b)
    image = Image.fromarray(np.clip(mask * 255, 0, 255).astype(np.uint8), "L")
    scale = alpha_a.shape[1] / OUT_W
    image = image.filter(ImageFilter.MaxFilter(max(3, int(round(9 * scale)) | 1)))
    image = image.filter(ImageFilter.GaussianBlur(max(0.45, 1.15 * scale)))
    halo = np.asarray(image, dtype=np.float32) / 255.0
    return np.clip(halo, 0.0, 1.0)


def build_strap_layers(source_a: np.ndarray, source_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    area = build_strap_area_mask((source_a.shape[1], source_a.shape[0]))
    alpha_a = build_clean_strap_alpha(source_a, area)
    alpha_b = build_clean_strap_alpha(source_b, area)
    halo = build_strap_halo(alpha_a, alpha_b)
    return source_a.copy(), source_b.copy(), alpha_a, alpha_b, halo


def apply_strap_overlay(
    frame: np.ndarray,
    layer_a: np.ndarray,
    layer_b: np.ndarray,
    alpha_a: np.ndarray,
    alpha_b: np.ndarray,
    halo: np.ndarray,
    alpha: float,
    opacity: float | None = None,
    wash_strength: float = 0.42,
) -> np.ndarray:
    weight_a, weight_b, phase_opacity = strap_layer_weights(alpha)
    if opacity is None:
        opacity = phase_opacity

    frame_image = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), "RGB")
    softened = np.asarray(frame_image.filter(ImageFilter.GaussianBlur(2.15)), dtype=np.float32)
    wash_gate = np.clip(halo[:, :, None] * wash_strength, 0.0, 0.82)
    washed = frame * (1.0 - wash_gate) + softened * wash_gate

    layer_alpha = np.clip(alpha_a * weight_a + alpha_b * weight_b, 0.0, 1.0)
    alpha_gate = np.clip(layer_alpha[:, :, None] * opacity, 0.0, 1.0)
    denom = np.maximum(layer_alpha[:, :, None], 1.0e-4)
    layer_rgb = (layer_a * (alpha_a[:, :, None] * weight_a) + layer_b * (alpha_b[:, :, None] * weight_b)) / denom
    dark_target = np.minimum(washed, layer_rgb)
    cleaned = washed * (1.0 - alpha_gate) + dark_target * alpha_gate
    return np.clip(cleaned, 0, 255)


def prepared_context(
    overwrite_sources: bool = False,
) -> tuple:
    v6_context = v6.prepared_context(overwrite_sources=overwrite_sources)
    source_a_4k = v6_context[2]
    source_b_4k = v6_context[3]
    strap_layers = build_strap_layers(source_a_4k, source_b_4k)
    return (*v6_context, *strap_layers)


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
    strap_layer_a: np.ndarray,
    strap_layer_b: np.ndarray,
    strap_alpha_a: np.ndarray,
    strap_alpha_b: np.ndarray,
    strap_halo: np.ndarray,
    motion_frames: list[Path],
) -> np.ndarray:
    frame = v6.render_frame_4k(
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
        highpass_cleanup_mask_4k,
        motion_frames,
    )
    alpha = v2.timeline_alpha(index, TARGET_FRAMES)
    return apply_strap_overlay(
        frame,
        strap_layer_a,
        strap_layer_b,
        strap_alpha_a,
        strap_alpha_b,
        strap_halo,
        alpha,
    )


def render_context_frame(context: tuple, motion_frames: list[Path], index: int) -> np.ndarray:
    return render_frame_4k(motion_frames[index], index, *context, motion_frames)


def write_review_artifacts(overwrite_sources: bool = False, review_dir: Path = REVIEW_DIR) -> list[Path]:
    v4.ensure_motion_frames(overwrite_sources=overwrite_sources, overwrite_rife=False)
    motion_frames = sorted(v2.MOTION_FRAME_DIR.glob("*.png"))
    context = prepared_context(overwrite_sources=overwrite_sources)
    review_dir.mkdir(parents=True, exist_ok=True)

    indices = sorted({0, 20, 40, 60, 80, 100, 120, 150, 180, 210, TARGET_FRAMES - 1})
    frame_paths: list[Path] = []
    for index in indices:
        frame = render_context_frame(context, motion_frames, index).astype(np.uint8)
        out = review_dir / f"rife_interpolated_v9_sample_{index:06d}.jpg"
        Image.fromarray(frame, "RGB").save(out, quality=94)
        frame_paths.append(out)

    outputs: list[Path] = list(frame_paths)
    overview = review_dir / "rife_interpolated_v9_overview.jpg"
    make_contact_sheet(frame_paths, overview)
    outputs.append(overview)
    torso = review_dir / "rife_interpolated_v9_torso_weapon_crops.jpg"
    make_crop_sheet(frame_paths, (1160, 430, 2550, 1780), torso)
    outputs.append(torso)
    chest = review_dir / "rife_interpolated_v9_chest_strap_close_crops.jpg"
    make_crop_sheet(frame_paths, (1390, 500, 2350, 1500), chest)
    outputs.append(chest)
    hair = review_dir / "rife_interpolated_v9_right_hair_field_crops.jpg"
    make_crop_sheet(frame_paths, (1810, 260, 3840, 1800), hair)
    outputs.append(hair)
    loop = review_dir / "rife_interpolated_v9_loop_check.jpg"
    write_loop_check(Image.open(frame_paths[0]), Image.open(frame_paths[-1]), loop)
    outputs.append(loop)
    preview = review_dir / "rife_interpolated_v9_preview.jpg"
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
        "chest_strap_close": (1390, 500, 2350, 1500),
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
            print(f"reviewed v9 crop frame {index + 1}/{TARGET_FRAMES}", flush=True)

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
                print(f"streamed v9 4k frame {index + 1}/{TARGET_FRAMES}", flush=True)
    finally:
        proc.stdin.close()
    return_code = proc.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return output


def write_project(video_path: Path, preview_path: Path = REVIEW_DIR / "rife_interpolated_v9_preview.jpg") -> None:
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_path, PROJECT_VIDEO)
    if preview_path.exists():
        shutil.copy2(preview_path, PROJECT_DIR / "preview.jpg")
    project = {
        "title": "Scathach Two Source RIFE Interpolated Loop V9 Strap Overlay",
        "description": "Looping 2160p60 4s wallpaper using the v6 RIFE base plus a clean extracted black strap/decor overlay to suppress faint double-line trails.",
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
            preview = REVIEW_DIR / "rife_interpolated_v9_preview.jpg"
            if not preview.exists():
                write_review_artifacts(overwrite_sources=args.overwrite_sources, review_dir=REVIEW_DIR)
            write_project(video, preview)
        print(video.resolve())


if __name__ == "__main__":
    main()

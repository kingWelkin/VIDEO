from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
PROJECT_DIR = ROOT / "wallpaper_engine_scathach"
SOURCE_ZIP = ROOT / "images_3840x2160.zip"
SOURCE_DIR = OUT_DIR / "two_source_rife_interpolated_sources"
REVIEW_DIR = OUT_DIR / "two_source_rife_interpolated_review"
OUTPUT_VIDEO = OUT_DIR / "scathach_two_source_rife_interpolated_2160p60_6s.mp4"
PROJECT_VIDEO = PROJECT_DIR / OUTPUT_VIDEO.name

TOOLS_DIR = Path("D:/codex_video_tools")
RIFE_DIR = TOOLS_DIR / "rife-ncnn-vulkan-20221029-windows"
RIFE_EXE = RIFE_DIR / "rife-ncnn-vulkan.exe"
RIFE_MODEL = RIFE_DIR / "rife-v4.6"
WORK_DIR = TOOLS_DIR / "scathach_two_source_rife_interpolated_work"
KEYFRAME_DIR = WORK_DIR / "keyframes_1080p"
RIFE_FRAME_DIR = WORK_DIR / "rife_frames_1080p60"
EFFECT_FRAME_DIR = WORK_DIR / "effect_frames_1080p60"

SOURCE_W = 3840
SOURCE_H = 2160
KEY_W = 1920
KEY_H = 1080
OUT_W = 3840
OUT_H = 2160
FPS = 60
SECONDS = 6
TARGET_FRAMES = FPS * SECONDS
CRF = "16"
PRESET = "medium"


def safe_reset_dir(path: Path, allowed_root: Path) -> None:
    resolved = path.resolve()
    allowed = allowed_root.resolve()
    if not str(resolved).lower().startswith(str(allowed).lower()):
        raise RuntimeError(f"refusing to delete outside {allowed}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def mask_float(image: Image.Image, blur: float = 0.0) -> np.ndarray:
    if blur > 0:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def periodic_phase(index: int, total_frames: int = TARGET_FRAMES) -> float:
    if total_frames <= 1:
        return 0.0
    return index / (total_frames - 1)


def pulse(index: int, total_frames: int, cycles: float, offset: float = 0.0) -> float:
    phase = periodic_phase(index, total_frames)
    return (0.5 + 0.5 * math.sin(2.0 * math.pi * (phase * cycles + offset))) ** 1.35


def find_ffmpeg() -> str:
    env_path = os.environ.get("FFMPEG_EXE")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            ROOT / ".codex_deps" / "imageio_ffmpeg" / "binaries" / "ffmpeg-win-x86_64-v7.1.exe",
            ROOT / ".codex_deps" / "imageio_ffmpeg" / "binaries" / "ffmpeg-win-x86_64-v7.0.2.exe",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    located = shutil.which("ffmpeg")
    if located:
        return located
    raise FileNotFoundError("ffmpeg was not found; restore .codex_deps/imageio_ffmpeg or set FFMPEG_EXE")


def extract_sources(zip_path: Path = SOURCE_ZIP, output_dir: Path = SOURCE_DIR, overwrite: bool = False) -> list[Path]:
    if not zip_path.exists():
        raise FileNotFoundError(f"missing source zip: {zip_path}")
    if overwrite:
        safe_reset_dir(output_dir, output_dir.parent)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_path) as zf:
        names = sorted(name for name in zf.namelist() if name.lower().endswith((".png", ".jpg", ".jpeg")))
        if len(names) != 2:
            raise ValueError(f"expected exactly two source images, found {len(names)}")
        paths: list[Path] = []
        for index, name in enumerate(names, 1):
            with zf.open(name) as stream:
                image = Image.open(stream).convert("RGB")
                if image.size != (SOURCE_W, SOURCE_H):
                    raise ValueError(f"{name} must be {SOURCE_W}x{SOURCE_H}, got {image.size}")
                out_path = output_dir / f"source_{index}_3840x2160.png"
                image.save(out_path, compress_level=1)
                paths.append(out_path)
    return paths


def load_source_images(overwrite: bool = False) -> tuple[Image.Image, Image.Image]:
    paths = [
        SOURCE_DIR / "source_1_3840x2160.png",
        SOURCE_DIR / "source_2_3840x2160.png",
    ]
    if overwrite or not all(path.exists() for path in paths):
        paths = extract_sources(SOURCE_ZIP, SOURCE_DIR, overwrite=overwrite)
    return Image.open(paths[0]).convert("RGB"), Image.open(paths[1]).convert("RGB")


def prepare_keyframe(image: Image.Image) -> Image.Image:
    if image.size != (SOURCE_W, SOURCE_H):
        image = image.resize((SOURCE_W, SOURCE_H), Image.Resampling.LANCZOS)
    frame = image.resize((KEY_W, KEY_H), Image.Resampling.LANCZOS)
    frame = ImageEnhance.Color(frame).enhance(1.025)
    frame = ImageEnhance.Contrast(frame).enhance(1.018)
    frame = ImageEnhance.Sharpness(frame).enhance(1.025)
    return frame


def write_keyframes_from_images(source_a: Image.Image, source_b: Image.Image, output_dir: Path = KEYFRAME_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = [prepare_keyframe(source_a), prepare_keyframe(source_b), prepare_keyframe(source_a)]
    paths: list[Path] = []
    for index, frame in enumerate(frames):
        path = output_dir / f"{index:08d}.png"
        frame.save(path, compress_level=1)
        paths.append(path)
    return paths


def write_keyframes(overwrite_sources: bool = False, overwrite_keyframes: bool = True) -> list[Path]:
    if overwrite_keyframes:
        safe_reset_dir(KEYFRAME_DIR, WORK_DIR)
    else:
        KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)
    source_a, source_b = load_source_images(overwrite=overwrite_sources)
    return write_keyframes_from_images(source_a, source_b, KEYFRAME_DIR)


def build_rife_command(input_dir: Path, output_dir: Path, frames: int = TARGET_FRAMES) -> list[str]:
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


def run_rife(frames: int = TARGET_FRAMES, overwrite: bool = False) -> None:
    if not RIFE_EXE.exists():
        raise FileNotFoundError(f"RIFE executable not found: {RIFE_EXE}")
    if not RIFE_MODEL.exists():
        raise FileNotFoundError(f"RIFE model not found: {RIFE_MODEL}")
    existing = sorted(RIFE_FRAME_DIR.glob("*.png")) if RIFE_FRAME_DIR.exists() else []
    if not overwrite and len(existing) >= frames:
        print(f"reuse RIFE frames: {RIFE_FRAME_DIR}", flush=True)
        return
    safe_reset_dir(RIFE_FRAME_DIR, WORK_DIR)
    subprocess.run(build_rife_command(KEYFRAME_DIR, RIFE_FRAME_DIR, frames), check=True)


def source_red_mask(frame: np.ndarray) -> np.ndarray:
    red = frame[:, :, 0]
    green = frame[:, :, 1]
    blue = frame[:, :, 2]
    pixels = ((red > 76.0) & ((red - green) > 24.0) & ((red - blue) > 8.0)).astype(np.uint8) * 255
    return mask_float(Image.fromarray(pixels, "L"), 0.7)


def soft_red_glow(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    core = Image.fromarray((source_red_mask(frame) * 255).astype(np.uint8), "L")
    glow = core.filter(ImageFilter.GaussianBlur(6.5))
    wide = core.filter(ImageFilter.GaussianBlur(15.0))
    return mask_float(glow, 0.0), mask_float(wide, 0.0)


def sharpen_frame(frame: np.ndarray) -> np.ndarray:
    image = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), "RGB")
    soft = np.asarray(image.filter(ImageFilter.GaussianBlur(0.85)), dtype=np.float32)
    return np.clip(frame + (frame - soft) * 0.14, 0, 255)


def postprocess_rife_frame(frame: np.ndarray, index: int, total_frames: int = TARGET_FRAMES) -> np.ndarray:
    result = sharpen_frame(frame.astype(np.float32))
    red_gate = source_red_mask(result)
    red_pulse = 0.03 + 0.11 * pulse(index, total_frames, 2.0, -0.10)
    result[:, :, 0] += red_gate * red_pulse * 10.0
    result[:, :, 1] += red_gate * red_pulse * 0.8
    result[:, :, 2] += red_gate * red_pulse * 2.0

    bg_breath = 0.975 + 0.035 * pulse(index, total_frames, 1.0, -0.18)
    upper = np.zeros((result.shape[0], result.shape[1]), dtype=np.float32)
    upper[: round(result.shape[0] * 0.55), :] = 1.0
    upper = mask_float(Image.fromarray((upper * 255).astype(np.uint8), "L"), 22.0)[:, :, None]
    result = result * (1.0 - upper) + result * bg_breath * upper
    return np.clip(result, 0, 255)


def write_effect_frames(frames: int = TARGET_FRAMES, overwrite: bool = False) -> list[Path]:
    rife_frames = sorted(RIFE_FRAME_DIR.glob("*.png"))
    if len(rife_frames) < frames:
        raise RuntimeError(f"RIFE frame count {len(rife_frames)} < requested {frames}")
    if overwrite:
        safe_reset_dir(EFFECT_FRAME_DIR, WORK_DIR)
    else:
        EFFECT_FRAME_DIR.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for index, path in enumerate(rife_frames[:frames]):
        out_path = EFFECT_FRAME_DIR / f"{index:08d}.png"
        if out_path.exists() and not overwrite:
            outputs.append(out_path)
            continue
        frame = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        processed = postprocess_rife_frame(frame, index, frames)
        Image.fromarray(processed.astype(np.uint8), "RGB").save(out_path, compress_level=1)
        outputs.append(out_path)
        if index % 25 == 0:
            print(f"postprocessed RIFE frame {index + 1}/{frames}", flush=True)
    return outputs


def build_ffmpeg_command(output: Path, input_dir: Path = EFFECT_FRAME_DIR, frames: int = TARGET_FRAMES) -> list[str]:
    return [
        find_ffmpeg(),
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
        f"scale={OUT_W}:{OUT_H}:flags=lanczos,unsharp=5:5:0.20:3:3:0.04,format=yuv420p",
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


def encode_video(output: Path = OUTPUT_VIDEO, frames: int = TARGET_FRAMES) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(build_ffmpeg_command(output, EFFECT_FRAME_DIR, frames), check=True)
    return output


def make_contact_sheet(paths: list[Path], out_path: Path, thumb_size: tuple[int, int] = (384, 216)) -> None:
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
    sheet.save(out_path, quality=92)


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
    sheet.save(out_path, quality=92)


def write_loop_check(first: Image.Image, last: Image.Image, out_path: Path) -> None:
    diff = ImageChops.difference(first.convert("RGB"), last.convert("RGB")).convert("L")
    heat = ImageOps.colorize(ImageOps.autocontrast(diff), black="#101018", white="#ff4058", mid="#6837c7")
    images = [first.convert("RGB"), last.convert("RGB"), heat]
    thumbs: list[Image.Image] = []
    for image in images:
        image.thumbnail((384, 216), Image.Resampling.LANCZOS)
        thumbs.append(image)
    sheet = Image.new("RGB", (384 * 3, 240), (18, 18, 22))
    for index, (label, image) in enumerate(zip(("first", "last", "difference"), thumbs)):
        x = index * 384
        sheet.paste(image, (x + (384 - image.width) // 2, 0))
        ImageDraw.Draw(sheet).text((x + 8, 222), label, fill=(235, 235, 235))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def write_review_artifacts(frames: int = TARGET_FRAMES, review_dir: Path = REVIEW_DIR) -> list[Path]:
    review_dir.mkdir(parents=True, exist_ok=True)
    sample_count = 9 if frames >= 180 else 5
    indices = sorted({round(i * (frames - 1) / (sample_count - 1)) for i in range(sample_count)})
    frame_paths: list[Path] = []
    for index in indices:
        src = EFFECT_FRAME_DIR / f"{index:08d}.png"
        if not src.exists():
            continue
        image = Image.open(src).convert("RGB")
        out = review_dir / f"rife_interpolated_sample_{index:06d}.jpg"
        image.save(out, quality=93)
        frame_paths.append(out)
    outputs: list[Path] = list(frame_paths)
    if frame_paths:
        overview = review_dir / "rife_interpolated_overview.jpg"
        make_contact_sheet(frame_paths, overview)
        outputs.append(overview)
        face = review_dir / "rife_interpolated_face_crops.jpg"
        make_crop_sheet(frame_paths, (780, 85, 1168, 355), face)
        outputs.append(face)
        flower = review_dir / "rife_interpolated_flower_crops.jpg"
        make_crop_sheet(frame_paths, (0, 520, 1920, 1080), flower)
        outputs.append(flower)
        loop = review_dir / "rife_interpolated_loop_check.jpg"
        write_loop_check(Image.open(frame_paths[0]), Image.open(frame_paths[-1]), loop)
        outputs.append(loop)
        preview = review_dir / "rife_interpolated_preview.jpg"
        shutil.copy2(frame_paths[0], preview)
        outputs.append(preview)
    return outputs


def write_project(video_path: Path, preview_path: Path = REVIEW_DIR / "rife_interpolated_preview.jpg") -> None:
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_path, PROJECT_VIDEO)
    if preview_path.exists():
        shutil.copy2(preview_path, PROJECT_DIR / "preview.jpg")
    project = {
        "title": "Scathach Two Source RIFE Interpolated Loop",
        "description": "Looping 2160p60 6s wallpaper using dense RIFE interpolation between two 4K source frames, with A-B-A keyframes, animated expression transition, source-gated red glow, and no artificial red bands.",
        "type": "video",
        "file": PROJECT_VIDEO.name,
        "preview": "preview.jpg",
        "tags": ["Anime"],
        "visibility": "private",
    }
    (PROJECT_DIR / "project.json").write_text(json.dumps(project, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["keyframes", "rife", "effects", "review", "video", "all"], default="all")
    parser.add_argument("--frames", type=int, default=TARGET_FRAMES)
    parser.add_argument("--overwrite-sources", action="store_true")
    parser.add_argument("--overwrite-rife", action="store_true")
    parser.add_argument("--overwrite-effects", action="store_true")
    parser.add_argument("--no-project", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {"keyframes", "rife", "effects", "review", "video", "all"}:
        keyframes = write_keyframes(overwrite_sources=args.overwrite_sources, overwrite_keyframes=True)
        print(f"keyframes: {KEYFRAME_DIR} ({len(keyframes)})")
    if args.mode in {"rife", "effects", "review", "video", "all"}:
        run_rife(args.frames, overwrite=args.overwrite_rife)
        print(f"RIFE frames: {RIFE_FRAME_DIR}")
    if args.mode in {"effects", "review", "video", "all"}:
        effect_paths = write_effect_frames(args.frames, overwrite=args.overwrite_effects)
        print(f"effect frames: {EFFECT_FRAME_DIR} ({len(effect_paths)})")
    if args.mode in {"review", "all"}:
        review = write_review_artifacts(args.frames, REVIEW_DIR)
        print(f"review artifacts: {REVIEW_DIR} ({len(review)})")
    if args.mode in {"video", "all"}:
        video = encode_video(OUTPUT_VIDEO, args.frames)
        if not args.no_project and args.frames == TARGET_FRAMES:
            preview = REVIEW_DIR / "rife_interpolated_preview.jpg"
            if not preview.exists():
                write_review_artifacts(args.frames, REVIEW_DIR)
            write_project(video, preview)
        print(video.resolve())


if __name__ == "__main__":
    main()

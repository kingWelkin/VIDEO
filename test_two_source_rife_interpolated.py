from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

import make_scathach_wallpaper_two_source_rife_interpolated as gen


class TwoSourceRifeInterpolatedTests(unittest.TestCase):
    def test_timeline_keyframes_are_source_a_source_b_source_a(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_a = Image.new("RGB", (3840, 2160), (10, 20, 30))
            source_b = Image.new("RGB", (3840, 2160), (80, 40, 20))
            output_dir = root / "keys"

            paths = gen.write_keyframes_from_images(source_a, source_b, output_dir)

            self.assertEqual([path.name for path in paths], ["00000000.png", "00000001.png", "00000002.png"])
            first = Image.open(paths[0]).convert("RGB")
            middle = Image.open(paths[1]).convert("RGB")
            last = Image.open(paths[2]).convert("RGB")
            self.assertEqual(first.size, (1920, 1080))
            self.assertEqual(middle.size, (1920, 1080))
            self.assertEqual(last.size, (1920, 1080))
            self.assertEqual(first.getpixel((20, 20)), last.getpixel((20, 20)))
            self.assertNotEqual(first.getpixel((20, 20)), middle.getpixel((20, 20)))

    def test_rife_command_requests_dense_6_second_loop(self) -> None:
        command = gen.build_rife_command(Path("keys"), Path("rife"), frames=gen.TARGET_FRAMES)

        self.assertEqual(command[0], str(gen.RIFE_EXE))
        self.assertIn("-n", command)
        self.assertEqual(command[command.index("-n") + 1], "360")
        self.assertEqual(command[command.index("-i") + 1], "keys")
        self.assertEqual(command[command.index("-o") + 1], "rife")
        self.assertIn(str(gen.RIFE_MODEL), command)

    def test_source_red_masks_do_not_authorize_artificial_red_bands(self) -> None:
        plain = np.full((1080, 1920, 3), (24, 24, 30), dtype=np.float32)
        red = gen.source_red_mask(plain)

        self.assertLess(float(red.max()), 0.01)

    def test_postprocess_keeps_plain_nonred_frame_without_red_streaks(self) -> None:
        plain = np.full((1080, 1920, 3), (24, 24, 30), dtype=np.float32)
        processed = gen.postprocess_rife_frame(plain, index=300, total_frames=600)
        red_excess = processed[:, :, 0] - np.maximum(processed[:, :, 1], processed[:, :, 2])

        self.assertLess(float(red_excess.max()), 4.0)

    def test_postprocess_does_not_bleed_red_glow_outside_source_red_regions(self) -> None:
        frame = np.full((1080, 1920, 3), (24, 24, 30), dtype=np.float32)
        frame[880:940, 900:1020, :] = (180, 18, 24)

        processed = gen.postprocess_rife_frame(frame, index=180, total_frames=600)
        red_source = gen.source_red_mask(frame) > 0.01
        red_gain = processed[:, :, 0] - frame[:, :, 0]

        self.assertLess(float(red_gain[~red_source].max()), 3.0)

    def test_output_paths_target_rife_interpolated_wallpaper(self) -> None:
        self.assertEqual(gen.TARGET_FRAMES, 360)
        self.assertEqual(gen.OUTPUT_VIDEO.name, "scathach_two_source_rife_interpolated_2160p60_6s.mp4")
        self.assertEqual(gen.PROJECT_VIDEO.name, "scathach_two_source_rife_interpolated_2160p60_6s.mp4")

    def test_extract_sources_from_zip_requires_two_4k_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = root / "images.zip"
            with ZipFile(zip_path, "w") as zf:
                for index, color in enumerate(((1, 2, 3), (4, 5, 6)), 1):
                    image_path = root / f"image_{index}.png"
                    Image.new("RGB", (3840, 2160), color).save(image_path)
                    zf.write(image_path, f"image_{index}_3840x2160.png")

            paths = gen.extract_sources(zip_path, root / "sources", overwrite=True)

            self.assertEqual(len(paths), 2)
            with Image.open(paths[0]) as first, Image.open(paths[1]) as second:
                self.assertEqual(first.size, (3840, 2160))
                self.assertEqual(second.size, (3840, 2160))

    def test_review_artifacts_sample_short_previews_by_timeline_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            effects = root / "effects"
            review = root / "review"
            effects.mkdir()
            old_effect_dir = gen.EFFECT_FRAME_DIR
            gen.EFFECT_FRAME_DIR = effects
            try:
                for index in (0, 30, 60, 89, 119):
                    Image.new("RGB", (1920, 1080), (index % 255, 20, 30)).save(effects / f"{index:08d}.png")

                outputs = gen.write_review_artifacts(frames=120, review_dir=review)

                names = {path.name for path in outputs}
                self.assertIn("rife_interpolated_sample_000060.jpg", names)
                self.assertIn("rife_interpolated_overview.jpg", names)
            finally:
                gen.EFFECT_FRAME_DIR = old_effect_dir


if __name__ == "__main__":
    unittest.main()

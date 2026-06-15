from __future__ import annotations

import unittest

import numpy as np

import make_scathach_wallpaper_two_source_rife_interpolated_v4 as gen


class TwoSourceRifeInterpolatedV4Tests(unittest.TestCase):
    def test_v4_uses_4k_plate_output_name(self) -> None:
        self.assertEqual(gen.TARGET_FRAMES, 240)
        self.assertEqual(gen.OUTPUT_VIDEO.name, "scathach_two_source_rife_interpolated_2160p60_4s_v4_plate.mp4")
        self.assertEqual(gen.PROJECT_VIDEO.name, "scathach_two_source_rife_interpolated_2160p60_4s_v4_plate.mp4")

    def test_scaled_problem_mask_targets_lower_right_and_excludes_face(self) -> None:
        mask = gen.build_problem_detail_mask_4k((3840, 2160))

        self.assertGreater(float(mask[1520, 2760]), 0.70)
        self.assertGreater(float(mask[1880, 3180]), 0.70)
        self.assertLess(float(mask[340, 1920]), 0.05)
        self.assertLess(float(mask[960, 1220]), 0.35)

    def test_apply_4k_detail_plate_increases_masked_detail_only(self) -> None:
        frame = np.full((80, 120, 3), 90.0, dtype=np.float32)
        high_a = np.zeros_like(frame)
        high_b = np.zeros_like(frame)
        high_a[35:, 60::2, :] = 36.0
        high_a[35:, 61::2, :] = -36.0
        mask = np.zeros((80, 120), dtype=np.float32)
        mask[35:, 60:] = 1.0

        enhanced = gen.apply_4k_detail_plate(frame, high_a, high_b, mask, alpha=0.0, amount=0.8)

        self.assertGreater(float(enhanced[35:, 60:, :].std()), 10.0)
        self.assertAlmostEqual(float(enhanced[:20, :40, :].std()), 0.0, places=5)

    def test_source_edge_mask_finds_thin_edges_without_flat_areas(self) -> None:
        source_a = np.full((80, 120, 3), 20.0, dtype=np.float32)
        source_b = source_a.copy()
        source_a[:, 70:72, :] = 220.0
        source_b[35:37, :, :] = 220.0
        problem_mask = np.zeros((80, 120), dtype=np.float32)

        edge_mask = gen.build_source_edge_mask(source_a, source_b, problem_mask)

        self.assertGreater(float(edge_mask[40, 70]), 0.45)
        self.assertGreater(float(edge_mask[35, 80]), 0.45)
        self.assertLess(float(edge_mask[10, 10]), 0.05)

    def test_flower_light_weight_alternates_dark_and_bright(self) -> None:
        self.assertLess(gen.flower_light_weight(0, total_frames=240), -0.80)
        self.assertGreater(gen.flower_light_weight(60, total_frames=240), 0.80)
        self.assertLess(gen.flower_light_weight(120, total_frames=240), -0.80)

    def test_apply_flower_lighting_brightens_and_darks_masked_region_only(self) -> None:
        frame = np.full((16, 16, 3), 100.0, dtype=np.float32)
        mask = np.zeros((16, 16), dtype=np.float32)
        mask[8:, :] = 1.0

        bright = gen.apply_flower_lighting(frame, mask, index=60, total_frames=240)
        dark = gen.apply_flower_lighting(frame, mask, index=0, total_frames=240)

        self.assertGreater(float(bright[12, 8, 0]), 112.0)
        self.assertLess(float(dark[12, 8, 0]), 92.0)
        self.assertAlmostEqual(float(bright[2, 2, 0]), 100.0, places=5)

    def test_apply_edge_deghost_is_mask_limited(self) -> None:
        frame = np.full((16, 16, 3), 100.0, dtype=np.float32)
        source_a = frame.copy()
        source_b = frame.copy()
        source_a[:, 8, :] = 10.0
        source_b[:, 8, :] = 180.0
        mask = np.zeros((16, 16), dtype=np.float32)
        mask[:, 7:10] = 1.0

        deghosted = gen.apply_edge_deghost(frame, source_a, source_b, mask, alpha=0.2)

        self.assertLess(float(deghosted[8, 8, 0]), 90.0)
        self.assertAlmostEqual(float(deghosted[2, 2, 0]), 100.0, places=5)


if __name__ == "__main__":
    unittest.main()

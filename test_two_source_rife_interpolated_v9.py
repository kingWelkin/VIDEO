from __future__ import annotations

import unittest

import numpy as np

import make_scathach_wallpaper_two_source_rife_interpolated_v9 as gen


class TwoSourceRifeInterpolatedV9Tests(unittest.TestCase):
    def test_v9_uses_strap_overlay_output_name(self) -> None:
        self.assertEqual(gen.OUTPUT_VIDEO.name, "scathach_two_source_rife_interpolated_2160p60_4s_v9_strap_overlay.mp4")
        self.assertEqual(gen.PROJECT_VIDEO.name, "scathach_two_source_rife_interpolated_2160p60_4s_v9_strap_overlay.mp4")

    def test_clean_strap_alpha_keeps_dark_straps_and_excludes_red_and_gold(self) -> None:
        source = np.full((80, 120, 3), 135.0, dtype=np.float32)
        source[20:65, 50:54, :] = 18.0
        source[30:34, 20:105, :] = np.array([214.0, 18.0, 28.0], dtype=np.float32)
        source[54:60, 82:88, :] = np.array([211.0, 156.0, 73.0], dtype=np.float32)
        area = np.zeros((80, 120), dtype=np.float32)
        area[10:72, 12:110] = 1.0

        alpha = gen.build_clean_strap_alpha(source, area)

        self.assertGreater(float(alpha[44, 52]), 0.72)
        self.assertLess(float(alpha[32, 60]), 0.14)
        self.assertLess(float(alpha[56, 85]), 0.14)
        self.assertLess(float(alpha[44, 4]), 0.04)

    def test_clean_strap_alpha_excludes_smooth_dark_body_shading(self) -> None:
        source = np.full((64, 96, 3), np.array([96.0, 70.0, 85.0], dtype=np.float32), dtype=np.float32)
        source[16:52, 44:48, :] = 18.0
        area = np.ones((64, 96), dtype=np.float32)

        alpha = gen.build_clean_strap_alpha(source, area)

        self.assertGreater(float(alpha[34, 46]), 0.72)
        self.assertLess(float(alpha[34, 24]), 0.10)

    def test_strap_layer_weights_crossfade_only_at_mid_source_switch(self) -> None:
        early = gen.strap_layer_weights(alpha=0.25)
        middle = gen.strap_layer_weights(alpha=0.50)
        late = gen.strap_layer_weights(alpha=0.75)

        self.assertGreater(early[0], 0.93)
        self.assertLess(early[1], 0.07)
        self.assertAlmostEqual(middle[0], middle[1], delta=0.02)
        self.assertLess(middle[2], early[2])
        self.assertGreater(late[1], 0.93)
        self.assertLess(late[0], 0.07)

    def test_apply_strap_overlay_washes_nearby_ghost_and_redraws_clean_dark_line(self) -> None:
        frame = np.full((48, 48, 3), 132.0, dtype=np.float32)
        frame[:, 15:17, :] = 76.0
        frame[:, 24:26, :] = 82.0
        layer_a = np.full_like(frame, 132.0)
        layer_b = np.full_like(frame, 132.0)
        layer_a[:, 14:16, :] = 18.0
        layer_b[:, 31:33, :] = 18.0
        alpha_a = np.zeros((48, 48), dtype=np.float32)
        alpha_b = np.zeros((48, 48), dtype=np.float32)
        alpha_a[:, 14:16] = 1.0
        alpha_b[:, 31:33] = 1.0
        halo = np.zeros((48, 48), dtype=np.float32)
        halo[:, 12:28] = 1.0

        cleaned = gen.apply_strap_overlay(
            frame,
            layer_a,
            layer_b,
            alpha_a,
            alpha_b,
            halo,
            alpha=0.25,
            opacity=1.0,
            wash_strength=1.0,
        )

        self.assertLess(float(cleaned[24, 15, 0]), 34.0)
        self.assertGreater(float(cleaned[24, 25, 0]), 104.0)
        self.assertAlmostEqual(float(cleaned[24, 40, 0]), 132.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()

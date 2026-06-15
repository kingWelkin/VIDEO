from __future__ import annotations

import unittest

import numpy as np

import make_scathach_wallpaper_two_source_rife_interpolated_v3 as gen


class TwoSourceRifeInterpolatedV3Tests(unittest.TestCase):
    def test_v3_keeps_short_loop_and_uses_new_output_name(self) -> None:
        self.assertEqual(gen.TARGET_FRAMES, 240)
        self.assertEqual(gen.OUTPUT_VIDEO.name, "scathach_two_source_rife_interpolated_2160p60_4s_v3.mp4")
        self.assertEqual(gen.PROJECT_VIDEO.name, "scathach_two_source_rife_interpolated_2160p60_4s_v3.mp4")

    def test_problem_detail_mask_targets_lower_right_without_face(self) -> None:
        mask = gen.build_problem_detail_mask((1920, 1080))

        self.assertGreater(float(mask[760, 1380]), 0.70)
        self.assertGreater(float(mask[920, 1580]), 0.70)
        self.assertLess(float(mask[175, 960]), 0.05)

    def test_enhance_problem_detail_increases_masked_contrast_only(self) -> None:
        frame = np.full((72, 96, 3), 80.0, dtype=np.float32)
        reference = frame.copy()
        reference[30:, 48::2, :] = 125.0
        reference[30:, 49::2, :] = 35.0
        mask = np.zeros((72, 96), dtype=np.float32)
        mask[30:, 48:] = 1.0

        enhanced = gen.enhance_problem_detail(frame, reference, mask)

        self.assertGreater(float(enhanced[30:, 48:, :].std()), 8.0)
        self.assertAlmostEqual(float(enhanced[:20, :30, :].std()), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()

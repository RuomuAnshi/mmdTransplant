"""Synthetic color-matching tests: no external images, names or Blender."""
import importlib.util
import math
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "skin_color", Path(__file__).resolve().parents[1] / "mmd_transplant/skin_color.py")
color = importlib.util.module_from_spec(spec)
spec.loader.exec_module(color)


def repeated(rgb, count=16):
    return [tuple(rgb) + (1.0,)] * count


class SkinColorTests(unittest.TestCase):
    def test_bridge_basis_can_represent_a_brighter_head_without_clipping(self):
        head, body = (.8,.6,.5,1), (.7,.5,.4,1)
        basis = color.bridge_diffuse_basis(head, body)
        for source in (head, body):
            encoded = tuple(source[i]/basis[i] for i in range(3))
            self.assertTrue(all(0 <= c <= 1 for c in encoded))
            self.assertRGBAlmostEqual(tuple(encoded[i]*basis[i] for i in range(3)), source[:3])
        with self.assertRaises(ValueError):
            color.bridge_diffuse_basis(head, (0,.5,.4,1))

    def assertRGBAlmostEqual(self, actual, expected, places=8):
        for first, second in zip(actual, expected):
            self.assertAlmostEqual(first, second, places=places)

    def test_srgb_conversion_round_trip_and_thresholds(self):
        for rgb in ((0, 0, 0), (1, 1, 1), (0.02, 0.5, 0.9), (0.04045,) * 3):
            self.assertRGBAlmostEqual(color.linear_to_srgb(color.srgb_to_linear(rgb)),
                                      rgb, places=6)
        self.assertAlmostEqual(color.srgb_to_linear((0.5,) * 3)[0],
                               0.21404114048223255)
        self.assertAlmostEqual(color.srgb_to_linear((0.02,) * 3)[0], 0.02 / 12.92)

    def test_bilinear_uses_bottom_up_pixel_centers(self):
        # Bottom row: red, green. Top row: blue, white.
        pixels = [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1]
        self.assertEqual(color.sample_bilinear_rgba(pixels, 2, 2, (0.25, 0.25)),
                         (1, 0, 0, 1))
        self.assertEqual(color.sample_bilinear_rgba(pixels, 2, 2, (0.25, 0.75)),
                         (0, 0, 1, 1))
        self.assertEqual(color.sample_bilinear_rgba(pixels, 2, 2, (0.5, 0.5)),
                         (0.5, 0.5, 0.5, 1))

    def test_bilinear_repeat_wrap_matches_seams_and_negative_coordinates(self):
        pixels = [1, 0, 0, 1, 0, 1, 0, 0]
        self.assertRGBAlmostEqual(color.sample_bilinear_rgba(pixels, 2, 1, (0, 0)),
                                  (0.5, 0.5, 0))
        self.assertEqual(color.sample_bilinear_rgba(pixels, 2, 1, (0, 0)),
                         color.sample_bilinear_rgba(pixels, 2, 1, (1, 1)))
        self.assertEqual(color.sample_bilinear_rgba(pixels, 2, 1, (-0.75, -0.5)),
                         color.sample_bilinear_rgba(pixels, 2, 1, (0.25, 0.5)))
        self.assertAlmostEqual(color.sample_bilinear_rgba(pixels, 2, 1, (0, 0))[3], 0.5)

    def test_rgb_buffer_is_opaque_and_one_pixel_repeats(self):
        self.assertEqual(color.sample_bilinear_rgba([0.1, 0.2, 0.3], 1, 1,
                                                   (5.2, -9.3), channels=3),
                         (0.1, 0.2, 0.3, 1.0))

    def test_sampling_refuses_invalid_buffer_or_uv(self):
        for args in (([0] * 4, 0, 1, (0, 0)), ([0] * 3, 1, 1, (0, 0)),
                     ([0] * 4, 1, 1, (math.nan, 0))):
            with self.assertRaises(ValueError):
                color.sample_bilinear_rgba(*args)

    def test_median_filters_transparent_and_isolated_color_outliers(self):
        base = (0.45, 0.25, 0.18)
        samples = repeated(base, 20)
        samples += [(0, 1, 0, 0), (1, 0, 1, 0.4), (0.98, 0.98, 0.98, 1),
                    (0, 0, 0, 1), (math.nan, 0, 0, 1)]
        result = color.estimate_base_color(samples)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["opaque_count"], 22)
        self.assertEqual(result["inlier_count"], 20)
        self.assertRGBAlmostEqual(result["rgb_linear"], base)
        self.assertGreater(result["confidence"], 0.75)

    def test_untrusted_samples_refuse_color_matching(self):
        target = repeated((0.6, 0.4, 0.3))
        cases = [repeated((0.4, 0.2, 0.1), 2),
                 [(0.2, 0.2, 0.2, 0)] * 16,
                 repeated((0.1, 0.1, 0.1), 8) + repeated((0.9, 0.9, 0.9), 8),
                 repeated((0.4, 0.2, 0.1), 8) + [(0, 0, 0, 0)] * 16,
                 repeated((0.001, 0.001, 0.001))]
        for source in cases:
            result = color.match_neck_color(source, target)
            self.assertEqual(result["status"], "skipped", result)
            self.assertEqual(result["gain"], (1, 1, 1))
            self.assertTrue(result["reason"].startswith("source_"))

    def test_dark_base_is_unreliable_and_uniform_thirty_two_samples_pass(self):
        dark = color.estimate_base_color(repeated((0.001, 0.001, 0.001), 32))
        self.assertEqual(dark["status"], "unreliable")
        self.assertEqual(dark["reason"], "near_black_samples")
        solid = color.estimate_base_color(repeated((0.4, 0.23, 0.14), 32))
        self.assertEqual(solid["status"], "ok")
        self.assertEqual(solid["confidence"], 1.0)

    def test_gradient_has_exact_endpoints_midpoint_and_opaque_alpha(self):
        head, body = (0.8, 0.65, 0.5), (0.4, 0.2, 0.1)
        pixels = color.gradient_rgba(head, body, width=2, height=5)
        self.assertEqual(len(pixels), 2 * 5 * 4)
        self.assertRGBAlmostEqual(pixels[:3], body)
        self.assertRGBAlmostEqual(pixels[-4:-1], head)
        self.assertRGBAlmostEqual(pixels[2 * 2 * 4:2 * 2 * 4 + 3],
                                  tuple((a + b) / 2 for a, b in zip(head, body)))
        self.assertEqual(pixels[3::4], [1.0] * 10)
        self.assertTrue(all(0 <= channel <= 1 for channel in pixels))
        for row in range(5):
            self.assertEqual(pixels[row * 8:row * 8 + 4], pixels[row * 8 + 4:row * 8 + 8])

    def test_gradient_strength_zero_and_partial_blend(self):
        head, body = (1, 0, 0.5), (0, 1, 0.25)
        disabled = color.gradient_rgba(head, body, width=1, height=5, strength=0)
        self.assertEqual(disabled, list(body + (1.0,)) * 5)
        partial = color.gradient_rgba(head, body, width=1, height=3, strength=0.5)
        self.assertRGBAlmostEqual(partial[-4:-1], (0.5, 0.5, 0.375))
        self.assertRGBAlmostEqual(partial[4:7], (0.25, 0.75, 0.3125))

    def test_gradient_smoothstep_stays_between_colors_and_refuses_invalid_input(self):
        head, body = (1, 0, 1), (0, 1, 0)
        pixels = color.gradient_rgba(head, body)
        self.assertTrue(all(0 <= channel <= 1 for channel in pixels))
        self.assertLess(pixels[16 * 4], 1 / 127)
        for kwargs in ({"width": 0}, {"height": 1}, {"width": 1.5},
                       {"strength": math.nan}, {"strength": -1}):
            with self.assertRaises(ValueError):
                color.gradient_rgba(head, body, **kwargs)
        with self.assertRaises(ValueError):
            color.gradient_rgba((math.inf, 0, 0), body)

    def test_same_color_does_not_drift(self):
        for rgb in ((0.3, 0.2, 0.1), (0.8, 0.65, 0.5)):
            result = color.match_neck_color(repeated(rgb), repeated(rgb))
            self.assertEqual(result["status"], "unchanged")
            self.assertEqual(result["gain"], (1, 1, 1))

    def test_strength_zero_disables_without_sampling(self):
        result = color.match_neck_color(None, None, strength=0)
        self.assertEqual(result["status"], "disabled")
        self.assertEqual(result["gain"], (1, 1, 1))

    def test_linear_gain_maps_base_and_preserves_relative_texture_detail(self):
        source, target = (0.4, 0.25, 0.2), (0.48, 0.3, 0.24)
        result = color.match_neck_color(repeated(source), repeated(target))
        self.assertEqual(result["status"], "matched")
        self.assertFalse(result["limited"])
        self.assertRGBAlmostEqual(result["gain"], (1.2, 1.2, 1.2))
        corrected = tuple(a * gain for a, gain in zip(source, result["gain"]))
        detail = tuple(a * 0.8 * gain for a, gain in zip(source, result["gain"]))
        self.assertRGBAlmostEqual(corrected, target)
        self.assertRGBAlmostEqual(detail, tuple(a * 0.8 for a in target))
        partial = color.match_neck_color(repeated(source), repeated(target), strength=0.5)
        self.assertRGBAlmostEqual(partial["gain"], (1.1, 1.1, 1.1))

    def test_extreme_gain_is_limited_before_strength(self):
        result = color.match_neck_color(repeated((0.1, 0.8, 0.3)),
                                       repeated((0.9, 0.1, 0.3)), strength=0.5)
        self.assertTrue(result["limited"])
        self.assertEqual(result["reason"], "gain_limited")
        self.assertRGBAlmostEqual(result["gain"], (1.25, 5 / 6, 1))
        self.assertTrue(all(math.isfinite(value) for value in result["gain"]))

    def test_srgb_samples_are_matched_in_linear_space(self):
        source, target = (0.6, 0.5, 0.4), (0.65, 0.53, 0.43)
        result = color.match_neck_color(repeated(source), repeated(target), color_space="srgb")
        source_linear = color.srgb_to_linear(source)
        corrected = tuple(a * gain for a, gain in zip(source_linear, result["gain"]))
        self.assertRGBAlmostEqual(color.linear_to_srgb(corrected), target)

    def test_invalid_configuration_is_rejected(self):
        samples = repeated((0.3, 0.2, 0.1))
        for kwargs in ({"strength": math.nan}, {"strength": 1.1}, {"min_gain": 0},
                       {"max_gain": 0.9}, {"color_space": "unknown"}, {"min_samples": 2},
                       {"alpha_threshold": -0.1}):
            with self.assertRaises(ValueError):
                color.match_neck_color(samples, samples, **kwargs)


if __name__ == "__main__":
    unittest.main()

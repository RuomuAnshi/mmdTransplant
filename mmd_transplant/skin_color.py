"""Conservative color matching for a small neck transition, without Blender.

Image samples are interpreted as scene-linear RGB(A) by default. Pixel buffers,
including Blender image pixels, must instead be interpreted according to their
own color-space metadata; an sRGB byte image requires ``color_space="srgb"``.
The returned gain and generated gradient buffer are linear. A caller saving the
gradient to an sRGB PNG must encode it accordingly. Corrections are intended for
the local neck transition, not an entire face or a shared source image.
"""
import math
from statistics import median


def bridge_diffuse_basis(head, body):
    """Give a derived strip enough color range for both source materials."""
    if len(head) != 4 or len(body) != 4 or not all(math.isfinite(c) and 0 <= c <= 1 for c in tuple(head)+tuple(body)):
        raise ValueError('皮肤材质颜色超出支持范围')
    if min(body[:3]) < .05:
        raise ValueError('身体材质颜色过暗，无法安全匹配')
    return tuple(max(a, b) for a, b in zip(head[:3], body[:3])) + (body[3],)


def srgb_to_linear(rgb):
    """Convert three normalized sRGB channels to scene-linear RGB."""
    return tuple(value / 12.92 if value <= 0.04045
                 else ((value + 0.055) / 1.055) ** 2.4 for value in rgb[:3])


def linear_to_srgb(rgb):
    """Convert three nonnegative scene-linear channels to normalized sRGB."""
    return tuple(value * 12.92 if value <= 0.0031308
                 else 1.055 * value ** (1.0 / 2.4) - 0.055 for value in rgb[:3])


def sample_bilinear_rgba(pixels, width, height, uv, channels=4):
    """Sample a flat, bottom-up pixel buffer with repeat wrapping.

    Pixel centers lie at ``((x + .5) / width, (y + .5) / height)``. This matches
    repeating texture interpolation at the UV boundary rather than pinning UV
    0 and 1 to different edge pixels. RGB buffers receive an opaque alpha.
    Values and alpha are interpolated in the color space of the input buffer.
    """
    if width <= 0 or height <= 0 or channels not in (3, 4):
        raise ValueError("Image dimensions and channels are invalid")
    if len(pixels) != width * height * channels:
        raise ValueError("Image pixel buffer length is inconsistent")
    if len(uv) != 2 or not all(math.isfinite(value) for value in uv):
        raise ValueError("Texture coordinate must contain two finite values")
    x, y = (uv[0] % 1.0) * width - 0.5, (uv[1] % 1.0) * height - 0.5
    left, bottom = math.floor(x), math.floor(y)
    fx, fy = x - left, y - bottom
    result = [0.0] * 4
    for row, row_weight in ((bottom, 1.0 - fy), (bottom + 1, fy)):
        for column, column_weight in ((left, 1.0 - fx), (left + 1, fx)):
            offset = ((row % height) * width + column % width) * channels
            weight = row_weight * column_weight
            for channel in range(channels):
                result[channel] += pixels[offset + channel] * weight
            if channels == 3:
                result[3] += weight
    return tuple(result)


def _rms_distance(first, second):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)) / 3.0)


def estimate_base_color(samples, *, color_space="linear", min_samples=8,
                        alpha_threshold=0.9):
    """Estimate a reliable base color using an opaque, robust median cluster.

    Transparent, nonfinite and out-of-range samples are discarded. A median
    absolute-deviation filter rejects isolated color outliers; wide or poorly
    supported clusters are rejected rather than guessing a correction. The
    result contains no asset identifiers and does not mutate the samples.
    """
    if color_space not in ("linear", "srgb"):
        raise ValueError("Color space must be linear or srgb")
    if not isinstance(min_samples, int) or min_samples < 3:
        raise ValueError("At least three color samples are required")
    if not math.isfinite(alpha_threshold) or not 0 <= alpha_threshold <= 1:
        raise ValueError("Alpha threshold must be between zero and one")
    samples = list(samples)
    opaque = []
    for sample in samples:
        if len(sample) not in (3, 4):
            continue
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in sample):
            continue
        if len(sample) == 4 and sample[3] < alpha_threshold:
            continue
        rgb = tuple(sample[:3])
        opaque.append(srgb_to_linear(rgb) if color_space == "srgb" else rgb)
    result = {"status": "unreliable", "reason": "insufficient_opaque_samples",
              "rgb_linear": None, "rgb_srgb": None, "confidence": 0.0,
              "sample_count": len(samples), "opaque_count": len(opaque),
              "inlier_count": 0, "spread": None}
    if len(opaque) < min_samples:
        return result
    center = tuple(median(rgb[i] for rgb in opaque) for i in range(3))
    distances = [_rms_distance(rgb, center) for rgb in opaque]
    middle_distance = median(distances)
    deviation = median(abs(distance - middle_distance) for distance in distances)
    cutoff = max(0.035, middle_distance + 3.5 * 1.4826 * deviation)
    inliers = [rgb for rgb, distance in zip(opaque, distances) if distance <= cutoff]
    result["inlier_count"] = len(inliers)
    if len(inliers) < min_samples:
        result["reason"] = "insufficient_inliers"
        return result
    center = tuple(median(rgb[i] for rgb in inliers) for i in range(3))
    ordered_distances = sorted(_rms_distance(rgb, center) for rgb in inliers)
    spread = ordered_distances[math.ceil(len(ordered_distances) * 0.9) - 1]
    result["spread"] = spread
    support = len(inliers) / max(1, len(samples))
    cluster_support = len(inliers) / len(opaque)
    # The spread threshold is scene-linear: a broad mix of skin, fabric and
    # texture highlights must not drive a guessed tint.
    if spread > 0.12:
        result["reason"] = "heterogeneous_samples"
        return result
    confidence = min(1.0, support, cluster_support, max(0.0, 1.0 - spread / 0.24))
    result["confidence"] = confidence
    if confidence < 0.55:
        result["reason"] = "weak_sample_support"
        return result
    luminance = sum(value * weight for value, weight in zip(center, (0.2126, 0.7152, 0.0722)))
    if luminance < 0.015:
        result.update(reason="near_black_samples", confidence=0.0)
        return result
    result.update(status="ok", reason="stable_color_cluster", rgb_linear=center,
                  rgb_srgb=linear_to_srgb(center))
    return result


def gradient_rgba(head_linear, body_linear, width=16, height=128, strength=1.0):
    """Create an opaque, bottom-up linear RGBA neck-transition pixel buffer.

    The bottom row is exactly the body color. The top row blends from body to
    head according to strength, reaching exactly the head color at strength 1.
    A smoothstep between the rows has no endpoint overshoot. This independent
    texture is suitable for a new bridge material; no source pixels are edited.
    """
    if (not isinstance(width, int) or not isinstance(height, int)
            or width < 1 or height < 2):
        raise ValueError("Gradient dimensions require positive width and at least two rows")
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Gradient strength must be between zero and one")
    if (len(head_linear) != 3 or len(body_linear) != 3
            or not all(math.isfinite(value) and 0 <= value <= 1
                       for value in tuple(head_linear) + tuple(body_linear))):
        raise ValueError("Gradient colors must contain three normalized linear channels")
    pixels = []
    for row in range(height):
        t = row / (height - 1)
        blend = strength * t * t * (3.0 - 2.0 * t)
        rgb = tuple(body + (head - body) * blend
                    for head, body in zip(head_linear, body_linear))
        pixels.extend((rgb + (1.0,)) * width)
    return pixels


def match_neck_color(source_samples, target_samples, *, color_space="linear",
                     strength=1.0, min_samples=8, min_gain=2.0 / 3.0,
                     max_gain=1.5, alpha_threshold=0.9):
    """Compute a bounded linear RGB multiplier from head-side to body skin.

    Source/target sampling reports explain a refused correction. The strength
    controls interpolation from the identity multiplier to the bounded gain.
    A multiplicative correction keeps texture detail and does not introduce a
    flat replacement color. ``limited`` identifies an intentionally incomplete
    match when the requested gain exceeds the conservative bounds.
    """
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Color matching strength must be between zero and one")
    if (not all(math.isfinite(value) for value in (min_gain, max_gain))
            or not 0 < min_gain <= 1 <= max_gain):
        raise ValueError("Gain bounds must be positive and contain one")
    result = {"status": "skipped", "reason": "disabled", "gain": (1.0, 1.0, 1.0),
              "confidence": 0.0, "limited": False, "source": None, "target": None}
    if strength == 0:
        result["status"] = "disabled"
        return result
    source = estimate_base_color(source_samples, color_space=color_space,
                                 min_samples=min_samples, alpha_threshold=alpha_threshold)
    target = estimate_base_color(target_samples, color_space=color_space,
                                 min_samples=min_samples, alpha_threshold=alpha_threshold)
    result.update(source=source, target=target,
                  confidence=min(source["confidence"], target["confidence"]))
    for side, sample in (("source", source), ("target", target)):
        if sample["status"] != "ok":
            result["reason"] = side + "_" + sample["reason"]
            return result
        luminance = sum(value * weight for value, weight in zip(
            sample["rgb_linear"], (0.2126, 0.7152, 0.0722)))
        if luminance < 0.015:
            result["reason"] = side + "_near_black_samples"
            result["confidence"] = 0.0
            return result
    source_rgb, target_rgb = source["rgb_linear"], target["rgb_linear"]
    if max(abs(a - b) for a, b in zip(source_rgb, target_rgb)) < 1e-7:
        result.update(status="unchanged", reason="colors_already_match")
        return result
    raw = tuple(b / max(a, 1e-4) for a, b in zip(source_rgb, target_rgb))
    bounded = tuple(max(min_gain, min(max_gain, value)) for value in raw)
    limited = any(abs(a - b) > 1e-10 for a, b in zip(raw, bounded))
    gain = tuple(1.0 + strength * (value - 1.0) for value in bounded)
    result.update(status="matched", reason="gain_limited" if limited else "color_matched",
                  gain=gain, limited=limited)
    return result

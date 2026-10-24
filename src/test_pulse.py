
from pathlib import Path
import argparse

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch


REGIONS = ("forehead", "lcheek", "rcheek")

# Landmark groups used to estimate facial geometry.
LEFT_EYE = 33
RIGHT_EYE = 263
LEFT_EYE_OUTER = 130
RIGHT_EYE_OUTER = 359
NOSE_TIP = 1
UPPER_LIP = 13
LOWER_LIP = 14


def landmark_xy(landmarks, index, width, height):
    point = landmarks[index]
    return np.array([point.x * width, point.y * height])


def polygon_mask(shape, polygon):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    points = np.round(polygon).astype(np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask > 0


def build_rois(landmarks, width, height):
    """Build approximate skin regions using facial landmarks."""
    points = np.array(
        [[p.x * width, p.y * height] for p in landmarks],
        dtype=np.float32,
    )

    face_top = points[10]
    eye_left = points[LEFT_EYE]
    eye_right = points[RIGHT_EYE]
    eye_mid_y = (eye_left[1] + eye_right[1]) / 2

    eye_outer_left = points[LEFT_EYE_OUTER]
    eye_outer_right = points[RIGHT_EYE_OUTER]
    nose = points[NOSE_TIP]
    upper_lip = points[UPPER_LIP]
    lower_lip = points[LOWER_LIP]

    face_width = np.linalg.norm(eye_right - eye_left) * 2.0
    face_height = max(
        np.linalg.norm(points[10] - points[152]), 1.0
    )

    # The forehead rectangle lies above the eyes and below the
    # upper face. Its boundaries are based on the eye landmarks.
    eye_span = abs(eye_right[0] - eye_left[0])
    forehead = np.array([
        [eye_left[0] - 0.12 * eye_span, face_top[1] + 0.10 * face_height],
        [eye_right[0] + 0.12 * eye_span, face_top[1] + 0.10 * face_height],
        [eye_right[0] + 0.12 * eye_span, eye_mid_y - 0.10 * face_height],
        [eye_left[0] - 0.12 * eye_span, eye_mid_y - 0.10 * face_height],
    ])

    # Approximate cheek regions between the eyes and mouth.
    # These are conservative regions, not a validated skin mask.
    y_top = eye_mid_y + 0.07 * face_height
    y_bottom = min(
        upper_lip[1] - 0.03 * face_height,
        nose[1] + 0.18 * face_height,
    )

    face_center_x = (eye_left[0] + eye_right[0]) / 2
    half_eye_span = eye_span / 2

    left_cheek = np.array([
        [face_center_x - 0.95 * half_eye_span, y_top],
        [face_center_x - 0.18 * half_eye_span, y_top],
        [face_center_x - 0.18 * half_eye_span, y_bottom],
        [face_center_x - 0.95 * half_eye_span, y_bottom],
    ])

    right_cheek = np.array([
        [face_center_x + 0.18 * half_eye_span, y_top],
        [face_center_x + 0.95 * half_eye_span, y_top],
        [face_center_x + 0.95 * half_eye_span, y_bottom],
        [face_center_x + 0.18 * half_eye_span, y_bottom],
    ])

    # Reject geometrically implausible ROIs.
    regions = {
        "forehead": forehead,
        "lcheek": left_cheek,
        "rcheek": right_cheek,
    }

    for name, polygon in regions.items():
        if (
            not np.isfinite(polygon).all()
            or polygon[:, 0].min() < 0
            or polygon[:, 0].max() >= width
            or polygon[:, 1].min() < 0
            or polygon[:, 1].max() >= height
        ):
            raise ValueError(f"Invalid {name} ROI.")

    if y_bottom <= y_top or face_width < 20:
        raise ValueError("Face is too small or ROI geometry is invalid.")

    return regions


def choose_face(face_landmarks_list, previous_center, width, height):
    """Choose the largest face initially, then the nearest face center."""
    candidates = []

    for landmarks in face_landmarks_list:
        points = np.array([
            [p.x * width, p.y * height] for p in landmarks.landmark
        ])

        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)

        center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        area = max(0, x1 - x0) * max(0, y1 - y0)

        candidates.append((landmarks, center, area))

    if not candidates:
        return None, None

    if previous_center is None:
        selected = max(candidates, key=lambda item: item[2])
    else:
        selected = min(
            candidates,
            key=lambda item: np.linalg.norm(
                item[1] - previous_center
            ),
        )

    return selected[0], selected[1]


def extract_rgb(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Could not determine video FPS.")

    rgb_values = {name: [] for name in REGIONS}
    preview = None
    previous_center = None
    frame_count = 0
    missing_faces = 0
    bad_rois = 0

    # Multiple faces are detected so we can select and track one target.
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=5,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            height, width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mesh.process(rgb_frame)

            if not result.multi_face_landmarks:
                missing_faces += 1
                for name in REGIONS:
                    rgb_values[name].append([np.nan] * 3)
                continue

            selected, center = choose_face(
                result.multi_face_landmarks,
                previous_center,
                width,
                height,
            )
            previous_center = center

            try:
                polygons = build_rois(selected.landmark, width, height)
            except ValueError:
                bad_rois += 1
                for name in REGIONS:
                    rgb_values[name].append([np.nan] * 3)
                continue

            frame_rgb_values = {}
            masks = {}

            for name, polygon in polygons.items():
                mask = polygon_mask(frame.shape, polygon)
                masks[name] = mask

                # Require enough pixels for a meaningful mean.
                if mask.sum() < 50:
                    frame_rgb_values[name] = [np.nan] * 3
                    continue

                pixels = rgb_frame[mask]

                # Reject very dark or nearly white pixels. This is
                # only a simple quality filter, not full skin detection.
                brightness = pixels.mean(axis=1)
                keep = (brightness > 25) & (brightness < 245)
                pixels = pixels[keep]

                if len(pixels) < 50:
                    frame_rgb_values[name] = [np.nan] * 3
                else:
                    frame_rgb_values[name] = pixels.mean(axis=0)

            for name in REGIONS:
                rgb_values[name].append(frame_rgb_values[name])

            # Draw a clean preview once, from one frame only.
            if preview is None and all(
                np.isfinite(frame_rgb_values[name]).all()
                for name in REGIONS
            ):
                preview = frame.copy()

                for name, polygon in polygons.items():
                    pts = np.round(polygon).astype(np.int32)
                    cv2.polylines(
                        preview, [pts], True, (0, 255, 0), 2
                    )
                    x, y = pts[0]
                    cv2.putText(
                        preview, name,
                        (int(x), max(15, int(y) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1,
                    )
    finally:
        cap.release()
        mesh.close()

    if frame_count == 0:
        raise ValueError("The video contains no readable frames.")

    signals = {}
    quality = {}

    for name, rows in rgb_values.items():
        arr = np.asarray(rows, dtype=float)
        bad = ~np.isfinite(arr).all(axis=1)
        bad_fraction = float(bad.mean())
        longest_gap = max_consecutive_true(bad)

        quality[name] = {
            "bad_fraction": bad_fraction,
            "longest_gap": longest_gap,
        }

        # Do not silently repair long gaps or excessive missing data.
        if bad_fraction > 0.05 or longest_gap > max(3, int(0.15 * fps)):
            raise ValueError(
                f"{name} ROI is unreliable: "
                f"{bad_fraction:.1%} invalid frames, "
                f"longest gap={longest_gap} frames. "
                "Use a clearer video or improve ROI placement."
            )

        # Interpolate only short, isolated gaps.
        indices = np.arange(len(arr))
        for channel in range(3):
            valid = np.isfinite(arr[:, channel])
            if valid.sum() < 10:
                raise ValueError(f"Too few valid samples for {name}.")
            arr[:, channel] = np.interp(
                indices, indices[valid], arr[valid, channel]
            )

        signals[name] = arr

    return (
        fps, signals, preview, frame_count,
        missing_faces, bad_rois, quality,
    )


def max_consecutive_true(values):
    longest = current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def pos(rgb, fps):
    """Windowed POS pulse extraction from a T x 3 RGB array."""
    rgb = np.asarray(rgb, dtype=float)

    window = max(2, int(round(1.6 * fps)))
    step = max(1, window // 2)

    if len(rgb) < window:
        raise ValueError(
            "Video is too short for POS. Record a longer clip."
        )

    pulse = np.zeros(len(rgb), dtype=float)
    weights = np.zeros(len(rgb), dtype=float)

    for start in range(0, len(rgb) - window + 1, step):
        block = rgb[start:start + window].T
        means = block.mean(axis=1, keepdims=True)

        if np.any(means <= 1e-6):
            continue

        normalized = block / means - 1.0
        s1 = normalized[1] - normalized[2]
        s2 = -2 * normalized[0] + normalized[1] + normalized[2]

        std2 = np.std(s2)
        if std2 < 1e-8:
            continue

        h = s1 + (np.std(s1) / std2) * s2
        h -= h.mean()

        pulse[start:start + window] += h
        weights[start:start + window] += 1

    valid = weights > 0
    if valid.mean() < 0.8:
        raise ValueError("POS produced too few valid samples.")

    pulse[valid] /= weights[valid]
    pulse[~valid] = np.interp(
        np.flatnonzero(~valid),
        np.flatnonzero(valid),
        pulse[valid],
    )

    return pulse


def filter_pulse(pulse, fps, low=0.7, high=3.0):
    if high >= fps / 2:
        raise ValueError(
            f"FPS must exceed {2 * high} for the selected filter."
        )

    if len(pulse) < 30:
        raise ValueError("Not enough samples to filter reliably.")

    sos = butter(
        3, [low, high], btype="bandpass",
        fs=fps, output="sos",
    )
    return sosfiltfilt(sos, pulse)


def estimate_bpm(pulse, fps):
    # Use the entire signal for the spectrum; frequency resolution
    # depends on duration and is not necessarily 0.1 BPM.
    frequencies, power = welch(
        pulse,
        fs=fps,
        nperseg=min(len(pulse), int(fps * 8)),
    )
    band = (frequencies >= 0.7) & (frequencies <= 3.0)

    if not band.any():
        raise ValueError("No frequencies in the chosen heart-rate band.")

    band_power = power[band]
    peak_index = np.argmax(band_power)
    peak_frequency = frequencies[band][peak_index]
    peak_power = band_power[peak_index]
    total_band_power = band_power.sum()

    concentration = (
        float(peak_power / total_band_power)
        if total_band_power > 0 else 0.0
    )

    return (
        float(peak_frequency * 60),
        frequencies,
        power,
        concentration,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", default="data/raw/real/my_video.mp4"
    )
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    (
        fps, signals, preview, total, missing, bad_rois, quality
    ) = extract_rgb(video)

    duration = total / fps
    print(f"Video FPS: {fps:.2f}")
    print(f"Frames processed: {total}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Frames without detected face: {missing}")
    print(f"Frames with invalid ROI geometry: {bad_rois}")

    pulse_signals = {}
    bpm_values = {}
    spectra = {}

    for name, rgb in signals.items():
        raw_pulse = pos(rgb, fps)
        filtered = filter_pulse(raw_pulse, fps)
        bpm, frequencies, power, concentration = estimate_bpm(
            filtered, fps
        )

        pulse_signals[name] = filtered
        bpm_values[name] = bpm
        spectra[name] = (frequencies, power)

        print(
            f"{name}: RGB shape={rgb.shape}, "
            f"dominant-frequency estimate={bpm:.1f} BPM, "
            f"spectral concentration={concentration:.3f}, "
            f"invalid-frame fraction={quality[name]['bad_fraction']:.1%}"
        )

    # This is a preliminary signal-quality warning, not a clinical
    # or forensic validation of the estimated heart rate.
    if duration < 20:
        print(
            "WARNING: Short recording. BPM estimates may be unstable; "
            "use a longer clip."
        )

    cache_dir = Path("cache/signals")
    cache_dir.mkdir(parents=True, exist_ok=True)

    output = cache_dir / f"{video.stem}__clean.npz"
    np.savez_compressed(
        output,
        rgb_forehead=signals["forehead"],
        rgb_lcheek=signals["lcheek"],
        rgb_rcheek=signals["rcheek"],
        fps=np.float64(fps),
        label=np.array("real"),
        pulse_forehead=pulse_signals["forehead"],
        pulse_lcheek=pulse_signals["lcheek"],
        pulse_rcheek=pulse_signals["rcheek"],
    )
    print(f"Saved RGB and pulse cache: {output}")

    if preview is not None:
        preview_path = cache_dir / f"{video.stem}__rois.jpg"
        cv2.imwrite(str(preview_path), preview)
        print(f"Saved ROI preview: {preview_path}")
    else:
        print("WARNING: No valid single-frame ROI preview was generated.")

    time_axis = np.arange(total) / fps
    fig, axes = plt.subplots(3, 1, figsize=(13, 10))

    for name, rgb in signals.items():
        axes[0].plot(time_axis, rgb[:, 1], label=name)

    axes[0].set_title("Mean green-channel intensity")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Pixel intensity")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for name, pulse in pulse_signals.items():
        axes[1].plot(np.arange(len(pulse)) / fps, pulse, label=name)

    axes[1].set_title("POS pulse estimates (bandpass filtered)")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Relative signal (a.u.)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    for name, (frequencies, power) in spectra.items():
        band = (frequencies >= 0.7) & (frequencies <= 3.0)
        axes[2].plot(
            frequencies[band] * 60,
            power[band],
            label=name,
        )

    axes[2].set_title("Pulse power spectrum")
    axes[2].set_xlabel("Frequency (BPM)")
    axes[2].set_ylabel("Power")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = cache_dir / f"{video.stem}__pulse.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")

    plt.show()


if __name__ == "__main__":
    main()

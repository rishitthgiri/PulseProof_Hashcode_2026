import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import os
import urllib.request

def ensure_haar_cascade():
    """Downloads the frontalface haar cascade locally if missing."""
    cascade_filename = "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_filename):
        print("Downloading OpenCV Haar Cascade file locally...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, cascade_filename)
    return cascade_filename

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Design a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply butter bandpass filter to signal."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def extract_rppg_signals(video_path):
    """
    Reads a video, detects faces using OpenCV Haar Cascades, crops forehead 
    and cheeks based on face proportions, computes mean RGB, and plots the POS pulse wave.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Fallback if metadata is missing

    # Ensure local cascade file is available
    cascade_path = ensure_haar_cascade()
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Error: Could not load Haar cascade classifier.")
        return

    rgb_forehead = []
    rgb_lcheek = []
    rgb_rcheek = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) > 0:
            # Take the first detected face (x, y, w, h)
            fx, fy, fw, fh = faces[0]

            # Define facial ROIs proportional to the face bounding box:
            fy_min, fy_max = fy + int(0.05 * fh), fy + int(0.25 * fh)
            fx_min, fx_max = fx + int(0.3 * fw), fx + int(0.7 * fw)

            lc_min_y, lc_max_y = fy + int(0.5 * fh), fy + int(0.7 * fh)
            lc_min_x, lc_max_x = fx + int(0.15 * fw), fx + int(0.4 * fw)

            rc_min_y, rc_max_y = fy + int(0.5 * fh), fy + int(0.7 * fh)
            rc_min_x, rc_max_x = fx + int(0.6 * fw), fx + int(0.85 * fw)

            def crop_mean(ymin, ymax, xmin, xmax):
                ymin, ymax = max(0, ymin), min(h, ymax)
                xmin, xmax = max(0, xmin), min(w, xmax)
                if ymin >= ymax or xmin >= xmax:
                    return [0.0, 0.0, 0.0]
                roi = rgb_frame[ymin:ymax, xmin:xmax]
                return np.mean(roi, axis=(0, 1))

            rgb_forehead.append(crop_mean(fy_min, fy_max, fx_min, fx_max))
            rgb_lcheek.append(crop_mean(lc_min_y, lc_max_y, lc_min_x, lc_max_x))
            rgb_rcheek.append(crop_mean(rc_min_y, rc_max_y, rc_min_x, rc_max_x))

    cap.release()

    rgb_forehead = np.array(rgb_forehead)
    rgb_lcheek = np.array(rgb_lcheek)
    rgb_rcheek = np.array(rgb_rcheek)

    if len(rgb_forehead) == 0:
        print("No face detected in video frames!")
        return

    # --- POS (Plane-Orthogonal-to-Skin) Algorithm Implementation ---
    def pos_algorithm(rgb_signal):
        mean_rgb = np.mean(rgb_signal, axis=0)
        if np.any(mean_rgb == 0):
            return np.zeros(len(rgb_signal))
        normalized = rgb_signal / mean_rgb - 1.0
        
        proj_matrix = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
        projection = np.dot(normalized, proj_matrix.T)
        
        s = projection[:, 0] + (np.std(projection[:, 0]) / (np.std(projection[:, 1]) + 1e-8)) * projection[:, 1]
        return s

    raw_pulse = pos_algorithm(rgb_forehead)
    filtered_pulse = bandpass_filter(raw_pulse, lowcut=0.7, highcut=3.0, fs=fps, order=4)

    # Plot the extracted pulse
    time_axis = np.arange(len(filtered_pulse)) / fps
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, filtered_pulse, color='crimson', linewidth=1.5)
    plt.title("Extracted rPPG Pulse Wave (OpenCV ROI + POS + Bandpass)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return rgb_forehead, rgb_lcheek, rgb_rcheek, fps

if __name__ == "__main__":
    # Use raw string `r"..."` to prevent escape sequence warnings on Windows paths
    test_video = r"C:\College Stuff\PuleProof\data\test_clip.mp4" 
    print(f"Processing test video: {test_video}")
    extract_rppg_signals(test_video)
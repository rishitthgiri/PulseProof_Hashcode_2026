
from pathlib import Path
import argparse
import subprocess

CRF_LEVELS = [23, 28, 35, 40]


def run_ffmpeg(command):
    print("Running:", " ".join(map(str, command)))
    subprocess.run(command, check=True)


def compress_video(source, output_dir):
    source = Path(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source.is_file():
        raise FileNotFoundError(f"Input video not found: {source}")

    # Preserve a separate copy of the source as the clean condition.
    clean_path = output_dir / f"{source.stem}__clean.mp4"
    if source.resolve() != clean_path.resolve():
        run_ffmpeg([
            "ffmpeg", "-y", "-i", str(source),
            "-map", "0:v:0", "-an",
            "-c:v", "libx264", "-crf", "0",
            "-preset", "ultrafast",
            str(clean_path),
        ])

    for crf in CRF_LEVELS:
        output = output_dir / f"{source.stem}__crf{crf}.mp4"
        run_ffmpeg([
            "ffmpeg", "-y", "-i", str(source),
            "-map", "0:v:0", "-an",
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            str(output),
        ])

    # Approximation of a messaging-app transcode, not an actual
    # WhatsApp encoding profile.
    messaging = output_dir / f"{source.stem}__messaging_approx.mp4"
    run_ffmpeg([
        "ffmpeg", "-y", "-i", str(source),
        "-map", "0:v:0", "-an",
        "-vf", "scale=-2:480",
        "-c:v", "libx264", "-crf", "28",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        str(messaging),
    ])

    print(f"Compression variants saved in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate FFmpeg compression conditions for a video."
    )
    parser.add_argument("--input", required=True, help="Source video")
    parser.add_argument(
        "--output-dir", default="data/processed",
        help="Directory for generated variants",
    )
    args = parser.parse_args()
    compress_video(args.input, args.output_dir)


if __name__ == "__main__":
    main()

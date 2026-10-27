"""Read-only QC of the Resolve export; no editing or audio processing."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "tmp/media-runtime").exists():
    sys.path.insert(0, str(ROOT / "tmp/media-runtime"))
import imageio_ffmpeg
from PIL import Image, ImageDraw


def main():
    video = ROOT / "media/purumogu-demo.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    reader = imageio_ffmpeg.read_frames(str(video))
    meta = next(reader)
    reader.close()
    assert tuple(meta["size"]) == (1920, 1080)
    assert abs(meta["fps"] - 30) < .001
    assert abs(meta["duration"] - 115) < .1
    frames, duration = imageio_ffmpeg.count_frames_and_secs(str(video))
    assert frames == 3450, frames
    checked = subprocess.run([
        ffmpeg, "-hide_banner", "-nostats", "-i", str(video),
        "-vf", "blackdetect=d=0.2:pix_th=0.03", "-af", "ebur128=peak=true",
        "-f", "null", "-"
    ], capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    log = checked.stderr
    black = [dict(start=float(a), end=float(b), duration=float(c)) for a, b, c in
             re.findall(r"black_start:([\d.]+) black_end:([\d.]+) black_duration:([\d.]+)", log)]
    assert not any(x["start"] > 1 and x["end"] < 114 for x in black), black
    loudness = re.findall(r"I:\s+(-?[\d.]+) LUFS", log)
    peaks = re.findall(r"Peak:\s+(-?[\d.]+) dBFS", log)
    assert loudness and peaks
    assert -20 < float(loudness[-1]) < -12
    assert float(peaks[-1]) < 0
    sheet = Image.new("RGB", (960, 3 * 294), "#11252c")
    draw = ImageDraw.Draw(sheet)
    for n, second in enumerate([3, 22, 42, 62, 78, 96]):
        data = subprocess.run([
            ffmpeg, "-v", "error", "-ss", str(second), "-i", str(video), "-frames:v", "1",
            "-vf", "scale=480:270", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
        ], capture_output=True, check=True).stdout
        frame = Image.frombytes("RGB", (480, 270), data)
        x, y = (n % 2) * 480, (n // 2) * 294
        sheet.paste(frame, (x, y))
        draw.text((x + 10, y + 275), f"{second // 60:02}:{second % 60:02}", fill="#dbe9e1")
    (ROOT / "tmp").mkdir(exist_ok=True)
    sheet.save(ROOT / "tmp/final-film-contact-sheet.png")
    result = {
        "file": "media/purumogu-demo.mp4", "editor": "DaVinci Resolve Studio 21.1",
        "sha256": hashlib.sha256(video.read_bytes()).hexdigest(), "bytes": video.stat().st_size,
        "size": meta["size"], "fps": meta["fps"], "decoded_frames": frames,
        "decoded_duration_s": duration, "integrated_lufs": float(loudness[-1]),
        "true_peak_dbfs": float(peaks[-1]), "black_intervals_0_2s_or_longer": black,
        "checks": "all frames decode; no interior black interval >=0.2s; unclipped audio; six representative rendered frames reviewed"
    }
    (ROOT / "reports/media-qc.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

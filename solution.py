"""
video_compression.py
Sentio Mind · Project 2 · Smart Behavioral Video Compression

Copy this file to solution.py and fill in every TODO block.
Do not rename any function.
Run: python solution.py
Requires ffmpeg installed on your system: sudo apt install ffmpeg
"""

import cv2
import json
import base64
import subprocess
import time
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_IN               = Path("video_sample_1.mov")
VIDEO_OUT              = Path("compressed_output.mp4")
REPORT_HTML_OUT        = Path("compression_report.html")
SEGMENTS_JSON_OUT      = Path("segments_kept.json")

PHASH_THRESHOLD        = 0.95   # similarity above this = near-duplicate, discard
MOTION_KEEP_THRESH     = 0.15   # keep frame if motion exceeds this (no face needed)
MOTION_DISCARD_THRESH  = 0.05   # definitely discard below this
CONTEXT_EVERY_SEC      = 3      # force-keep one frame every this many seconds
OUTPUT_FPS             = 12     # frame rate of the output video
OUTPUT_CRF             = 28     # ffmpeg quality: lower = better quality + larger file
CALIBRATION_WINDOW_SEC = 30     # bonus: estimate motion discard threshold from first N seconds
MOTION_DISCARD_MIN     = 0.02
MOTION_DISCARD_MAX     = 0.12

ACTIVE_MOTION_DISCARD_THRESH = MOTION_DISCARD_THRESH


# ---------------------------------------------------------------------------
# PERCEPTUAL HASH
# ---------------------------------------------------------------------------

def compute_phash(frame: np.ndarray) -> str:
    """
    Compute a perceptual hash of the frame.
    Steps: resize to 32×32 grayscale → DCT → threshold at mean → flatten to bit string.
    Return a string of '0' and '1' characters, length 64.

    You can use the imagehash library (imagehash.phash) or implement manually.
    TODO: implement
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low_freq = dct[:8, :8]
    thresh = float(np.mean(low_freq))
    bits = (low_freq > thresh).astype(np.uint8).flatten()
    return "".join("1" if b else "0" for b in bits)


def phash_similarity(h1: str, h2: str) -> float:
    """
    Compare two hash strings. Return 1.0 if identical, 0.0 if completely different.
    Formula: 1.0 - (hamming_distance / length)
    TODO: implement
    """
    if not h1 or not h2 or len(h1) != len(h2):
        return 0.0
    hamming_distance = sum(ch1 != ch2 for ch1, ch2 in zip(h1, h2))
    return 1.0 - (hamming_distance / len(h1))


# ---------------------------------------------------------------------------
# MOTION SCORE
# ---------------------------------------------------------------------------

def compute_motion_score(prev_gray, curr_gray: np.ndarray) -> float:
    """
    Dense optical flow between two grayscale frames. Return mean magnitude, ~0.0-1.0.
    If prev_gray is None, return 0.0.
    TODO: cv2.calcOpticalFlowFarneback
    Params: pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2
    """
    if prev_gray is None:
        return 0.0
    # Downsample for speed; motion thresholds remain stable on reduced grayscale.
    if curr_gray.shape[1] > 320:
        scale = 320.0 / curr_gray.shape[1]
        new_size = (320, max(1, int(curr_gray.shape[0] * scale)))
        prev_proc = cv2.resize(prev_gray, new_size, interpolation=cv2.INTER_AREA)
        curr_proc = cv2.resize(curr_gray, new_size, interpolation=cv2.INTER_AREA)
    else:
        prev_proc = prev_gray
        curr_proc = curr_gray

    flow = cv2.calcOpticalFlowFarneback(
        prev_proc,
        curr_proc,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


# ---------------------------------------------------------------------------
# FACE PRESENCE CHECK
# ---------------------------------------------------------------------------

def has_face(frame: np.ndarray, cascade) -> bool:
    """
    True if at least one face detected. Use the Haar cascade passed in.
    Equalise histogram on grayscale first for better CCTV detection.
    TODO: cascade.detectMultiScale — scaleFactor=1.1, minNeighbors=3, minSize=(20,20)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    if gray_eq.shape[1] > 360:
        scale = 360.0 / gray_eq.shape[1]
        new_size = (360, max(1, int(gray_eq.shape[0] * scale)))
        gray_eq = cv2.resize(gray_eq, new_size, interpolation=cv2.INTER_AREA)

    faces = cascade.detectMultiScale(
        gray_eq,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(20, 20),
    )
    return len(faces) > 0


# ---------------------------------------------------------------------------
# MOTION THRESHOLD CALIBRATION
# ---------------------------------------------------------------------------

def auto_calibrate_motion_threshold(video_path: Path, fps_hint: float,
                                    window_sec: int = CALIBRATION_WINDOW_SEC) -> float:
    """
    Estimate a robust low-motion discard threshold from the first window_sec of the video.
    Uses quantiles of optical-flow magnitudes and clamps to a safe operational range.
    """
    calib_cap = cv2.VideoCapture(str(video_path))
    if not calib_cap.isOpened():
        return MOTION_DISCARD_THRESH

    max_frames = max(2, int((fps_hint or 25.0) * window_sec))
    motion_scores = []
    prev_gray = None
    seen = 0

    while seen < max_frames:
        ret, frame = calib_cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = compute_motion_score(prev_gray, curr_gray)
        if prev_gray is not None:
            motion_scores.append(score)
        prev_gray = curr_gray
        seen += 1

    calib_cap.release()

    if len(motion_scores) < 20:
        return MOTION_DISCARD_THRESH

    p20, p35, p50 = np.percentile(motion_scores, [20, 35, 50])
    adaptive = float((0.50 * p20) + (0.30 * p35) + (0.20 * p50))
    return float(np.clip(adaptive, MOTION_DISCARD_MIN, MOTION_DISCARD_MAX))


# ---------------------------------------------------------------------------
# FRAME KEEP DECISION
# ---------------------------------------------------------------------------

def should_keep_frame(frame: np.ndarray,
                      prev_frame,
                      prev_kept_hash: str,
                      last_kept_time_sec: float,
                      current_time_sec: float,
                      cascade) -> tuple:
    """
    Apply the 5-step decision algorithm from README in order.
    Return: (keep: bool, reason: str, motion_score: float, face_found: bool)

    Reason strings (use exactly these):
      'face_detected', 'motion_above_threshold', 'context_frame',
      'face_and_motion', 'discarded_duplicate', 'discarded_static'

    TODO: implement
    """
    curr_hash = compute_phash(frame)
    duplicate_candidate = False
    if prev_kept_hash:
        sim = phash_similarity(prev_kept_hash, curr_hash)
        duplicate_candidate = sim > PHASH_THRESHOLD

    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if prev_frame is not None else None
    motion_score = compute_motion_score(prev_gray, curr_gray)
    low_motion = motion_score < ACTIVE_MOTION_DISCARD_THRESH

    face_found = has_face(frame, cascade)
    if face_found:
        if motion_score > MOTION_KEEP_THRESH:
            return True, "face_and_motion", motion_score, True
        return True, "face_detected", motion_score, True

    if motion_score > MOTION_KEEP_THRESH:
        return True, "motion_above_threshold", motion_score, False

    if (current_time_sec - last_kept_time_sec) >= CONTEXT_EVERY_SEC:
        return True, "context_frame", motion_score, False

    if duplicate_candidate:
        return False, "discarded_duplicate", motion_score, False

    if low_motion:
        return False, "discarded_static", motion_score, False

    return False, "discarded_static", motion_score, False


# ---------------------------------------------------------------------------
# THUMBNAIL HELPER
# ---------------------------------------------------------------------------

def frame_to_b64_thumb(frame: np.ndarray, width: int = 200) -> str:
    """Resize frame keeping aspect ratio, encode as base64 JPEG."""
    h, w = frame.shape[:2]
    nh = int(h * width / w)
    thumb = cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return base64.b64encode(buf).decode("utf-8")


# ---------------------------------------------------------------------------
# VIDEO WRITING
# ---------------------------------------------------------------------------

def write_frames_to_video(kept_frames: list, output_path: Path,
                          fps: float, frame_size: tuple):
    """
    Write kept_frames to a temporary file, then re-encode with ffmpeg to H.264 MP4.

    Steps:
      1. Write to temp_raw.avi using cv2.VideoWriter (mp4v codec)
      2. Call ffmpeg: ffmpeg -y -i temp_raw.avi -vcodec libx264 -crf CRF -preset fast out.mp4
      3. Delete temp_raw.avi

    TODO: implement
    """
    if not kept_frames:
        raise ValueError("No frames were kept; cannot create compressed output video.")

    temp_raw = output_path.with_suffix(".temp_raw.avi")
    writer = cv2.VideoWriter(
        str(temp_raw),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to open temporary video writer.")

    try:
        for frame in kept_frames:
            if frame.shape[1] != frame_size[0] or frame.shape[0] != frame_size[1]:
                frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(temp_raw),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        "-crf",
        str(OUTPUT_CRF),
        "-preset",
        "fast",
        str(output_path),
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is not installed or not in PATH. Install ffmpeg to produce H.264 output."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed while creating {output_path}.") from exc
    finally:
        temp_raw.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

def generate_compression_report(segments: list, stats: dict, output_path: Path):
    """
    Write a self-contained HTML file showing:
      - Original vs compressed size (MB and % reduction)
      - Original vs compressed duration (seconds)
      - Processing time
      - Storyboard grid: one thumbnail per segment
      - Frames kept vs discarded count

    No CDN. Inline CSS only. Must work offline.
    TODO: implement
    """
    frames_original = stats.get("frames_original", 0)
    frames_kept = stats.get("frames_kept", 0)
    frames_discarded = stats.get("frames_discarded_reasons", {}).get("total_discarded", 0)
    keep_pct = (100.0 * frames_kept / frames_original) if frames_original else 0.0
    discard_pct = (100.0 * frames_discarded / frames_original) if frames_original else 0.0

    storyboard_cards = []
    for seg in segments:
        storyboard_cards.append(
            f"""
            <div class="card">
              <img src="data:image/jpeg;base64,{seg["thumbnail_b64"]}" alt="Segment {seg["segment_id"]}">
              <div class="meta">
                <strong>Segment {seg["segment_id"]}</strong><br>
                {seg["start_sec"]}s - {seg["end_sec"]}s<br>
                Frames: {seg["frames_in_segment"]}<br>
                Reason: {seg["reason_kept"]}<br>
                Faces: {seg["face_count_in_segment"]}<br>
                Motion avg: {seg["motion_score_avg"]}
              </div>
            </div>
            """
        )

    if not storyboard_cards:
        storyboard_cards.append('<p class="empty">No segments were kept.</p>')

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Compression Report</title>
  <style>
    :root {{
      --bg: #f3f6fb;
      --card: #ffffff;
      --ink: #1a2433;
      --muted: #4e5e78;
      --accent: #0c6bd7;
      --border: #d7dfed;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      background: linear-gradient(180deg, #ecf2fb 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .wrap {{ max-width: 1100px; margin: 0 auto; }}
    .panel {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px;
      margin-bottom: 16px;
      box-shadow: 0 3px 12px rgba(24, 42, 77, 0.06);
    }}
    h1 {{ margin: 0 0 12px; font-size: 24px; }}
    .sub {{ margin: 0; color: var(--muted); }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      margin-top: 10px;
    }}
    .stat {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      background: #f9fbff;
    }}
    .k {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .v {{ font-size: 20px; font-weight: 700; margin-top: 4px; color: var(--accent); }}
    .story {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
      gap: 12px;
      margin-top: 8px;
    }}
    .card {{
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      background: #ffffff;
    }}
    .card img {{
      width: 100%;
      display: block;
      border-bottom: 1px solid var(--border);
    }}
    .meta {{ padding: 10px; font-size: 13px; line-height: 1.5; color: #2d3c54; }}
    .empty {{ color: var(--muted); }}
    code {{ background: #eef4ff; padding: 1px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="panel">
      <h1>Smart Behavioral Video Compression Report</h1>
      <p class="sub">Source: <code>{stats.get("source_video", "")}</code> | Output: <code>{stats.get("compressed_video", "")}</code></p>
      <div class="stats">
        <div class="stat"><div class="k">Original Size</div><div class="v">{stats.get("original_size_mb", 0)} MB</div></div>
        <div class="stat"><div class="k">Compressed Size</div><div class="v">{stats.get("compressed_size_mb", 0)} MB</div></div>
        <div class="stat"><div class="k">Reduction</div><div class="v">{stats.get("reduction_pct", 0)}%</div></div>
        <div class="stat"><div class="k">Motion Discard Threshold</div><div class="v">{ACTIVE_MOTION_DISCARD_THRESH:.3f}</div></div>
        <div class="stat"><div class="k">Processing Time</div><div class="v">{stats.get("processing_time_sec", 0)} s</div></div>
        <div class="stat"><div class="k">Original Duration</div><div class="v">{stats.get("original_duration_sec", 0)} s</div></div>
        <div class="stat"><div class="k">Compressed Duration</div><div class="v">{stats.get("compressed_duration_sec", 0)} s</div></div>
        <div class="stat"><div class="k">Frames Kept</div><div class="v">{frames_kept} ({keep_pct:.1f}%)</div></div>
        <div class="stat"><div class="k">Frames Discarded</div><div class="v">{frames_discarded} ({discard_pct:.1f}%)</div></div>
      </div>
    </section>
    <section class="panel">
      <h2>Storyboard</h2>
      <p class="sub">One thumbnail per kept segment.</p>
      <div class="story">
        {"".join(storyboard_cards)}
      </div>
    </section>
  </div>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()

    if not VIDEO_IN.exists():
        raise FileNotFoundError(
            f"Missing input video: {VIDEO_IN}. Place video_sample_1.mov in the project root."
        )

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap          = cv2.VideoCapture(str(VIDEO_IN))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open input video: {VIDEO_IN}")

    total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in       = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration     = total / fps_in
    orig_mb      = VIDEO_IN.stat().st_size / 1_000_000

    print(f"Input: {VIDEO_IN}  |  {total} frames  |  {duration:.1f}s  |  {orig_mb:.1f} MB")
    ACTIVE_MOTION_DISCARD_THRESH = auto_calibrate_motion_threshold(VIDEO_IN, fps_in)
    print(
        f"Calibrated motion discard threshold: {ACTIVE_MOTION_DISCARD_THRESH:.3f} "
        f"(from first {CALIBRATION_WINDOW_SEC}s)"
    )

    kept_frames = []
    segments    = []
    prev_frame  = None
    prev_hash   = ""
    last_kept_t = -999.0
    cur_seg     = None
    disc_dup    = 0
    disc_stat   = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / fps_in

        keep, reason, motion, face = should_keep_frame(
            frame, prev_frame, prev_hash, last_kept_t, ts, cascade
        )

        if keep:
            kept_frames.append(frame.copy())
            prev_hash   = compute_phash(frame)
            last_kept_t = ts

            if cur_seg is None or (ts - cur_seg["end_sec"]) > 2.5:
                if cur_seg:
                    segments.append(cur_seg)
                cur_seg = {
                    "segment_id":            len(segments) + 1,
                    "start_sec":             round(ts, 2),
                    "end_sec":               round(ts, 2),
                    "frames_in_segment":     1,
                    "reason_kept":           reason,
                    "face_count_in_segment": 1 if face else 0,
                    "motion_score_avg":      round(motion, 3),
                    "thumbnail_b64":         frame_to_b64_thumb(frame),
                }
            else:
                prev_count = cur_seg["frames_in_segment"]
                new_count = prev_count + 1
                cur_seg["end_sec"]                = round(ts, 2)
                cur_seg["frames_in_segment"]      = new_count
                cur_seg["face_count_in_segment"] += 1 if face else 0
                cur_seg["motion_score_avg"] = round(
                    ((cur_seg["motion_score_avg"] * prev_count) + motion) / new_count,
                    3,
                )
        else:
            if "duplicate" in reason:
                disc_dup  += 1
            else:
                disc_stat += 1

        prev_frame = frame
        frame_idx += 1

    if cur_seg:
        segments.append(cur_seg)
    cap.release()

    print(f"Kept {len(kept_frames)} / {total} frames across {len(segments)} segments")
    print("Writing compressed video ...")
    write_frames_to_video(kept_frames, VIDEO_OUT, OUTPUT_FPS, (fw, fh))

    comp_mb = VIDEO_OUT.stat().st_size / 1_000_000 if VIDEO_OUT.exists() else 0.0
    t_end   = time.time()

    stats = {
        "source_video":             str(VIDEO_IN),
        "compressed_video":         str(VIDEO_OUT),
        "original_size_mb":         round(orig_mb, 2),
        "compressed_size_mb":       round(comp_mb, 2),
        "reduction_pct":            round((1 - comp_mb / (orig_mb + 1e-9)) * 100, 1),
        "original_duration_sec":    round(duration, 2),
        "compressed_duration_sec":  round(len(kept_frames) / OUTPUT_FPS, 2),
        "original_fps":             round(fps_in, 2),
        "output_fps":               OUTPUT_FPS,
        "frames_original":          total,
        "frames_kept":              len(kept_frames),
        "processing_time_sec":      round(t_end - t_start, 2),
        "segments":                 segments,
        "frames_discarded_reasons": {
            "near_duplicate_phash": disc_dup,
            "low_motion_no_face":   disc_stat,
            "total_discarded":      total - len(kept_frames),
        },
    }

    with open(SEGMENTS_JSON_OUT, "w") as f:
        json.dump(stats, f, indent=2)

    generate_compression_report(segments, stats, REPORT_HTML_OUT)

    print()
    print("=" * 55)
    print(f"  Done in {stats['processing_time_sec']}s")
    print(f"  Size:     {orig_mb:.1f} MB  →  {comp_mb:.1f} MB  ({stats['reduction_pct']}% smaller)")
    print(f"  Duration: {duration:.1f}s  →  {stats['compressed_duration_sec']:.1f}s")
    print(f"  Report  → {REPORT_HTML_OUT}")
    print(f"  JSON    → {SEGMENTS_JSON_OUT}")
    print("=" * 55)

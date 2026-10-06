# transcribe.py
# Author: Jibril Sulaiman
# What:   Turns a multi-hour meeting recording into a timestamped Markdown transcript
#         with a speaker label at every speaker change, using the Google Gemini API
#         (Files API + generate_content).
# Why:    One request for a 7+ hour file runs out of output, loops, or fails all at once.
#         This splits the audio into 45-minute chunks with ffmpeg, transcribes each chunk
#         with thinking off and a strict prompt, shifts every timestamp to absolute time,
#         retries 503s with backoff, re-requests chunks that stopped short (coverage
#         check), stops cleanly on the free-tier daily quota, and resumes on the next run.
#         Short reference voice clips of each attendee help Gemini match speakers.
# How:    1. Put your recording next to this file and set audio_path below.
#         2. Fill in attendees and reference_timestamps (see README, Step 6).
#         3. Set GEMINI_API_KEY in your environment, then:  python transcribe.py
#         Rerun the same command after a quota stop; finished chunks are reused.
# Needs:  Python 3.9+, pip install google-genai, ffmpeg + ffprobe on PATH.

import os
import re
import sys
import time
import json
import subprocess
from google import genai
from google.genai import types
from google.genai import errors

def log(msg):
    print(msg, flush=True)

# ---- CONFIG ----
audio_path = "meeting.mp3"  # REPLACE_WITH your recording file name
chunk_seconds = 2700  # 45 minutes per chunk
chunk_dir = "chunks"
output_path = "transcript.md"
progress_path = "progress.json"  # not used by this version: resume reads output_path instead (README, Step 8)
max_retries = 5
max_continuations_per_chunk = 6
coverage_tolerance_seconds = 45

# Everyone who speaks in the recording. Names are hints for Gemini and become the
# speaker labels in the transcript. REPLACE these placeholders.
attendees = [
    "Speaker Name 1", "Speaker Name 2", "Speaker Name 3", "Speaker Name 4",
]

# Approximate start time (in the ORIGINAL full audio, seconds) of a clean,
# clearly-identified moment of each person's voice - mostly self-introductions
# from early in the meeting. Used to build reference voice samples so Gemini
# can match voices in later chunks against a known sample, not just guess
# from context each time.
reference_timestamps = {
    # "Name exactly as in attendees": start second of a clean 12-second sample
    "Speaker Name 1": 60,     # 00:01:00  REPLACE
    "Speaker Name 2": 95,     # 00:01:35  REPLACE
    "Speaker Name 3": 130,    # 00:02:10  REPLACE
    "Speaker Name 4": 170,    # 00:02:50  REPLACE
}
reference_clip_duration = 12  # seconds
reference_clip_dir = "reference_clips"

log("Starting chunked transcription script (coverage-aware, quota-conscious)...")

if not os.path.exists(audio_path):
    log(f"ERROR: File not found at {audio_path}. Check the filename and folder.")
    sys.exit(1)

# ---- SPLIT AUDIO INTO CHUNKS (skip if already done) ----
os.makedirs(chunk_dir, exist_ok=True)
chunk_pattern = os.path.join(chunk_dir, "chunk_%03d.mp3")
existing_chunks = sorted(f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".mp3"))

if existing_chunks:
    log(f"Found {len(existing_chunks)} existing chunk(s) in '{chunk_dir}'. Reusing them (skipping split).")
    chunk_files = existing_chunks
else:
    log(f"Splitting audio into {chunk_seconds // 60}-minute chunks with ffmpeg...")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-f", "segment",
         "-segment_time", str(chunk_seconds), "-c", "copy", chunk_pattern],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        log("ERROR: ffmpeg failed to split the audio.")
        log(result.stderr[-2000:])
        sys.exit(1)
    chunk_files = sorted(f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".mp3"))
    if not chunk_files:
        log("ERROR: No chunks were created.")
        sys.exit(1)
    log(f"Split complete. {len(chunk_files)} chunk(s) created.")

def get_duration_seconds(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return None

def extract_reference_clips():
    """Cut a short labeled voice sample per attendee from the original audio,
    using known self-introduction timestamps. Returns dict name -> clip path."""
    os.makedirs(reference_clip_dir, exist_ok=True)
    clip_paths = {}
    for name, start_sec in reference_timestamps.items():
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
        clip_path = os.path.join(reference_clip_dir, f"{safe_name}.mp3")
        if not os.path.exists(clip_path):
            log(f"  Extracting reference clip for {name} (from {format_hms(start_sec)})...")
            result = subprocess.run(
                ["ffmpeg", "-y", "-ss", str(start_sec), "-t", str(reference_clip_duration),
                 "-i", audio_path, "-c", "copy", clip_path],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                log(f"  WARNING: Failed to extract reference clip for {name}. Skipping this reference.")
                continue
        clip_paths[name] = clip_path
    return clip_paths

def format_hms(total_seconds):
    total_seconds = int(total_seconds)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

turn_pattern = re.compile(r"\*\*\[(\d{2}):(\d{2}):(\d{2})\]\s*([^\*\n]+?)\*\*")

def parse_segments(raw_text):
    matches = list(turn_pattern.finditer(raw_text))
    segments = []
    for i, m in enumerate(matches):
        hh, mm, ss, speaker = m.groups()
        seconds = int(hh) * 3600 + int(mm) * 60 + int(ss)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        content = raw_text[start:end].strip()
        segments.append((seconds, speaker.strip(), content))
    return segments

def drop_repetition(segments, min_allowed=-1):
    kept = []
    max_seen = min_allowed
    dropped = 0
    for seconds, speaker, content in segments:
        if seconds < max_seen:
            dropped += 1
            continue
        kept.append((seconds, speaker, content))
        max_seen = seconds
    return kept, dropped

# ---- LOAD EXISTING OUTPUT (if any) AND GROUP INTO PER-CHUNK LOCAL SEGMENTS ----
existing_chunk_segments = {}  # idx -> list of (local_seconds, speaker, content)
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        old_text = f.read()
    old_matches = list(turn_pattern.finditer(old_text))
    old_segments = []
    for i, m in enumerate(old_matches):
        hh, mm, ss, speaker = m.groups()
        abs_seconds = int(hh) * 3600 + int(mm) * 60 + int(ss)
        start = m.end()
        end = old_matches[i + 1].start() if i + 1 < len(old_matches) else len(old_text)
        content = old_text[start:end].strip()
        old_segments.append((abs_seconds, speaker.strip(), content))

    for abs_seconds, speaker, content in old_segments:
        idx = abs_seconds // chunk_seconds
        local_seconds = abs_seconds - idx * chunk_seconds
        existing_chunk_segments.setdefault(idx, []).append((local_seconds, speaker, content))

    log(f"Loaded {len(old_segments)} existing segments from previous output, grouped into {len(existing_chunk_segments)} chunk(s).")
else:
    log("No existing output file found. Starting fresh.")

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

generation_config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_budget=0),
    max_output_tokens=65536
)

log("Extracting and uploading speaker reference voice samples...")
reference_clip_paths = extract_reference_clips()
reference_files = {}
for name, clip_path in reference_clip_paths.items():
    try:
        ref_file = client.files.upload(file=clip_path)
        # Reference clips are tiny, but still wait for ACTIVE state just in case.
        wait_count = 0
        while ref_file.state.name == "PROCESSING" and wait_count < 10:
            time.sleep(2)
            ref_file = client.files.get(name=ref_file.name)
            wait_count += 1
        if ref_file.state.name == "ACTIVE":
            reference_files[name] = ref_file
        else:
            log(f"  WARNING: Reference clip for {name} did not become ACTIVE. Skipping.")
    except Exception as e:
        log(f"  WARNING: Failed to upload reference clip for {name}: {e}")

log(f"Uploaded {len(reference_files)} of {len(reference_clip_paths)} reference voice samples.")

def build_reference_parts():
    """Build the list of (label, file) parts to prepend to every chunk request
    so Gemini can compare voices against known samples."""
    parts = ["Here are short reference voice samples of known meeting attendees, each labeled with their name. Use these to help identify who is speaking in the meeting audio that follows, by comparing voice characteristics (pitch, tone, pacing, accent) - not just content clues."]
    for name, ref_file in reference_files.items():
        parts.append(f"Reference voice sample - this speaker is {name}:")
        parts.append(ref_file)
    return parts

class DailyQuotaExceeded(Exception):
    pass

def generate_with_retry(contents):
    attempt = 0
    while True:
        attempt += 1
        try:
            return client.models.generate_content(
                model="gemini-flash-latest", contents=contents, config=generation_config
            )
        except errors.ServerError as e:
            if attempt >= max_retries:
                log(f"ERROR: Server error persisted after {max_retries} attempts: {e}")
                raise
            wait_time = min(30 * attempt, 180)
            log(f"  Server error (attempt {attempt}/{max_retries}): {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
        except errors.ClientError as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg and ("PerDay" in msg or "per day" in msg.lower()):
                raise DailyQuotaExceeded(msg)
            if attempt >= max_retries:
                log(f"ERROR: Client error persisted after {max_retries} attempts: {e}")
                raise
            wait_time = min(15 * attempt, 60)
            log(f"  Client error (attempt {attempt}/{max_retries}): {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

def transcribe_chunk_fully(chunk_path, chunk_duration, chunk_label):
    """Upload the chunk and keep requesting continuations until the whole
    duration is covered (or we hit the continuation cap)."""
    log("Uploading chunk to Gemini...")
    upload_start = time.time()
    myfile = client.files.upload(file=chunk_path)
    log(f"Upload complete in {time.time() - upload_start:.0f}s. State: {myfile.state.name}")

    poll_count = 0
    while myfile.state.name == "PROCESSING":
        poll_count += 1
        log(f"  Still processing chunk... (check #{poll_count})")
        time.sleep(5)
        myfile = client.files.get(name=myfile.name)

    if myfile.state.name != "ACTIVE":
        log(f"ERROR: Chunk ended in unexpected state: {myfile.state.name}.")
        return []

    base_prompt = f"""Transcribe this audio chunk in full, from start to finish. This is one segment of a longer meeting recording. Reference voice samples for known attendees were provided before this audio - use them to match voices whenever possible.

Identify the speaker every time the speaker changes, and include a timestamp at each speaker change, relative to the START of THIS chunk (i.e. the chunk begins at 00:00:00).

Likely speakers (meeting attendees): {', '.join(attendees)}.

SPEAKER IDENTIFICATION PRIORITY (use in this order):
1. Voice match against the reference samples provided (pitch, tone, pacing, accent). This is the most reliable method - prioritize it over guessing from context.
2. Contextual clues: self-introductions, being addressed by name, references to their known role.
3. If neither gives a confident match, label the speaker as Speaker 1, Speaker 2, etc., and stay consistent with that label for that same voice throughout this chunk.

CRITICAL FORMATTING RULES (follow exactly, no exceptions):
- Every single speaker turn, no matter how short, must start on its own new line, with a blank line between turns.
- Never place more than one speaker tag on the same line.
- Timestamps must always move forward. Never repeat or re-transcribe a portion of the audio you have already covered.
- You MUST transcribe all the way to the end of this audio chunk. Do not stop early.

CRITICAL CONTENT RULES:
- Output ONLY the final transcript. No reasoning, no commentary, no preamble, no headers.

Format each turn exactly like this, with a blank line in between:

**[HH:MM:SS] Speaker Name**
Transcribed text for this turn."""

    contents = build_reference_parts() + [base_prompt, myfile]
    all_segments = []
    continuation_count = 0
    target_seconds = int(chunk_duration) if chunk_duration else chunk_seconds

    while True:
        log(f"  Sending transcription request (attempt {continuation_count + 1})...")
        gen_start = time.time()
        response = generate_with_retry(contents)
        log(f"  Response received in {time.time() - gen_start:.0f}s.")

        new_segments = parse_segments(response.text)
        min_allowed = all_segments[-1][0] if all_segments else -1
        new_segments, dropped = drop_repetition(new_segments, min_allowed)
        if dropped:
            log(f"  WARNING: dropped {dropped} repeated/looped segment(s) from this response.")

        all_segments.extend(new_segments)
        last_covered = all_segments[-1][0] if all_segments else 0
        log(f"  Coverage so far: up to {format_hms(last_covered)} of {format_hms(target_seconds)}.")

        if last_covered >= target_seconds - coverage_tolerance_seconds:
            log("  Full chunk coverage reached.")
            break

        continuation_count += 1
        if continuation_count > max_continuations_per_chunk:
            log(f"  WARNING: Hit max continuation attempts. Chunk may still be missing content after {format_hms(last_covered)}.")
            break

        log(f"  Chunk stopped early at {format_hms(last_covered)}. Requesting continuation...")
        continue_prompt = f"""Continue transcribing the SAME audio chunk from earlier. You previously transcribed up to timestamp {format_hms(last_covered)} (relative to the start of this chunk). Reference voice samples for known attendees were provided before this audio - keep using them to match voices.

Continue from exactly {format_hms(last_covered)} onward, in the same format, until you reach the end of the audio. Do NOT repeat any content you already transcribed before this timestamp. Do NOT include any preamble or explanation, just continue the transcript directly.

Format each turn exactly like this, with a blank line in between:

**[HH:MM:SS] Speaker Name**
Transcribed text for this turn."""
        contents = build_reference_parts() + [continue_prompt, myfile]

    return all_segments

def write_chunk(idx, segs, mode):
    offset_seconds = idx * chunk_seconds
    with open(output_path, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("## Meeting Transcript\n\n")
        for local_seconds, speaker, content in segs:
            abs_seconds = local_seconds + offset_seconds
            f.write(f"**[{format_hms(abs_seconds)}] {speaker}**\n\n{content}\n\n")

try:
    for idx, chunk_name in enumerate(chunk_files):
        chunk_path = os.path.join(chunk_dir, chunk_name)
        chunk_label = f"Chunk {idx + 1}/{len(chunk_files)}"
        chunk_duration = get_duration_seconds(chunk_path) or chunk_seconds

        old_segs = existing_chunk_segments.get(idx, [])
        old_coverage = old_segs[-1][0] if old_segs else 0

        log(f"--- {chunk_label} ({chunk_name}) --- real duration {format_hms(chunk_duration)}, previous coverage {format_hms(old_coverage)}")

        if old_segs and old_coverage >= chunk_duration - coverage_tolerance_seconds:
            log(f"  Already fully covered from existing output. Reusing without calling the API.")
            segs = old_segs
        else:
            log(f"  Incomplete or missing. Regenerating this chunk from scratch...")
            segs = transcribe_chunk_fully(chunk_path, chunk_duration, chunk_label)
            log(f"{chunk_label} done. ({len(segs)} segments)")

        # Write this chunk's portion immediately so progress survives a crash
        # or a quota cutoff partway through the run.
        write_chunk(idx, segs, "w" if idx == 0 else "a")

    log(f"All chunks processed. Final transcript saved to: {output_path}")
    log("Done.")

except DailyQuotaExceeded as e:
    log("")
    log("=" * 70)
    log("STOPPED: Daily API quota exceeded for this model on the free tier.")
    log("Chunks completed before this point were NOT lost - rerun this exact")
    log("script later (after the quota resets) and it will reuse everything")
    log("already fully covered, only regenerating what's still incomplete.")
    log("=" * 70)
    sys.exit(1)
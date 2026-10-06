<!--
  README.md -- Gemini long meeting transcription with speaker labels
  Author:  Jibril Sulaiman
  Date:    2026-10-06
  What:    Click-by-click guide for turning a multi-hour meeting recording into a
           timestamped Markdown transcript with speaker labels, using Python, ffmpeg and
           the Google Gemini API (Files API) on the free tier.
  Why:     One request for a 7+ hour recording runs out of output, loops and fails all at
           once; the free tier allows only a handful of requests a day per project; and
           speaker names come out wrong unless Gemini hears each voice first.
-->

# Gemini: long meeting transcription with speaker labels

Turn a long meeting recording (one hour or seven-plus hours) into a Markdown transcript
with a timestamp and a speaker name at every speaker change. A Python script splits the
audio into 45-minute chunks with ffmpeg, sends each chunk to the Google Gemini API,
shifts the timestamps back to the full meeting's clock, and stitches the result into one
file.

It is built to survive the free tier. It retries when Gemini is overloaded, re-asks for
any chunk that stopped short, stops cleanly when the daily quota runs out, and picks up
where it left off the next day. A 12-second voice sample of each attendee goes in with
every chunk so Gemini can match voices to names instead of guessing.

Setup takes about 30 minutes, most of it finding each person's voice sample. You need a
Google account, Python and ffmpeg. No billing, no GPU.

## Why it exists

The goal was simple: a written, speaker-labelled record of a planning meeting that ran
more than seven hours (about 500 MB of MP3). Not every chat assistant accepts audio, and
Gemini does: its Files API takes audio files up to 2 GB, so it looked like one upload
and one request. It wasn't:

1. **One request for the whole recording doesn't work.** The upload is fine, but a single
   response has an output limit far smaller than a seven-hour transcript. It's also all
   or nothing: one failure late in the request loses everything.
2. **The model writes its thinking into the transcript.** With thinking on, lines like
   *"wait, let me check"* land between the dialogue. Thinking has to be switched off and
   the prompt has to forbid commentary.
3. **Timestamps drift and repeat.** Gemini sometimes put several speaker turns on one line,
   and sometimes looped back and transcribed the same stretch twice. Timestamps then ran
   backwards in the stitched file.
4. **Chunks stop early without saying so.** A 45-minute chunk could come back covering 30
   minutes, and the transcript silently jumped ahead.
5. **The service is busy.** `503 UNAVAILABLE` hit partway through a run. Without retries
   and resume, the run was lost.
6. **The free tier is small, and it's per project, not per key.** The model allowed about
   20 requests per day on the free tier during the build. Quota belongs to the Google
   Cloud project, so a second key in the same project adds nothing. A key made on a
   different Google account outside AI Studio returned `403 PERMISSION_DENIED` until the
   key was created in AI Studio itself, which switches on the API it needs.
7. **Speaker names were poor.** Given only a list of names, Gemini guessed from context
   (who was addressed, who introduced themselves). Voices that never said their own name
   got the wrong label. Giving it a short labelled voice sample of each person, cut from
   their self-introduction, made the matching much better.

This repo is the version that handled all seven.

### Alternatives that were considered (not part of this build)

| Option | Why it wasn't used |
|---|---|
| pyannote.audio (speaker diarization) | Labels *Speaker A / B* only, never names; needs a Hugging Face token and a GPU to be practical (Google Colab's T4); slow on a laptop CPU |
| OpenAI Whisper, run locally | Good text, but no speaker separation at all |
| AssemblyAI or Deepgram | Speaker separation included, but paid per audio hour |

If you need exact speaker separation for legal or compliance records, a dedicated
diarization service is the better tool. This guide is for a good working transcript at
no cost.

## How it works

```text
 meeting.mp3 ──ffmpeg──► chunks/chunk_000.mp3 … chunk_009.mp3   (45 min each)
      │
      └──ffmpeg──► reference_clips/<Name>.mp3                     (12 s per attendee)

 for each chunk:
   already fully covered in transcript.md?  ── yes ──► reuse, no API call
          │ no
   upload chunk (Files API) ─► wait until ACTIVE
   generate_content( [voice samples + names] + strict prompt + chunk ), thinking off
   parse "**[HH:MM:SS] Name**" turns ─► drop any turn that goes backwards (loops)
   covered to the end?  ── no ──► ask to continue from the last timestamp (up to 6 times)
   shift timestamps by chunk number × 45 min ─► write to transcript.md
   503 ─► wait 30 s, 60 s, 90 s… and retry      429 daily quota ─► stop, rerun tomorrow
```

## What's in this repo

| Path | What it is | Where it runs |
|---|---|---|
| [`README.md`](README.md) | This guide | — |
| [`transcribe.py`](transcribe.py) | The transcription script (Steps 6 to 8) | Your computer, Python 3.9+ |
| [`list_models.py`](list_models.py) | Prints the Gemini models your key can call (Step 5) | Your computer, Python 3.9+ |
| [`.gitignore`](.gitignore) | Keeps recordings, chunks, voice clips and transcripts out of git | — |

## Table of contents

- [1. Requirements](#1-requirements)
- [2. Setup, step by step](#2-setup-step-by-step)
  - [Step 1: Install Python and the Gemini library](#step-1-install-python-and-the-gemini-library)
  - [Step 2: Install ffmpeg](#step-2-install-ffmpeg)
  - [Step 3: Create a Gemini API key in AI Studio](#step-3-create-a-gemini-api-key-in-ai-studio)
  - [Step 4: Put the key in an environment variable](#step-4-put-the-key-in-an-environment-variable)
  - [Step 5: Check which models your key can use](#step-5-check-which-models-your-key-can-use)
  - [Step 6: Set up the folder, speakers and voice samples](#step-6-set-up-the-folder-speakers-and-voice-samples)
  - [Step 7: Run it](#step-7-run-it)
  - [Step 8: Resume after a quota stop](#step-8-resume-after-a-quota-stop)
  - [Step 9: Review the transcript](#step-9-review-the-transcript)
- [Settings reference](#settings-reference)
- [Troubleshooting](#troubleshooting)
- [Limits](#limits)
- [Security and privacy](#security-and-privacy)

## 1. Requirements

| Need | Why |
|---|---|
| A Google account | To create a Gemini API key in Google AI Studio |
| Python 3.9 or later (built and run on 3.13) | Runs the scripts |
| The `google-genai` Python package | Google's Gemini library |
| ffmpeg and ffprobe on your PATH | Cut the recording into chunks and voice samples, and measure chunk length |
| The recording as an audio file (MP3 used here) | The script copies audio streams without re-encoding, so keep it a normal MP3 |
| About 10 minutes of listening | To note when each person first speaks clearly (Step 6) |

You do **not** need billing, a GPU or a Google Cloud console project set up by hand.

## 2. Setup, step by step

Commands are for **PowerShell on Windows** first, with macOS/Linux equivalents where they
differ.

### Step 1: Install Python and the Gemini library

About 5 minutes.

**1a.** Check Python:

```powershell
python --version
```

Anything from `Python 3.9` up works. If the command isn't found, install Python from
https://www.python.org/downloads/ (tick **Add python.exe to PATH** in the installer,
wording may differ) or from the Microsoft Store, then open a new terminal.

> ⚠️ **`python` or `python3`.** A Microsoft Store install answers to both. A python.org
> install on Windows usually answers to `python` only. Use whichever one printed a version
> for every command in this guide.

**1b.** Install the library:

```powershell
python -m pip install google-genai
```

**1c.** Test it:

```powershell
python -c "from google import genai; print('google-genai ok')"
```

✅ **Check:** it prints `google-genai ok`.

### Step 2: Install ffmpeg

About 5 minutes.

**2a.** On Windows, install with winget:

```powershell
winget install Gyan.FFmpeg
```

On macOS: `brew install ffmpeg`. On Debian/Ubuntu: `sudo apt install ffmpeg`.

**2b.** **Close the terminal and open a new one.** PATH changes only reach new windows.

**2c.** Test both tools (the script uses `ffprobe` too, to measure each chunk):

```powershell
ffmpeg -version
ffprobe -version
```

✅ **Check:** both print a version line starting `ffmpeg version` / `ffprobe version`.

### Step 3: Create a Gemini API key in AI Studio

About 3 minutes.

**3a.** Go to https://aistudio.google.com/apikey and sign in. The page heading is
**API Keys**, with a left menu of **Dashboard**, **API Keys**, **Projects**, **Usage**,
**Rate Limit**, **Spend** and **Billing**.

**3b.** If this is your first visit, accept the Gemini API terms when asked (wording may
differ).

**3c.** Click **Create API key** (top right). Choose an existing project or let AI Studio
create one (wording may differ). Copy the key that appears.

**3d.** Back on the **API Keys** list, your key shows its last four characters, the
**Project** it belongs to, the **Created** date and a **Billing Tier** of **Free tier**
(with a **Set up billing** link you don't need to click).

> ⚠️ **Create the key in AI Studio, not in the Google Cloud console.** AI Studio switches
> on the Generative Language API for the project as it makes the key. A key created on a
> different Google account's Cloud project returned `403 PERMISSION_DENIED` during the
> build. If AI Studio doesn't list a project you expect, the page says *"This list only
> shows API keys for projects imported into Google AI Studio."*; use **Import projects**
> or just create a new key here.

> ⚠️ **Quota is per project, not per key.** Two keys in the same project share one daily
> allowance. Check your real numbers under **Rate Limit** and **Usage** in the left menu.

> ⚠️ **Don't screenshot the key** or paste it into a chat or a ticket. If you do, delete
> the key in AI Studio and create a new one.

✅ **Check:** the key is copied, and the **API Keys** list shows it with **Free tier**.

### Step 4: Put the key in an environment variable

About 2 minutes. Both scripts read the key from `GEMINI_API_KEY`. It never goes in a file.

**4a. PowerShell, this window only:**

```powershell
$env:GEMINI_API_KEY = "PASTE_YOUR_KEY_HERE"
```

**4b. PowerShell, permanently** (then close and reopen the terminal):

```powershell
setx GEMINI_API_KEY "PASTE_YOUR_KEY_HERE"
```

**4c. macOS/Linux (bash or zsh):**

```bash
export GEMINI_API_KEY="PASTE_YOUR_KEY_HERE"
```

**4d.** Test that it's set without printing it:

```powershell
if ($env:GEMINI_API_KEY) { "key is set ($($env:GEMINI_API_KEY.Length) chars)" } else { "key is NOT set" }
```

> ⚠️ **`export` is bash, not PowerShell.** In PowerShell it fails with *"The term
> 'export' is not recognized as the name of a cmdlet, function, script file, or operable
> program."* Use `$env:` or `setx`.

> ⚠️ **`setx` doesn't affect the window you ran it in.** Open a new terminal, or also run
> the `$env:` line once.

> ⚠️ **Never put the key inside the script.** Both scripts read
> `os.environ["GEMINI_API_KEY"]`. That is the *name* of the variable. If you paste the
> key itself between those brackets, Python looks for a variable named after your key
> and fails with `KeyError: '<your key>'`, and the key is now in your file and on screen.

✅ **Check:** 4d prints `key is set (... chars)`.

### Step 5: Check which models your key can use

About 2 minutes.

**5a.** From the repo folder, run [`list_models.py`](list_models.py):

```powershell
python list_models.py
```

Runs [`list_models.py`](list_models.py).

It prints one model name per line, such as `models/gemini-flash-latest`.

**5b.** [`transcribe.py`](transcribe.py) calls **`gemini-flash-latest`**. That's an
alias Google points at its current Flash model, so the script doesn't break when a
dated model is retired. Confirm `models/gemini-flash-latest` is in your list. If it
isn't, pick another Flash model from the list and change the `model=` value inside
`generate_with_retry` in [`transcribe.py`](transcribe.py).

> ⚠️ **Fixed model names get retired.** The first version of this script named
> `gemini-2.5-flash` and failed with `404 NOT_FOUND`: *"This model
> models/gemini-2.5-flash is no longer available to new users. Please update your code
> to use a newer model for the latest features and improvements."* The alias avoids that.

✅ **Check:** the list prints and includes the model the script will call.

### Step 6: Set up the folder, speakers and voice samples

About 15 minutes, most of it listening.

**6a.** Put [`transcribe.py`](transcribe.py) and your recording in the same folder.
Either rename the recording to `meeting.mp3` or change `audio_path` at the top of the
script to its file name.

**6b.** Open [`transcribe.py`](transcribe.py) and fill in **`attendees`**: every person
who speaks, spelled the way you want them labelled in the transcript.

```python
attendees = [
    "Speaker Name 1", "Speaker Name 2", "Speaker Name 3", "Speaker Name 4",
]
```

**6c.** Find a clean voice sample for each person. Play the recording in any audio
player and, for each attendee, note the time when they speak **alone, clearly, for at
least 12 seconds**. Self-introductions near the start of a meeting are ideal. Write the
time down as hours:minutes:seconds, then convert to seconds (`00:04:27` = 4 × 60 + 27 =
`267`).

**6d.** Fill in **`reference_timestamps`** with those seconds:

```python
reference_timestamps = {
    "Speaker Name 1": 60,     # 00:01:00
    "Speaker Name 2": 95,     # 00:01:35
    "Speaker Name 3": 130,    # 00:02:10
    "Speaker Name 4": 170,    # 00:02:50
}
```

| Setting | What to put | Rule |
|---|---|---|
| `attendees` | Every speaker's name | These become the labels. Shown to Gemini as *"Likely speakers"*. |
| `reference_timestamps` | Name → start second of a 12-second solo sample | The name must match `attendees` **exactly**. The second is counted from the start of the **full** recording, not a chunk. |
| `reference_clip_duration` | `12` (default) | Seconds cut for each sample |

> ⚠️ **One voice per sample.** If someone else talks over the 12 seconds, Gemini learns
> the wrong voice and mislabels that person all meeting. Move the start time until the
> clip is one person only.

> ⚠️ **No good sample for someone?** Leave them out of `reference_timestamps` but keep
> them in `attendees`. Gemini then names them from context (being addressed by name,
> their role), and falls back to `Speaker 1`, `Speaker 2` when unsure.

> ⚠️ **Clips are cut once and reused.** They are saved to `reference_clips/` and not
> recut on later runs. If you change a timestamp, delete that person's file in
> `reference_clips/` first.

✅ **Check:** every name in `reference_timestamps` appears in `attendees`, character for
character.

### Step 7: Run it

About 5 to 10 minutes per 45-minute chunk, unattended.

**7a.** In the folder from Step 6, with the key set (Step 4):

```powershell
python transcribe.py
```

Runs [`transcribe.py`](transcribe.py).

**7b.** What happens, in order:

| Stage | Console shows | Creates |
|---|---|---|
| Split | `Splitting audio into 45-minute chunks with ffmpeg...` then `Split complete. 10 chunk(s) created.` | `chunks/chunk_000.mp3`, `chunk_001.mp3`, … |
| Resume check | `No existing output file found. Starting fresh.` | — |
| Voice samples | `Extracting reference clip for <Name> (from 00:01:00)...` then `Uploaded 4 of 4 reference voice samples.` | `reference_clips/<Name>.mp3` |
| Each chunk | `--- Chunk 1/10 (chunk_000.mp3) --- real duration 00:45:00, previous coverage 00:00:00` | — |
| Upload | `Uploading chunk to Gemini...`, `Upload complete in …s. State: …`, then `Still processing chunk... (check #1)` every 5 s until ready | — |
| Transcribe | `Sending transcription request (attempt 1)...`, `Response received in …s.`, `Coverage so far: up to 00:44:52 of 00:45:00.`, `Full chunk coverage reached.` | — |
| Save | `Chunk 1/10 done. (… segments)` | Appends the chunk to `transcript.md` |
| Finish | `All chunks processed. Final transcript saved to: transcript.md` then `Done.` | — |

A seven-hour-plus recording makes 10 chunks. Each chunk is written to `transcript.md` as
soon as it finishes, so you can open the file while the run continues.

**7c.** If you see these, the script is handling them; let it run:

| Console line | Meaning |
|---|---|
| `Server error (attempt 2/5): 503 UNAVAILABLE ... Retrying in 60s...` | Gemini overloaded. Waits 30 s, 60 s, 90 s… (max 180 s), up to 5 tries. |
| `Chunk stopped early at 00:31:10. Requesting continuation...` | The coverage check caught a short answer. It asks Gemini to continue from that time, up to 6 more requests per chunk. |
| `WARNING: dropped 3 repeated/looped segment(s) from this response.` | Gemini looped back. Turns with a timestamp earlier than one already kept are thrown away. |
| `WARNING: Hit max continuation attempts...` | 6 continuations weren't enough. The chunk is saved as far as it got and will be retried next run (Step 8). |

> ⚠️ **Count your requests.** Every chunk costs at least one request, and each
> continuation costs another. Ten chunks fit in a 20-a-day allowance only if most
> answer in one go. A long silence at the end of a chunk (a break, a muted room) can
> look like a short answer and use extra continuations.

> ⚠️ **Thinking is off on purpose.** `thinking_budget=0` in the script stops the model's
> reasoning ("wait, let me check") from landing in the transcript. Don't turn it back on.

✅ **Check:** `transcript.md` grows chunk by chunk, and the run ends with `Done.` or with
the quota message in Step 8.

### Step 8: Resume after a quota stop

About 1 minute, the next day.

**8a.** When the free daily allowance runs out, the script stops itself instead of
retrying:

```text
======================================================================
STOPPED: Daily API quota exceeded for this model on the free tier.
Chunks completed before this point were NOT lost - rerun this exact
script later (after the quota resets) and it will reuse everything
already fully covered, only regenerating what's still incomplete.
======================================================================
```

**8b.** Make a copy of `transcript.md` (for example `transcript-backup.md`). See the
warning below.

**8c.** After the quota resets (check **Rate Limit** in AI Studio), run the same command
again in the same folder:

```powershell
python transcribe.py
```

Runs [`transcribe.py`](transcribe.py).

It reads `transcript.md`, works out how far each chunk got, and prints
`Already fully covered from existing output. Reusing without calling the API.` for each
finished chunk. Only missing or short chunks are sent to Gemini.

> ⚠️ **Back up `transcript.md` before a rerun.** Each run rewrites the file from the
> first chunk, in order. If the quota runs out again partway through, chunks *after*
> the one it stopped on are not written back, even if an earlier run had finished them.
> They are still in your backup copy. This matters most when you rerun to repair a
> short chunk early in the meeting.

> ⚠️ **Leave `chunks/`, `reference_clips/` and `transcript.md` where they are.** That's
> the resume state. (`progress.json` is not used by this version; resume works from
> `transcript.md`.) Uploaded files on Gemini's side expire after 48 hours, which doesn't
> matter here because each run uploads again.

> ⚠️ **Changed `chunk_seconds`?** Delete `chunks/` and `transcript.md` and start fresh.
> Old chunks are reused as they are, and the timestamp shift assumes every chunk is
> `chunk_seconds` long.

✅ **Check:** the rerun skips finished chunks without API calls and continues from the
first incomplete one.

### Step 9: Review the transcript

About 15 minutes for a spot check.

**9a.** Open `transcript.md`. Every turn looks like this, with absolute times from the
start of the recording:

```markdown
## Meeting Transcript

**[01:32:05] Speaker Name 2**

Okay, so the next item is the launch date.

**[01:32:11] Speaker Name 1**

We moved it a week. I'll send the new plan tonight.
```

**9b.** Spot-check three places against the audio: near the start, near a chunk
boundary (00:45:00, 01:30:00, 02:15:00…), and near the end. Check the speaker names and
that the timestamps line up within a few seconds.

**9c.** Search the file for `Speaker 1`, `Speaker 2`. Those are turns Gemini couldn't
match. Usually one find-and-replace per chunk fixes them, because the label stays
consistent within a chunk (not across chunks).

> ⚠️ **Speaker names are best effort.** Voice samples improve them a lot, but two people
> with similar voices can still be swapped. Don't use the labels as a formal record
> without checking.

✅ **Check:** timestamps only move forward through the file, and the last timestamp is
close to the recording's length.

## Settings reference

All at the top of [`transcribe.py`](transcribe.py):

| Setting | Default | What it does |
|---|---|---|
| `audio_path` | `meeting.mp3` | The recording to transcribe |
| `chunk_seconds` | `2700` | Chunk length (45 minutes). Long enough to keep context, short enough for one answer. |
| `chunk_dir` | `chunks` | Where ffmpeg writes the chunks |
| `output_path` | `transcript.md` | The transcript, and the resume state |
| `progress_path` | `progress.json` | Not used by this version |
| `max_retries` | `5` | Tries per request on a server error (and on non-daily client errors) |
| `max_continuations_per_chunk` | `6` | Extra "continue from here" requests when a chunk stops short |
| `coverage_tolerance_seconds` | `45` | A chunk counts as complete when its last turn is within this many seconds of the end |
| `attendees` | placeholders | Speaker names (Step 6) |
| `reference_timestamps` | placeholders | Start second of each person's voice sample (Step 6) |
| `reference_clip_duration` | `12` | Length of each voice sample, in seconds |
| `reference_clip_dir` | `reference_clips` | Where the voice samples are saved |

Environment variable: `GEMINI_API_KEY` (Step 4). Model: `gemini-flash-latest`
(Step 5).

## Troubleshooting

| You got | Cause | Fix |
|---|---|---|
| *"The term 'export' is not recognized..."* | Bash syntax in PowerShell | Use `$env:` or `setx` (Step 4) |
| `KeyError: 'GEMINI_API_KEY'` | Variable not set in this window | Step 4; after `setx`, open a new terminal |
| `KeyError: '<your actual key>'` | The key was pasted into `os.environ[...]` | Put `"GEMINI_API_KEY"` back in the script; set the key in the environment (Step 4). Replace the exposed key. |
| `404 NOT_FOUND` ... *"no longer available to new users"* | Retired model name | Step 5 |
| `403 PERMISSION_DENIED` | Key from a project without the Generative Language API, or terms not accepted | Create the key in AI Studio (Step 3) |
| The quota STOPPED message on the first chunk | Today's allowance already used (shared by every key in the project) | Wait for the reset, or use a key in a different project (Step 3) |
| `503 UNAVAILABLE` five times, then a crash | Gemini overloaded for several minutes | Rerun later; finished chunks are kept (Step 8) |
| `ERROR: File not found at meeting.mp3` | Wrong name or folder | Step 6a |
| `ERROR: ffmpeg failed to split the audio.` or `ffmpeg` not recognized | ffmpeg not installed or not on PATH in this window | Step 2, then a new terminal |
| `WARNING: Failed to extract reference clip for <Name>` | Timestamp past the end of the file, or ffmpeg missing | Fix the second in `reference_timestamps` (Step 6d) |
| Thinking text (*"wait, let me check"*) in the transcript | Thinking turned back on, or prompt edited | Keep `thinking_budget=0` and the prompt rules (Step 7) |
| Timestamps jump back or repeat | A model loop that slipped past the filter, or chunk files from a different `chunk_seconds` | Step 8 (fresh start if `chunk_seconds` changed) |
| A section of the meeting is missing | A chunk hit the continuation cap | Rerun (Step 8); it regenerates short chunks |
| Chunks that were finished disappeared after a rerun | The rerun stopped on quota before rewriting them | Restore from your backup (Step 8b) |
| Many `Speaker 1` / `Speaker 2` labels | Weak or missing voice samples | Better 12-second solo samples (Step 6c); delete old clips first |

## Limits

- **Free-tier quota.** About 20 requests per day per project for the model used during
  the build. Google changes these numbers; your real limits are on the **Rate Limit**
  page in AI Studio.
- **Speaker labels are best effort,** not diarization. Similar voices can be swapped.
  Unmatched voices are labelled per chunk, so `Speaker 1` in one chunk is not
  necessarily `Speaker 1` in the next.
- **Chunk boundaries cut mid-sentence.** A sentence spanning 00:45:00 can be split or
  have a word dropped.
- **Only turn headers are shifted.** A timestamp the model writes inside the text of a
  turn stays chunk-relative.
- **Files API limits.** Up to 2 GB per file; files are deleted after 48 hours.
- **Audio format.** Chunks and clips are cut with stream copy (`-c copy`), so they keep
  the source format and file extension `.mp3`. Convert other formats to MP3 first
  (`ffmpeg -i input.m4a meeting.mp3`).

## Security and privacy

- The API key reads and spends against your Google project. Keep it in the environment
  only. If it ever appears in a screenshot, a chat, a commit or a pasted command, delete
  it in AI Studio and create a new one.
- **Free-tier content may be used by Google to improve its products.** Check Google's
  current Gemini API terms before sending a confidential meeting. A paid tier has
  different terms.
- Recordings, chunks, voice clips and transcripts contain what people said. The
  [`.gitignore`](.gitignore) keeps them out of git; don't commit them anywhere else
  either, and delete `chunks/` and `reference_clips/` when you're done.
- Tell attendees the meeting is being transcribed, and follow your workplace's rules on
  recording.

---

Built by [Jibril Sulaiman](https://github.com/carljibrilsulaimanii).

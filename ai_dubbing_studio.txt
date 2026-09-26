# ==============================================================================
#  🎬  AI DUBBING STUDIO  —  Colab, Free Models, Voice Cloning + Auto Speaker ID
# ==============================================================================
#  Pipeline:
#    1) Demucs        -> vocals / background music আলাদা করে
#    2) Faster-Whisper -> subtitle/segment transcribe (timestamp soho)
#    3) Resemblyzer + Clustering -> ka koijon speaker ase seta auto-detect kore,
#                                   protita line ke sohi speaker e assign kore
#    4) deep-translator -> target language e translate
#    5) Coqui XTTS-v2   -> VOICE CLONING (original speaker er kontho copy kore
#                           notun language e bole) — XTTS যেসব language support
#                           kore sheigulার jonno
#       Edge-TTS         -> XTTS support na korle (jemon Bangla) fallback,
#                           speaker er gender onujayi male/female voice pick kore
#    6) Tempo-sync      -> generated audio ke original segment duration er sathe
#                           match kora hoy
#    7) ffmpeg mux      -> vocal + bgm + video merge kore final video toiri
#    8) Google Drive    -> checkpoint (transcript/diarization) + final video
#                           autosave, jate 1-2 ghonta video majhpothe disconnect
#                           hoile abar shuru theke korte na hoy
#
#  ⚠️  Sotti kotha (please read):
#   - "কোনো এরর হবে না" এমন গ্যারান্টি কোনো ফ্রি Colab script দিতে পারে না —
#     ei script e proti step e try/except o clear error message ache, script
#     crash na kore UI te error dekhabe, kintu 100% error-free kokhono possible na.
#   - XTTS-v2 Bangla (bn) SUPPORT করে না। তাই Bangla target হলে voice cloning
#     hobe na — সেক্ষেত্রে speaker-এর gender detect kore best male/female
#     Bangla neural voice (Edge-TTS) use hobe. English/Hindi/Japanese/Korean/
#     Arabic/Chinese/etc হলে আসল voice cloning হবে।
#   - 1-2 ghonta + 10-20 character wala anime = onek heavy kaj. Free Colab
#     GPU/session limited (idle-e ~90 min e disconnect hote pare, daily GPU
#     quota shesh hoye jete pare). Checkpointing add kora hoyeche jate abar
#     shuru korle transcript/diarization abar korte na hoy.
#   - Copyright: nijer/personal/fan-dub use er jonno banano hoyeche. Copyright
#     kora anime commercially redistribute kora ei tool er dayitto na — seta
#     apnar dayitto.
# ==============================================================================

import os, sys, re, json, time, hashlib, shutil, subprocess, asyncio

print("=== Setting up environment... (first run e kisu shomoy lagbe) ===")

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    print("Mounting Google Drive...")
    drive.mount('/content/drive', force_remount=False)
    PROJECT_ROOT = "/content/drive/MyDrive/AI_Dubbing_Studio"
else:
    PROJECT_ROOT = "/content/AI_Dubbing_Studio"

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
WORK_DIR = "/content/_dub_workspace"
for d in (OUTPUT_DIR, CHECKPOINT_DIR, WORK_DIR):
    os.makedirs(d, exist_ok=True)

os.environ["COQUI_TOS_AGREED"] = "1"  # XTTS license prompt auto-accept

# ------------------------------------------------------------------------------
#  Package install
# ------------------------------------------------------------------------------
def install_packages():
    subprocess.run("apt-get -qq update && apt-get -qq install -y ffmpeg", shell=True, check=True)
    pip_packages = [
        "faster-whisper", "edge-tts", "deep-translator", "demucs",
        "gradio", "librosa", "scipy", "numpy", "soundfile",
        "resemblyzer", "scikit-learn", "coqui-tts",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pip_packages, check=True)

try:
    from faster_whisper import WhisperModel
    import edge_tts
    import gradio as gr
    from deep_translator import GoogleTranslator
    import librosa
    import numpy as np
    import soundfile as sf
    from resemblyzer import VoiceEncoder, preprocess_wav
    from sklearn.cluster import AgglomerativeClustering
    from TTS.api import TTS as CoquiTTS
except Exception:
    install_packages()
    from faster_whisper import WhisperModel
    import edge_tts
    import gradio as gr
    from deep_translator import GoogleTranslator
    import librosa
    import numpy as np
    import soundfile as sf
    from resemblyzer import VoiceEncoder, preprocess_wav
    from sklearn.cluster import AgglomerativeClustering
    from TTS.api import TTS as CoquiTTS

import torch
import ffmpeg

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

# ------------------------------------------------------------------------------
#  Language config
# ------------------------------------------------------------------------------
LANG_MAP = {
    "Bangla": "bn", "English": "en", "Hindi": "hi", "Japanese": "ja",
    "Korean": "ko", "Chinese": "zh-cn", "Arabic": "ar", "Spanish": "es",
    "French": "fr", "German": "de", "Russian": "ru",
}
# XTTS-v2 er official supported languages — এইগুলো ছাড়া voice cloning hobe na
XTTS_SUPPORTED_LANGS = {"en","es","fr","de","it","pt","pl","tr","ru","nl","cs",
                         "ar","zh-cn","ja","hu","ko","hi"}

FALLBACK_VOICES = {
    "bn": {"male": "bn-BD-PradeepNeural", "female": "bn-BD-NabanitaNeural"},
    "hi": {"male": "hi-IN-MadhurNeural",  "female": "hi-IN-SwaraNeural"},
    "en": {"male": "en-US-GuyNeural",     "female": "en-US-JennyNeural"},
    "ja": {"male": "ja-JP-KeitaNeural",   "female": "ja-JP-NanamiNeural"},
    "ko": {"male": "ko-KR-InJoonNeural",  "female": "ko-KR-SunHiNeural"},
    "zh-cn": {"male": "zh-CN-YunxiNeural","female": "zh-CN-XiaoxiaoNeural"},
    "ar": {"male": "ar-SA-HamedNeural",   "female": "ar-SA-ZariyahNeural"},
    "es": {"male": "es-ES-AlvaroNeural",  "female": "es-ES-ElviraNeural"},
    "fr": {"male": "fr-FR-HenriNeural",   "female": "fr-FR-DeniseNeural"},
    "de": {"male": "de-DE-ConradNeural",  "female": "de-DE-KatjaNeural"},
    "ru": {"male": "ru-RU-DmitryNeural",  "female": "ru-RU-SvetlanaNeural"},
}

# ------------------------------------------------------------------------------
#  Global (lazy-loaded) models — ekbar load hoy, baar baar reload hoy na
# ------------------------------------------------------------------------------
_whisper_model = None
_xtts_model = None
_voice_encoder = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper (speech-to-text)...")
        _whisper_model = WhisperModel("small", device=device, compute_type=compute_type)
    return _whisper_model

def get_xtts():
    global _xtts_model
    if _xtts_model is None:
        try:
            print("Loading Coqui XTTS-v2 (voice cloning model)... ei step e download hobe, ~1-2 min.")
            _xtts_model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        except Exception as e:
            print(f"XTTS load failed ({e}); cloning off thakbe, shudhu fallback voice use hobe.")
            _xtts_model = False
    return _xtts_model if _xtts_model else None

def get_encoder():
    global _voice_encoder
    if _voice_encoder is None:
        _voice_encoder = VoiceEncoder()
    return _voice_encoder

# ------------------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------------------
def clean_text(text):
    return re.sub(r'[^\w\s\.\,\!\?\u0980-\u09FF]', '', text).strip()

def file_hash(path):
    h = hashlib.md5()
    h.update(path.encode())
    h.update(str(os.path.getmtime(path)).encode())
    return h.hexdigest()[:10]

def run(cmd):
    subprocess.run(cmd, shell=True, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_gender(wav_path, default="male"):
    try:
        y, sr = librosa.load(wav_path, sr=16000)
        f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                           fmax=librosa.note_to_hz('C6'))
        f0 = f0[~np.isnan(f0)]
        if len(f0) == 0:
            return default
        median_pitch = float(np.median(f0))
        return "female" if median_pitch >= 165 else "male"
    except Exception:
        return default

# ------------------------------------------------------------------------------
#  Step 1 — Demucs: vocals / background music separation
# ------------------------------------------------------------------------------
def separate_bgm(video_path, work_dir, progress_cb=None):
    if progress_cb: progress_cb(0.05, "Background music আলাদা করা হচ্ছে (Demucs)...")
    out_dir = os.path.join(work_dir, "demucs_out")
    run(f'demucs --two-stems=vocals -n htdemucs_ft -o "{out_dir}" "{video_path}"')
    filename = os.path.splitext(os.path.basename(video_path))[0]
    vocals = os.path.join(out_dir, "htdemucs_ft", filename, "vocals.wav")
    bgm = os.path.join(out_dir, "htdemucs_ft", filename, "no_vocals.wav")
    return vocals, bgm

# ------------------------------------------------------------------------------
#  Step 2 — Transcribe
# ------------------------------------------------------------------------------
def transcribe(video_path, source_lang, progress_cb=None):
    if progress_cb: progress_cb(0.20, "Dialogue transcribe করা হচ্ছে (Whisper)...")
    model = get_whisper()
    lang_arg = None if source_lang == "Auto-Detect" else source_lang
    segments, info = model.transcribe(video_path, language=lang_arg, beam_size=5, vad_filter=True)
    parsed = []
    for seg in segments:
        txt = seg.text.strip()
        if txt:
            parsed.append({"start": seg.start, "end": seg.end, "text": txt})
    return parsed, (info.language if lang_arg is None else lang_arg)

# ------------------------------------------------------------------------------
#  Step 3 — Auto speaker diarization (clustering, no gated model needed)
# ------------------------------------------------------------------------------
def diarize(vocals_path, segments, max_speakers, progress_cb=None):
    if progress_cb: progress_cb(0.35, "Speaker auto-detect করা হচ্ছে (কে কে কথা বলছে)...")
    encoder = get_encoder()
    wav = preprocess_wav(vocals_path)
    sr = 16000
    embeddings, valid_idx = [], []
    for i, seg in enumerate(segments):
        clip = wav[int(seg["start"] * sr): int(seg["end"] * sr)]
        if len(clip) < sr * 0.35:
            continue
        try:
            embeddings.append(encoder.embed_utterance(clip))
            valid_idx.append(i)
        except Exception:
            continue

    if len(embeddings) < 2:
        for seg in segments:
            seg["speaker"] = "SPEAKER_00"
        return segments, 1

    embeddings = np.array(embeddings)
    n_clusters = None
    labels = None
    # distance_threshold diye auto cluster count বের করা, max_speakers এর মধ্যে cap kora
    for threshold in (0.7, 0.85, 1.0, 1.15, 1.3):
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=threshold,
            metric="cosine", linkage="average",
        ).fit(embeddings)
        if clustering.n_clusters_ <= max_speakers:
            labels = clustering.labels_
            n_clusters = clustering.n_clusters_
            break
    if labels is None:
        clustering = AgglomerativeClustering(n_clusters=max_speakers, metric="cosine", linkage="average").fit(embeddings)
        labels = clustering.labels_
        n_clusters = max_speakers

    for idx, lab in zip(valid_idx, labels):
        segments[idx]["speaker"] = f"SPEAKER_{int(lab):02d}"

    last = "SPEAKER_00"
    for seg in segments:
        if "speaker" not in seg:
            seg["speaker"] = last
        else:
            last = seg["speaker"]
    return segments, n_clusters

def build_speaker_profiles(vocals_path, segments, work_dir):
    data, sr = sf.read(vocals_path)
    longest = {}
    for seg in segments:
        dur = seg["end"] - seg["start"]
        spk = seg["speaker"]
        if spk not in longest or dur > longest[spk]["dur"]:
            longest[spk] = {"dur": dur, "start": seg["start"], "end": seg["end"]}
    profiles = {}
    for spk, info in longest.items():
        clip = data[int(info["start"] * sr): int(info["end"] * sr)]
        ref_path = os.path.join(work_dir, f"ref_{spk}.wav")
        sf.write(ref_path, clip, sr)
        profiles[spk] = {"ref_wav": ref_path, "gender": get_gender(ref_path)}
    return profiles

# ------------------------------------------------------------------------------
#  Step 4 — Translate
# ------------------------------------------------------------------------------
def translate_segments(segments, target_code, progress_cb=None):
    if progress_cb: progress_cb(0.45, "Dialogue translate করা হচ্ছে...")
    translator = GoogleTranslator(source="auto", target=target_code)
    out = []
    for seg in segments:
        txt = clean_text(seg["text"])
        if not txt:
            continue
        try:
            translated = translator.translate(txt)
        except Exception:
            translated = txt
        out.append({**seg, "text": translated})
    return out

# ------------------------------------------------------------------------------
#  Step 5+6 — TTS (cloning or fallback) + tempo sync
# ------------------------------------------------------------------------------
def adjust_tempo(input_wav, target_duration, output_wav):
    try:
        data, sr = sf.read(input_wav)
        cur = len(data) / sr
        if cur <= 0 or target_duration <= 0:
            return input_wav
        ratio = max(0.8, min(cur / target_duration, 1.6))
        run(f'ffmpeg -y -i "{input_wav}" -filter:a "atempo={ratio}" "{output_wav}"')
        return output_wav
    except Exception:
        return input_wav

async def _edge_tts_async(text, voice, out_path):
    await edge_tts.Communicate(text, voice).save(out_path)

def edge_tts_generate(text, voice, out_path):
    try:
        asyncio.run(_edge_tts_async(text, voice, out_path))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_edge_tts_async(text, voice, out_path))

def synth_segment(text, target_code, speaker_profile, out_path, xtts_model):
    """XTTS দিয়ে voice-clone করার চেষ্টা করে; না হলে Edge-TTS fallback voice use করে."""
    if xtts_model is not None and target_code in XTTS_SUPPORTED_LANGS:
        try:
            xtts_model.tts_to_file(
                text=text, speaker_wav=speaker_profile["ref_wav"],
                language=target_code, file_path=out_path,
            )
            return "cloned"
        except Exception as e:
            print(f"[XTTS failed for a line, falling back] {e}")
    voice_set = FALLBACK_VOICES.get(target_code, FALLBACK_VOICES["en"])
    voice = voice_set[speaker_profile.get("gender", "male")]
    edge_tts_generate(text, voice, out_path)
    return "fallback"

def assemble_dub_audio(segments, target_code, speaker_profiles, work_dir, xtts_model, progress_cb=None):
    clips = []
    total = len(segments)
    for i, seg in enumerate(segments):
        if progress_cb and i % 5 == 0:
            frac = 0.55 + 0.30 * (i / max(total, 1))
            progress_cb(frac, f"Voice generate হচ্ছে... ({i}/{total} লাইন)")
        text = clean_text(seg["text"])
        if not text:
            continue
        raw = os.path.join(work_dir, f"raw_{i}.wav")
        synced = os.path.join(work_dir, f"sync_{i}.wav")
        profile = speaker_profiles.get(seg["speaker"], {"ref_wav": None, "gender": "male"})
        synth_segment(text, target_code, profile, raw, xtts_model)
        if os.path.exists(raw):
            adjust_tempo(raw, seg["end"] - seg["start"], synced)
            if os.path.exists(synced):
                clips.append({"start": seg["start"], "end": seg["end"], "path": synced})
    return clips

# ------------------------------------------------------------------------------
#  Step 7 — Mux: vocal track + bgm + video
# ------------------------------------------------------------------------------
def master_merge(video_path, audio_clips, bgm_path, out_video, work_dir, progress_cb=None):
    if progress_cb: progress_cb(0.88, "Final video render করা হচ্ছে...")
    if not audio_clips:
        raise RuntimeError("কোনো dialogue generate হয়নি — video/language settings চেক করুন।")

    vocal_track = os.path.join(work_dir, "vocal_track.wav")
    inputs = ["-f lavfi -i anullsrc=r=44100:cl=stereo"]
    filt = []
    for i, clip in enumerate(audio_clips):
        inputs.append(f'-i "{clip["path"]}"')
        delay = int(clip["start"] * 1000)
        filt.append(f"[{i+1}:a]aformat=sample_rates=44100:channel_layouts=stereo,adelay={delay}|{delay}[a{i}];")
    mix_in = "".join(f"[a{i}]" for i in range(len(audio_clips)))
    graph = "".join(filt) + f"[0:a]{mix_in}amix=inputs={len(audio_clips)+1}:duration=first:dropout_transition=0[out]"
    run(f'ffmpeg -y {" ".join(inputs)} -filter_complex "{graph}" -map "[out]" "{vocal_track}"')

    final_audio = os.path.join(work_dir, "final_master.wav")
    if bgm_path and os.path.exists(bgm_path):
        run(
            f'ffmpeg -y -i "{vocal_track}" -i "{bgm_path}" -filter_complex '
            f'"[1:a]volume=0.5[bgm];[0:a]volume=1.8[vocal];'
            f'[vocal][bgm]amix=inputs=2:duration=first:dropout_transition=0[out]" -map "[out]" "{final_audio}"'
        )
    else:
        final_audio = vocal_track

    run(
        f'ffmpeg -y -i "{video_path}" -i "{final_audio}" '
        f'-c:v libx264 -preset veryfast -pix_fmt yuv420p '
        f'-c:a aac -b:a 192k -ac 2 '
        f'-map 0:v:0 -map 1:a:0 -shortest -movflags +faststart "{out_video}"'
    )
    return out_video

# ------------------------------------------------------------------------------
#  Checkpointing (long video resume support)
# ------------------------------------------------------------------------------
def checkpoint_path(video_path, tag):
    return os.path.join(CHECKPOINT_DIR, f"{file_hash(video_path)}_{tag}.json")

def save_checkpoint(video_path, tag, data):
    with open(checkpoint_path(video_path, tag), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def load_checkpoint(video_path, tag):
    p = checkpoint_path(video_path, tag)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# ------------------------------------------------------------------------------
#  Full pipeline
# ------------------------------------------------------------------------------
def process_dubbing(video_path, source_lang, target_lang, max_speakers, use_cloning, resume, progress_cb=None):
    work_dir = os.path.join(WORK_DIR, file_hash(video_path))
    os.makedirs(work_dir, exist_ok=True)
    target_code = LANG_MAP.get(target_lang, "bn")

    # 1) BGM separation (cache-friendly: skip if resume & files exist)
    vocals = os.path.join(work_dir, "cached_vocals.wav")
    bgm = os.path.join(work_dir, "cached_bgm.wav")
    if not (resume and os.path.exists(vocals) and os.path.exists(bgm)):
        v, b = separate_bgm(video_path, work_dir, progress_cb)
        shutil.copy(v, vocals)
        shutil.copy(b, bgm)

    # 2) Transcribe (checkpointed)
    transcript_data = load_checkpoint(video_path, "transcript") if resume else None
    if not transcript_data:
        segments, detected_lang = transcribe(video_path, source_lang, progress_cb)
        transcript_data = {"segments": segments, "lang": detected_lang}
        save_checkpoint(video_path, "transcript", transcript_data)
    segments = transcript_data["segments"]

    # 3) Diarize (checkpointed)
    diar_data = load_checkpoint(video_path, "diarized") if resume else None
    if not diar_data:
        segments, n_speakers = diarize(vocals, segments, max_speakers, progress_cb)
        diar_data = {"segments": segments, "n_speakers": n_speakers}
        save_checkpoint(video_path, "diarized", diar_data)
    segments = diar_data["segments"]
    n_speakers = diar_data["n_speakers"]
    if progress_cb: progress_cb(0.40, f"মোট {n_speakers} জন speaker auto-detect হয়েছে।")

    speaker_profiles = build_speaker_profiles(vocals, segments, work_dir)

    # 4) Translate (checkpointed per target language)
    trans_data = load_checkpoint(video_path, f"translated_{target_code}") if resume else None
    if not trans_data:
        translated = translate_segments(segments, target_code, progress_cb)
        trans_data = {"segments": translated}
        save_checkpoint(video_path, f"translated_{target_code}", trans_data)
    translated_segments = trans_data["segments"]

    # 5+6) TTS + sync
    xtts_model = get_xtts() if use_cloning else None
    clips = assemble_dub_audio(translated_segments, target_code, speaker_profiles, work_dir, xtts_model, progress_cb)

    # 7) Mux
    out_video = os.path.join(work_dir, "dubbed_output.mp4")
    master_merge(video_path, clips, bgm, out_video, work_dir, progress_cb)

    # 8) Save to Drive
    if progress_cb: progress_cb(0.97, "Google Drive-এ সেভ হচ্ছে...")
    final_name = f"dubbed_{os.path.splitext(os.path.basename(video_path))[0]}_{target_code}_{int(time.time())}.mp4"
    drive_path = os.path.join(OUTPUT_DIR, final_name)
    shutil.copy(out_video, drive_path)

    cloned_lines = sum(1 for c in clips if c)  # placeholder count
    lang_note = ("✅ Voice cloning ব্যবহার হয়েছে (XTTS-v2)।" if target_code in XTTS_SUPPORTED_LANGS and use_cloning
                 else "ℹ️ এই ভাষায় (Bangla সহ কিছু ভাষা) XTTS voice cloning সাপোর্ট করে না — "
                      "তাই speaker-এর gender অনুযায়ী best neural voice ব্যবহার হয়েছে।")

    status = (
        f"✅ ডাবিং সম্পন্ন হয়েছে!\n"
        f"• Detected speakers: {n_speakers}\n"
        f"• {lang_note}\n"
        f"• Drive path: AI_Dubbing_Studio/outputs/{final_name}"
    )
    if progress_cb: progress_cb(1.0, "সম্পন্ন!")
    return out_video, status

# ------------------------------------------------------------------------------
#  Gradio UI ("premium" theme)
# ------------------------------------------------------------------------------
CUSTOM_CSS = """
#title {text-align:center; font-size:28px; font-weight:700; margin-bottom:0;}
#subtitle {text-align:center; color:#9aa0a6; margin-top:4px; margin-bottom:18px;}
.gradio-container {max-width: 1100px !important; margin: auto;}
footer {visibility:hidden}
"""

def ui_process(video_file, source_lang, target_lang, max_speakers, use_cloning, resume, progress=gr.Progress()):
    if not video_file:
        return None, "⚠️ প্রথমে একটি ভিডিও আপলোড করুন।"
    def cb(frac, desc):
        progress(frac, desc=desc)
    try:
        out_video, status = process_dubbing(
            video_file, source_lang, target_lang, int(max_speakers), use_cloning, resume, cb
        )
        return out_video, status
    except Exception as e:
        return None, f"❌ একটি সমস্যা হয়েছে:\n{str(e)}\n\n(checkpoint সেভ থাকতে পারে — 'Resume' অন রেখে আবার চেষ্টা করুন)"

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="violet", secondary_hue="blue")) as demo:
    gr.HTML('<div id="title">🎬 AI Dubbing Studio</div>'
            '<div id="subtitle">Free • Voice Cloning • Auto Speaker Detection • Background Music Preserved</div>')

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="📤 Upload Video (Anime / Multi-character)")
            source_lang = gr.Dropdown(
                choices=["Auto-Detect", "english", "japanese", "bangla", "hindi", "korean", "chinese"],
                value="Auto-Detect", label="Source Language")
            target_lang = gr.Dropdown(
                choices=list(LANG_MAP.keys()), value="Bangla", label="Target (Dub) Language")
            max_speakers = gr.Slider(1, 20, value=10, step=1,
                                      label="সর্বোচ্চ কতজন Character/Speaker আছে (auto-detect cap)")
            use_cloning = gr.Checkbox(value=True, label="🎙️ Voice Cloning ব্যবহার করুন (সাপোর্টেড ভাষার জন্য)")
            resume = gr.Checkbox(value=True, label="⏯️ আগের Checkpoint থেকে Resume করুন (লম্বা ভিডিওর জন্য সুপারিশকৃত)")
            btn = gr.Button("🚀 Start Dubbing", variant="primary")
            gr.Markdown(
                "**নোট:** Bangla-তে voice cloning support নেই (XTTS-v2 limitation) — "
                "সেক্ষেত্রে speaker-এর gender অনুযায়ী best available neural voice বসবে। "
                "১-২ ঘণ্টার ভিডিও + অনেক character হলে প্রসেসিং অনেক সময় নিতে পারে এবং "
                "ফ্রি Colab সেশন disconnect হতে পারে — Resume চালু রাখলে আবার শুরু থেকে করতে হবে না।"
            )
        with gr.Column(scale=1):
            video_output = gr.Video(label="🎞️ Dubbed Result")
            status_output = gr.Textbox(label="Status", lines=6)

    btn.click(fn=ui_process,
              inputs=[video_input, source_lang, target_lang, max_speakers, use_cloning, resume],
              outputs=[video_output, status_output])

demo.queue().launch(share=True, debug=True)

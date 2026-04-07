"""
UniverSR — Audio Super-Resolution GUI
Gerçek API: UniverSR.from_pretrained() + model.enhance()

Kurulum:
    pip install gradio torch torchaudio einops omegaconf \
                librosa matplotlib soundfile requests tqdm torchdiffeq pillow

Çalıştırma:
    python universr_app.py
"""

import io, os, sys, warnings, subprocess, tempfile, textwrap, time
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torchaudio.transforms as TAT
import librosa, librosa.display
import gradio as gr
from PIL import Image

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS  (sabit — her iki model de aynı config kullanıyor)
# ══════════════════════════════════════════════════════════════════════════════
REPO_URL      = "https://github.com/woongzip1/UniverSR.git"
REPO_DIR      = Path("UniverSR")
TARGET_SR     = 48_000
N_FFT         = 1024
HOP           = 512
SR_TO_LR_BINS = {8000: 80, 12000: 128, 16000: 170, 24000: 256}
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

HF_REPOS = {
    "general": "woongzip1/universr-audio",
    "speech":  "woongzip1/universr-speech",
}

MODEL_LABELS = {
    "🎵 General  (music · SFX · mixed)": "general",
    "🗣️ Speech  (voice · recordings)":    "speech",
}

# ══════════════════════════════════════════════════════════════════════════════
#  BACKEND — Setup
# ══════════════════════════════════════════════════════════════════════════════
def log(msg): print(f"[UniverSR] {msg}", flush=True)

def ensure_repo():
    if not REPO_DIR.exists():
        log("Cloning UniverSR...")
        subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
    repo_str = str(REPO_DIR.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

# ══════════════════════════════════════════════════════════════════════════════
#  BACKEND — Model cache
# ══════════════════════════════════════════════════════════════════════════════
_cache: dict = {}

def load_model(model_type: str):
    if model_type in _cache:
        return _cache[model_type]

    ensure_repo()
    from universr import UniverSR

    repo_id = HF_REPOS[model_type]
    log(f"Loading UniverSR-{model_type.capitalize()} from {repo_id}...")
    model = UniverSR.from_pretrained(repo_id, device=DEVICE)
    model.eval()

    n = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"Ready — {n:.1f} M params on {DEVICE.upper()}")
    _cache[model_type] = model
    return model

# ══════════════════════════════════════════════════════════════════════════════
#  BACKEND — Audio helpers
# ══════════════════════════════════════════════════════════════════════════════
def simulate_lr(wav_np: np.ndarray, orig_sr: int, lr_sr_hz: int):
    """Return (wav_48k_lr_np, wav_lr_np) — both float32."""
    wav = torch.from_numpy(wav_np).unsqueeze(0)
    wav_lr  = TAT.Resample(orig_sr,  lr_sr_hz)(wav)
    wav_48k = TAT.Resample(lr_sr_hz, TARGET_SR)(wav_lr)
    return wav_48k.squeeze().numpy(), wav_lr.squeeze().numpy()

# ══════════════════════════════════════════════════════════════════════════════
#  BACKEND — Visualization
# ══════════════════════════════════════════════════════════════════════════════
_DARK = dict(
    figure_facecolor="#08090e", axes_facecolor="#0e1018",
    text_color="#c4d4ee", axes_labelcolor="#7a96be",
    xtick_color="#4a6280", ytick_color="#4a6280",
    axes_edgecolor="#1a2640", grid_color="#131c2e",
    grid_linestyle="--", font_family="monospace",
)
_RC = {k.replace("_", "."): v for k, v in _DARK.items()}


def _specshow(ax, sig, sr, cmap, title, title_color, lr_nyquist):
    D   = librosa.stft(sig.astype(np.float32), n_fft=N_FFT, hop_length=HOP)
    DB  = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(DB, sr=sr, hop_length=HOP,
                                   x_axis="time", y_axis="hz",
                                   ax=ax, cmap=cmap)
    ax.set_title(title, color=title_color, fontsize=10, pad=6)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Hz", fontsize=8)
    ax.axhline(y=lr_nyquist, color="#ffffff", lw=0.8, ls="--", alpha=0.5,
               label=f"LR Nyquist ({lr_nyquist//1000} kHz)")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.2, labelcolor="white")
    plt.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)


def make_figures(lr_np, hr_np, lr_sr_hz, model_type, ode_method, ode_steps, guidance):
    show = min(4.0, len(hr_np) / TARGET_SR)
    n    = int(show * TARGET_SR)
    lr_s = lr_np[:n].astype(np.float32)
    hr_s = hr_np[:n].astype(np.float32)
    t    = np.linspace(0, show, n)

    with plt.rc_context(_RC):
        # ── Waveform ────────────────────────────────────────────────────────
        fig_w, axes = plt.subplots(1, 2, figsize=(14, 2.8))
        fig_w.patch.set_facecolor("#08090e")
        fig_w.subplots_adjust(wspace=0.28, left=0.06, right=0.97, top=0.80, bottom=0.20)
        fig_w.suptitle(
            f"UniverSR-{model_type.capitalize()}  ·  "
            f"{lr_sr_hz//1000} kHz → 48 kHz  "
            f"({ode_method}, steps={ode_steps}, cfg={guidance})",
            fontsize=10, color="#4fc3f7", fontweight="bold", y=0.97
        )
        for ax, sig, label, color in [
            (axes[0], lr_s, f"LR Input  ({lr_sr_hz//1000} kHz)", "#ff5566"),
            (axes[1], hr_s, "SR Output (48 kHz)",                "#00e5aa"),
        ]:
            ax.plot(t, sig, color=color, lw=0.4, alpha=0.9)
            ax.set(xlim=[0, show], ylim=[-1, 1],
                   title=label, xlabel="Time (s)", ylabel="Amplitude")
            ax.title.set_color(color)
            ax.grid(True, alpha=0.2)

        # ── Spectrogram ─────────────────────────────────────────────────────
        fig_s = plt.figure(figsize=(14, 4.5))
        fig_s.patch.set_facecolor("#08090e")
        gs = gridspec.GridSpec(1, 2, figure=fig_s,
                               wspace=0.25, left=0.06, right=0.97,
                               top=0.84, bottom=0.14)
        _specshow(fig_s.add_subplot(gs[0, 0]), lr_s, TARGET_SR,
                  "inferno", f"Spectrogram — LR ({lr_sr_hz//1000} kHz)",
                  "#ff5566", lr_sr_hz // 2)
        _specshow(fig_s.add_subplot(gs[0, 1]), hr_s, TARGET_SR,
                  "viridis", "Spectrogram — SR (48 kHz)",
                  "#00e5aa", lr_sr_hz // 2)

    def to_pil(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf)

    return to_pil(fig_w), to_pil(fig_s)


# ══════════════════════════════════════════════════════════════════════════════
#  BACKEND — Main process
# ══════════════════════════════════════════════════════════════════════════════
def _pbar(step, total, label, detail=""):
    pct = int(step * 100 / total)
    det = f'<div style="color:#5a7090;font-size:0.78em;margin-bottom:14px">{detail}</div>' if detail else ""
    return (
        '<div style="font-family:monospace;padding:20px;background:#06080d;'
        'border:1px solid #1e2d45;border-radius:8px;margin-bottom:8px">'
        '<div style="color:#3a5270;font-size:0.72em;margin-bottom:10px;letter-spacing:2px">// PROCESSING</div>'
        f'<div style="color:#00e5aa;font-size:0.9em;font-weight:600;margin-bottom:6px">{label}</div>'
        f'{det}'
        '<div style="background:#10141f;border-radius:3px;height:6px;margin-bottom:8px;overflow:hidden">'
        f'<div style="background:linear-gradient(90deg,#00b888,#00e5aa);height:100%;width:{pct}%"></div>'
        '</div>'
        f'<div style="color:#2a5a78;font-size:0.74em">{step}/{total} &nbsp; {pct}%</div>'
        '</div>'
    )

_DONE_HTML = (
    '<div style="font-family:monospace;padding:14px 20px;background:#06080d;'
    'border:1px solid #00e5aa33;border-radius:8px;margin-bottom:8px">'
    '<span style="color:#00e5aa">OK &nbsp; Completed &nbsp; 100%</span></div>'
)
_ERR_HTML = (
    '<div style="font-family:monospace;padding:14px 20px;background:#06080d;'
    'border:1px solid #ff556633;border-radius:8px;margin-bottom:8px">'
    '<span style="color:#ff5566">ERROR</span></div>'
)


def process(audio_input, model_label, lr_sr_khz, ode_method, ode_steps, guidance, chunk_sec):
    """Generator with threaded blocking calls so UI updates between steps."""
    import traceback, threading, queue, torchaudio as _ta
    try:
        if audio_input is None:
            yield "", None, None, None, "Please upload an audio file."
            return

        model_type = MODEL_LABELS[model_label]
        lr_sr_hz   = int(lr_sr_khz) * 1000

        if lr_sr_hz not in SR_TO_LR_BINS:
            yield "", None, None, None, f"{lr_sr_khz} kHz not supported. Use 8 / 12 / 16 / 24."
            return

        orig_sr, wav_np = audio_input

        # Step 1
        yield _pbar(1, 4, "[1/4] Preparing audio..."), None, None, None, "Normalizing input..."
        if wav_np.ndim == 2:
            wav_np = wav_np.mean(axis=1)
        wav_np = wav_np.astype(np.float32)
        if np.abs(wav_np).max() > 1.0:
            wav_np /= 32768.0
        duration = len(wav_np) / orig_sr
        tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_in.name, wav_np, orig_sr)

        # Step 2 — load model in thread so UI updates
        if model_type not in _cache:
            msg2 = f"Downloading UniverSR-{model_type.capitalize()} from HuggingFace..."
            det2 = "~230 MB — please wait"
        else:
            msg2 = f"Loading UniverSR-{model_type.capitalize()} from cache..."
            det2 = ""
        yield _pbar(2, 4, "[2/4] Loading model...", det2), None, None, None, msg2

        q = queue.Queue()
        def _load():
            try:
                q.put(("ok", load_model(model_type), time.time()))
            except Exception as e:
                q.put(("err", e, None))
        t0 = time.time()
        th = threading.Thread(target=_load, daemon=True)
        th.start()
        # Poll every 0.5s while loading — keeps generator alive
        while th.is_alive():
            th.join(timeout=0.5)
            yield _pbar(2, 4, "[2/4] Loading model...", det2), None, None, None, msg2
        res = q.get()
        if res[0] == "err":
            raise res[1]
        model  = res[1]
        t_load = time.time() - t0

        # Step 3 — inference in thread
        inf_msg = f"ODE: {ode_method} / {ode_steps} steps / guidance={guidance}\n{lr_sr_hz//1000} kHz -> 48 kHz ({duration:.1f}s)"
        yield _pbar(3, 4, "[3/4] Enhancing audio...", f"{ode_method} / {ode_steps} steps"), None, None, None, "Running UniverSR...\n" + inf_msg

        # ── Smart chunked inference ─────────────────────────────────────────
        # 1. Check if model.enhance() natively supports chunk_sec
        import inspect as _inspect
        _enhance_params = _inspect.signature(model.enhance).parameters
        _native_chunk   = "chunk_sec" in _enhance_params

        SR_OUT = TARGET_SR
        C_LEN  = int(chunk_sec) * SR_OUT   # samples per chunk

        def _enhance_one(path_or_tensor, start_s=None, end_s=None):
            """Enhance a single segment, returns CPU tensor (1, T)."""
            import torchaudio as _ta2, tempfile as _tf2
            if isinstance(path_or_tensor, str):
                w, sr = _ta2.load(path_or_tensor)
                if w.shape[0] > 1:
                    w = w.mean(0, keepdim=True)
                if start_s is not None:
                    s = int(start_s * sr); e = int(end_s * sr)
                    w = w[:, s:e]
                seg_tmp = _tf2.NamedTemporaryFile(suffix=".wav", delete=False)
                _ta2.save(seg_tmp.name, w, sr)
                in_path = seg_tmp.name
            else:
                seg_tmp = _tf2.NamedTemporaryFile(suffix=".wav", delete=False)
                import torchaudio as _ta3
                _ta3.save(seg_tmp.name, path_or_tensor, lr_sr_hz)
                in_path = seg_tmp.name

            kw = dict(input_sr=lr_sr_hz, ode_method=ode_method,
                      ode_steps=int(ode_steps), guidance_scale=float(guidance))
            if _native_chunk:
                kw["chunk_sec"] = int(chunk_sec)
            with torch.no_grad():
                out = model.enhance(in_path, **kw)
            return out.cpu() if out.dim() == 2 else out.cpu().unsqueeze(0)

        # Load input audio to check total duration
        import torchaudio as _ta
        _wav_in, _sr_in = _ta.load(tmp_in.name)
        if _wav_in.shape[0] > 1:
            _wav_in = _wav_in.mean(0, keepdim=True)
        total_samples = _wav_in.shape[-1]
        total_dur     = total_samples / _sr_in
        n_chunks      = max(1, int(np.ceil(total_dur / chunk_sec)))

        q2 = queue.Queue()

        if n_chunks == 1 or _native_chunk:
            # Single pass
            def _infer():
                try:
                    t   = time.time()
                    q2.put(("progress", 1, 1))
                    out = _enhance_one(tmp_in.name)
                    q2.put(("ok", out, time.time() - t))
                except Exception as e:
                    q2.put(("err", e, 0))
        else:
            # Simple sequential chunking — no crossfade, just concatenate
            def _infer():
                try:
                    t      = time.time()
                    wav48  = TAT.Resample(_sr_in, SR_OUT)(_wav_in)  # (1, T48)
                    T48    = wav48.shape[-1]
                    parts  = []

                    for ci in range(n_chunks):
                        s48 = ci * C_LEN
                        e48 = min(s48 + C_LEN, T48)
                        seg = wav48[:, s48:e48]             # (1, seg_T) @ 48k

                        # Signal main thread which chunk we're on
                        q2.put(("progress", ci + 1, n_chunks))

                        seg_lr   = TAT.Resample(SR_OUT, lr_sr_hz)(seg)
                        enhanced = _enhance_one(seg_lr)     # (1, out_T) @ 48k
                        parts.append(enhanced)

                    result = torch.cat(parts, dim=-1).clamp(-1, 1)
                    q2.put(("ok", result, time.time() - t))
                except Exception as e:
                    import traceback; traceback.print_exc()
                    q2.put(("err", e, 0))

        th2 = threading.Thread(target=_infer, daemon=True)
        th2.start()

        cur_chunk, elapsed = 1, 0
        while th2.is_alive():
            th2.join(timeout=0.8)
            elapsed += 1
            # Drain progress messages without blocking
            while not q2.empty():
                msg = q2.get_nowait()
                if msg[0] == "progress":
                    cur_chunk = msg[1]
                elif msg[0] in ("ok", "err"):
                    q2.put(msg)   # put back for final read
                    break

            chunk_label = (f"chunk {cur_chunk}/{n_chunks}" if n_chunks > 1
                           else "single pass")
            yield (
                _pbar(3, 4,
                      f"[3/4] Enhancing... {elapsed}s",
                      f"{ode_method} / {ode_steps} steps  ·  {chunk_label}"),
                None, None, None,
                f"Processing {chunk_label}...\n" + inf_msg
            )

        # Final result
        while True:
            msg = q2.get()
            if msg[0] == "progress":
                continue
            break
        if msg[0] == "err":
            raise msg[1]
        wav_hr  = msg[1]
        t_infer = msg[2]

        # Step 4
        yield _pbar(4, 4, "[4/4] Building visualizations..."), None, None, None, "Generating spectrograms..."
        wav_hr = wav_hr.cpu()
        if wav_hr.dim() == 1:
            wav_hr = wav_hr.unsqueeze(0)
        hr_np = wav_hr.squeeze().numpy()
        lr_vis_np, _ = simulate_lr(wav_np, orig_sr, lr_sr_hz)
        lr_vis_np = lr_vis_np[:len(hr_np)]
        wave_img, spec_img = make_figures(
            lr_vis_np, hr_np, lr_sr_hz,
            model_type, ode_method, int(ode_steps), float(guidance)
        )
        tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        _ta.save(tmp_out.name, wav_hr, TARGET_SR)

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        info = (
            "Done\n" + "-"*34 + "\n"
            f"Model      : UniverSR-{model_type.capitalize()} ({n_params:.1f}M)\n"
            f"Device     : {DEVICE.upper()}\n" + "-"*34 + "\n"
            f"Input SR   : {orig_sr:,} Hz  ({duration:.2f}s)\n"
            f"LR sim     : {lr_sr_hz:,} Hz\n"
            f"Output SR  : {TARGET_SR:,} Hz\n"
            f"Upscale    : {TARGET_SR//lr_sr_hz}x\n" + "-"*34 + "\n"
            f"ODE        : {ode_method} / {ode_steps} steps\n"
            f"Guidance   : {guidance}\n"
            f"Chunk size : {chunk_sec}s\n" + "-"*34 + "\n"
            f"Load time  : {t_load:.1f}s\n"
            f"Inference  : {t_infer:.2f}s"
        )
        yield _DONE_HTML, tmp_out.name, wave_img, spec_img, info

    except Exception:
        tb = traceback.format_exc()
        print(tb)
        yield _ERR_HTML, None, None, None, "ERROR\n" + "-"*34 + "\n" + tb


# ══════════════════════════════════════════════════════════════════════════════
#  CSS — Terminal-green aesthetic  (sade görünüm, dolu hissiyat)
# ══════════════════════════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

:root {
    --bg0: #06080d;
    --bg1: #0b0e16;
    --bg2: #10141f;
    --bg3: #161c2a;
    --line: #1e2d45;
    --green: #00e5aa;
    --green2: #00b888;
    --blue: #4fc3f7;
    --muted: #3a5270;
    --text: #b8cce8;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'IBM Plex Sans', sans-serif;
}

/* ── Full-width override ── */
*, *::before, *::after { box-sizing: border-box; }

body { background: var(--bg0) !important; margin: 0; padding: 0; }

.gradio-container,
.gradio-container > .wrap,
.gradio-container > .contain,
.gradio-container > div {
    background: var(--bg0) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
    max-width: 100% !important;
    width: 100% !important;
    min-width: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Row = flex full width */
.gradio-container .gap,
.gradio-container > div > .gap {
    width: 100% !important;
    max-width: 100% !important;
}

/* Columns must not collapse */
.gradio-container .form,
.gradio-container .block,
.block { width: 100% !important; }

/* Header */
#usr-header {
    border-bottom: 1px solid var(--line);
    padding: 28px 24px 20px;
    margin-bottom: 16px;
    background: var(--bg0);
}
#usr-header h1 {
    font-family: var(--mono) !important;
    font-size: 1.8em !important;
    color: var(--green) !important;
    letter-spacing: 4px;
    margin: 0 !important;
}
#usr-header p {
    color: var(--muted) !important;
    font-size: 0.82em !important;
    font-family: var(--mono) !important;
    margin: 7px 0 0 !important;
    letter-spacing: 1px;
}
.tag-row { display:flex; gap:8px; margin-top:12px; flex-wrap:wrap; }
.tag {
    font-family: var(--mono);
    font-size: 0.71em;
    color: var(--green2);
    border: 1px solid #00e5aa33;
    border-radius: 3px;
    padding: 2px 10px;
    background: #00e5aa08;
}

/* Gradio blocks */
.block, .form, .gr-box,
label, .label-wrap {
    background: var(--bg2) !important;
    border: 1px solid var(--line) !important;
    border-radius: 6px !important;
}
label > span, .label-wrap span {
    font-family: var(--mono) !important;
    font-size: 0.72em !important;
    color: var(--muted) !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    background: transparent !important;
    border: none !important;
}
input, select, textarea {
    background: var(--bg3) !important;
    border: 1px solid var(--line) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    border-radius: 4px !important;
}

/* Slider */
input[type=range]::-webkit-slider-thumb { background: var(--green) !important; }
input[type=range]::-webkit-slider-runnable-track {
    background: linear-gradient(to right, var(--bg3), var(--green2)) !important;
}

/* Run button */
#run-btn {
    background: transparent !important;
    border: 1px solid var(--green) !important;
    border-radius: 4px !important;
    color: var(--green) !important;
    font-family: var(--mono) !important;
    font-size: 0.88em !important;
    font-weight: 600 !important;
    letter-spacing: 3px;
    text-transform: uppercase;
    height: 48px !important;
    transition: all 0.15s;
    box-shadow: 0 0 18px #00e5aa18;
}
#run-btn:hover {
    background: #00e5aa12 !important;
    box-shadow: 0 0 28px #00e5aa35;
}
/* Active: pressed */
#run-btn:active {
    background: #00e5aa35 !important;
    border-color: #00ffcc !important;
    color: #00ffee !important;
    box-shadow: 0 0 50px #00e5aa77, inset 0 0 20px #00e5aa22 !important;
    transform: scale(0.98) !important;
}
/* Processing: Gradio adds .generating class to the button */
#run-btn.generating {
    background: #00e5aa15 !important;
    border-color: #00e5aaaa !important;
    color: #00e5aa !important;
    cursor: wait !important;
    animation: btn-pulse 1.4s ease-in-out infinite;
}
@keyframes btn-pulse {
    0%, 100% { box-shadow: 0 0 14px #00e5aa22; border-color: #00e5aa44; }
    50%       { box-shadow: 0 0 40px #00e5aa77; border-color: #00e5aaee; }
}

/* Info box */
#info-box textarea {
    font-family: var(--mono) !important;
    font-size: 0.76em !important;
    color: #5a9070 !important;
    background: var(--bg0) !important;
    border: 1px solid var(--line) !important;
    line-height: 1.8 !important;
}

img { border-radius: 4px !important; }

/* Native Gradio component info text */
.gr-prose, span.text-sm.text-gray-500,
[class*="description"] {
    font-family: var(--mono) !important;
    font-size: 0.72em !important;
    color: #2a4a62 !important;
    margin-top: 3px !important;
}

#usr-footer {
    border-top: 1px solid var(--line);
    padding: 14px 24px;
    margin-top: 20px;
    text-align: center;
    font-family: var(--mono);
    font-size: 0.72em;
    color: var(--muted);
}
#usr-footer a { color: #2a5a78; text-decoration: none; }
#usr-footer a:hover { color: var(--blue); }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg1); }
::-webkit-scrollbar-thumb { background: var(--line); border-radius: 3px; }

/* Accordion */
.gr-accordion,
details, details > summary {
    background: var(--bg2) !important;
    border: 1px solid var(--line) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.82em !important;
}
details > summary {
    padding: 10px 14px !important;
    cursor: pointer !important;
    list-style: none !important;
    color: var(--green) !important;
    letter-spacing: 1px;
    border-bottom: none !important;
}
details[open] > summary {
    border-bottom: 1px solid var(--line) !important;
    border-radius: 6px 6px 0 0 !important;
}
details > summary::-webkit-details-marker { display: none; }
details > div, details > .gr-form {
    background: var(--bg2) !important;
    border: none !important;
    padding: 10px 6px 6px !important;
}
"""


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════
def build_ui():
    with gr.Blocks(css=CSS, title="UniverSR", theme=gr.themes.Base()) as demo:

        gr.HTML("""
        <div id="usr-header">
          <h1>◈ UNIVERSR</h1>
          <p>// vocoder-free audio super-resolution via flow matching</p>
          <div class="tag-row">
            <span class="tag">8 → 48 kHz</span>
            <span class="tag">Flow Matching</span>
            <span class="tag">iSTFT</span>
            <span class="tag">Speech · Music · SFX</span>
            <span class="tag">HuggingFace ✓</span>
          </div>
        </div>
        """)

        with gr.Tabs():

            # ══════════════════════════════════════════════════════════════
            #  TAB 1 — Enhance
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("⚡ Enhance"):
                with gr.Row(equal_height=False):

                    # ── Left: controls ────────────────────────────────────
                    with gr.Column(scale=3, min_width=280):

                        audio_in = gr.Audio(
                            label="Audio File",
                            type="numpy",
                            sources=["upload", "microphone"],
                        )

                        model_sel = gr.Dropdown(
                            choices=list(MODEL_LABELS.keys()),
                            value=list(MODEL_LABELS.keys())[0],
                            label="Model",
                            info="General: music / SFX / mixed audio  |  Speech: voice recordings",
                        )

                        lr_sr = gr.Dropdown(
                            choices=[8, 12, 16, 24],
                            value=8,
                            label="Input SR (kHz)",
                            info="Must match your file's actual sample rate — use the 🔍 File Info tab to check",
                        )

                        with gr.Accordion("⚙️  Advanced Settings", open=False):
                            ode_method = gr.Dropdown(
                                choices=["euler", "midpoint", "rk4"],
                                value="midpoint",
                                label="ODE Method",
                                info="euler (fastest) → midpoint (balanced) → rk4 (best quality)",
                            )
                            ode_steps = gr.Slider(
                                minimum=1, maximum=25, value=4, step=1,
                                label="ODE Steps",
                                info="More steps = better quality, slower — 4–10 is a good range",
                            )
                            guidance = gr.Slider(
                                minimum=1.0, maximum=5.0, value=1.5, step=0.5,
                                label="Guidance Scale",
                                info="Higher = stronger high-freq enhancement, may sound unnatural above 3.0",
                            )

                        chunk_sec = gr.Slider(
                            minimum=3, maximum=50, value=10, step=1,
                            label="Chunk Size (seconds)",
                            info="Audio is split into chunks to avoid OOM — lower if you run out of memory",
                        )

                        run_btn = gr.Button("⚡  ENHANCE", elem_id="run-btn")

                        info_box = gr.Textbox(
                            label="", lines=16,
                            interactive=False,
                            elem_id="info-box",
                            placeholder="// details appear here after processing",
                        )

                    # ── Right: output ─────────────────────────────────────
                    with gr.Column(scale=7):

                        progress_html = gr.HTML(value="")

                        audio_out = gr.Audio(
                            label="Super-Resolved (48 kHz)",
                            type="filepath",
                            interactive=False,
                        )
                        wave_img = gr.Image(label="", type="pil", show_label=False)
                        spec_img = gr.Image(label="", type="pil", show_label=False)

                run_btn.click(
                    fn      = process,
                    inputs  = [audio_in, model_sel, lr_sr, ode_method, ode_steps, guidance, chunk_sec],
                    outputs = [progress_html, audio_out, wave_img, spec_img, info_box],
                    show_progress = "hidden",
                )

            # ══════════════════════════════════════════════════════════════
            #  TAB 2 — File Info
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("🔍 File Info"):
                gr.HTML("""
                <div style="font-family:monospace;font-size:0.8em;color:#3a5270;
                            padding:12px 0 8px;letter-spacing:1px;">
                  // Upload a file to read its sample rate, duration, channels and bit depth.<br>
                  // Use the reported <b style="color:#00b888">Sample Rate</b> as your
                  <b style="color:#00b888">Input SR</b> in the Enhance tab.
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=4):
                        info_audio = gr.Audio(
                            label="Drop your audio file here",
                            type="filepath",
                            sources=["upload"],
                        )
                        info_btn = gr.Button("🔍  Inspect", elem_id="run-btn")

                    with gr.Column(scale=6):
                        info_result = gr.HTML(value="")

                def inspect_file(path):
                    if path is None:
                        return "<span style='color:#3a5270;font-family:monospace'>// no file uploaded</span>"
                    try:
                        import torchaudio as _ta
                        wav, sr = _ta.load(path)
                        ch      = wav.shape[0]
                        dur     = wav.shape[-1] / sr
                        # soundfile for bit depth — not all formats supported, fall back gracefully
                        try:
                            import soundfile as _sf
                            bits = _sf.info(path).subtype
                        except Exception:
                            bits = "unknown" 

                        # Closest supported LR_SR
                        supported = [8000, 12000, 16000, 24000]
                        closest   = min(supported, key=lambda x: abs(x - sr))
                        note_color = "#00e5aa" if sr in supported else "#f0a500"
                        note = (
                            f"✓ Exact match — set <b>Input SR = {sr//1000}</b>"
                            if sr in supported else
                            f"No exact match — closest supported: <b>Input SR = {closest//1000}</b>"
                        )

                        return f"""
<div style="font-family:monospace;background:#06080d;border:1px solid #1e2d45;
            border-radius:8px;padding:20px 24px;line-height:2">
  <div style="color:#3a5270;font-size:0.72em;letter-spacing:2px;margin-bottom:12px">// FILE INFO</div>
  <table style="border-collapse:collapse;width:100%">
    <tr><td style="color:#3a5270;padding-right:24px">Sample Rate</td>
        <td style="color:#00e5aa;font-weight:600">{sr:,} Hz &nbsp;({sr/1000:.1f} kHz)</td></tr>
    <tr><td style="color:#3a5270">Duration</td>
        <td style="color:#b8cce8">{dur:.2f} s &nbsp;({dur/60:.1f} min)</td></tr>
    <tr><td style="color:#3a5270">Channels</td>
        <td style="color:#b8cce8">{"Mono" if ch == 1 else f"Stereo ({ch}ch)"}</td></tr>
    <tr><td style="color:#3a5270">Bit depth</td>
        <td style="color:#b8cce8">{bits}</td></tr>
    <tr><td style="color:#3a5270">Frames</td>
        <td style="color:#b8cce8">{wav.shape[-1]:,}</td></tr>
  </table>
  <div style="margin-top:16px;padding:10px 14px;background:#0e1018;
              border-left:3px solid {note_color};border-radius:4px;
              color:{note_color};font-size:0.85em">{note}</div>
</div>"""
                    except Exception as e:
                        return f"<span style='color:#ff5566;font-family:monospace'>ERROR: {e}</span>"

                info_btn.click(
                    fn      = inspect_file,
                    inputs  = [info_audio],
                    outputs = [info_result],
                )

        gr.HTML("""
        <div id="usr-footer">
          <a href="https://arxiv.org/abs/2510.00771">arXiv:2510.00771</a> &nbsp;·&nbsp;
          <a href="https://github.com/woongzip1/UniverSR">GitHub</a> &nbsp;·&nbsp;
          <a href="https://huggingface.co/woongzip1/universr-audio">HF General</a> &nbsp;·&nbsp;
          <a href="https://huggingface.co/woongzip1/universr-speech">HF Speech</a>
        </div>
        """)

    return demo


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = False,   # True → public Gradio link
        inbrowser   = True,
    )

"""Shared pieces: voices, the Kokoro teacher, teacher rows, crops, the spectral objective and checkpoints."""
import copy
import glob
import json
import os
import random

import torch
import torch.nn.functional as F
import torchaudio

TEACHER_REPO = "hexgrad/Kokoro-82M"
TEACHER_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
VI_TEACHER_REPO = "xkore2908/ngoc_huyen_new"
VOICES = {"af_heart": "en", "if_sara": "it", "im_nicola": "it", "ngoc_huyen": "vi"}
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SENTENCES = {
    "en": os.path.join(DATA, "wikitext_sents.json"),
    "it": os.path.join(DATA, "italian_sents.json"),
    "vi": os.path.join(DATA, "vietnamese_sents.json"),
}
DEFAULT_VI_METADATA = "/kaggle/input/datasets/xkorenguyenthinh/ngoc-huyen-mini/metadata.csv"
SR = 24000
HOP = 600  # waveform samples per decoder frame (40 frames per second; F0 and N run at twice that rate)


def prepare_transformers_for_kokoro():
    import sys

    local_kokoro = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Kokoro-Vietnamese", "kokoro"))
    if os.path.isdir(local_kokoro) and local_kokoro not in sys.path:
        sys.path.insert(0, local_kokoro)
    try:
        import transformers.utils.import_utils as import_utils

        for flag in ("_torchvision_available", "_librosa_available", "_cv2_available"):
            if hasattr(import_utils, flag):
                setattr(import_utils, flag, False)
        if hasattr(import_utils, "_torchvision_version"):
            import_utils._torchvision_version = "N/A"
    except Exception:
        pass


def read_metadata_sentences(metadata_path):
    """Read text sentences from a pipe-delimited metadata.csv (id_or_filename|text)."""
    texts = []
    with open(metadata_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue
            col0, text = parts[0], parts[1]
            if col0.lower() in {"filename", "path", "audio"} and text.lower() == "text":
                continue
            if text:
                texts.append(text)
    return texts


def device(name=None):
    name = name or ("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    return name


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- teacher
def _find_first_existing(candidates):
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def teacher_files(voice):
    from huggingface_hub import hf_hub_download

    env_cfg = os.environ.get("TITL_TEACHER_CONFIG")
    env_mdl = os.environ.get("TITL_TEACHER_MODEL")
    env_vp = os.environ.get("TITL_TEACHER_VOICEPACK")
    if env_mdl and env_vp and os.path.isfile(env_mdl) and os.path.isfile(env_vp):
        cfg = env_cfg if (env_cfg and os.path.isfile(env_cfg)) else hf_hub_download(TEACHER_REPO, "config.json", revision=TEACHER_REVISION)
        return cfg, env_mdl, env_vp

    if VOICES.get(voice) == "vi":
        search_dirs = [
            os.environ.get("TITL_TEACHER_DIR"),
            "/content/kokoro",
            "/kaggle/working/kokoro",
            r"D:\Contents\Kokoro\kokoro_ngoc_huyen",
        ]
        for d in search_dirs:
            if not d or not os.path.isdir(d):
                continue
            mdl = _find_first_existing([
                env_mdl,
                os.path.join(d, "kokoro_vi_finetuned_3.pth"),
                os.path.join(d, "kokoro_vi.pth"),
                *sorted(glob.glob(os.path.join(d, "*.pth"))),
            ])
            vp = _find_first_existing([
                env_vp,
                os.path.join(d, "vi_finetuned_voice_3.pt"),
                os.path.join(d, "kokoro_vi_finetuned_3.pt"),
                os.path.join(d, f"{voice}.pt"),
                *sorted(glob.glob(os.path.join(d, "*.pt"))),
            ])
            cfg = _find_first_existing([
                env_cfg,
                os.path.join(d, "config.json"),
            ])
            if mdl and vp:
                if not cfg:
                    cfg = hf_hub_download(TEACHER_REPO, "config.json", revision=TEACHER_REVISION)
                return cfg, mdl, vp

        vi_repo = os.environ.get("TITL_TEACHER_REPO", VI_TEACHER_REPO)
        token = os.environ.get("HF_TOKEN")
        cfg = hf_hub_download(TEACHER_REPO, "config.json", revision=TEACHER_REVISION)
        mdl = hf_hub_download(vi_repo, "kokoro_vi_finetuned_3.pth", token=token)
        try:
            vp = hf_hub_download(vi_repo, "vi_finetuned_voice_3.pt", token=token)
        except Exception:
            vp = hf_hub_download(vi_repo, "kokoro_vi_finetuned_3.pt", token=token)
        return cfg, mdl, vp

    get = lambda name: hf_hub_download(TEACHER_REPO, name, revision=TEACHER_REVISION)
    return get("config.json"), get("kokoro-v1_0.pth"), get(f"voices/{voice}.pt")


def load_teacher(voice, dev="cpu"):
    """Kokoro-82M (pinned revision or custom finetuned checkpoint) with weight norm kept, and the voice pack [510, 1, 256]."""
    prepare_transformers_for_kokoro()
    from kokoro import KModel

    config, weights, pack = teacher_files(voice)
    model = KModel(repo_id=TEACHER_REPO, config=config, model=weights).to(dev).eval()
    return model, torch.load(pack, map_location="cpu", weights_only=True)


def style(pack, ps):
    """Kokoro indexes the voice pack by phoneme count; the decoder takes the first 128 dimensions."""
    return pack[len(ps) - 1]


def cpu_source_stft(decoder):
    """Compute the harmonic source's STFT on the CPU. Its phase feature sits on the +-pi branch cut for strong
    harmonic bins, and its sign there differs between devices; on the CPU the result is canonical."""
    st = decoder.generator.stft
    orig = type(st).transform

    def transform(x):
        mag, ph = orig(st, x.cpu())
        return mag.to(x.device), ph.to(x.device)

    st.transform = transform
    return decoder


def gpu_source_stft(decoder):
    """The same STFT on the device, without the CPU round trip. DC and Nyquist bins of a real signal are real:
    their imaginary part is set to +0.0, so their phase is exactly 0 or +pi, as on the CPU."""
    st = decoder.generator.stft

    def transform(x):
        X = torch.stft(x, st.filter_length, st.hop_length, st.win_length, window=st.window.to(x.device), return_complex=True)
        re, im = X.real, X.imag.clone()
        im[:, 0], im[:, -1] = 0.0, 0.0
        return X.abs(), torch.atan2(im, re)

    st.transform = transform
    return decoder


class FixedSource(torch.nn.Module):
    """Stands in for the generator's source module: returns an excitation set from outside."""

    def __init__(self):
        super().__init__()
        self.src = None

    def forward(self, f0):
        return self.src, None, None


class Excitation:
    """The teacher's own source module (sine harmonics plus noise, mixed by a learned layer) on the device."""

    def __init__(self, teacher, dev):
        g = teacher.decoder.generator
        self.m_source = copy.deepcopy(g.m_source).to(dev)
        self.f0_upsamp = copy.deepcopy(g.f0_upsamp).to(dev)

    @torch.no_grad()
    def __call__(self, F0):
        """F0 [B, 2F] -> excitation [B, 600F, 1]."""
        return self.m_source(self.f0_upsamp(F0[:, None]).transpose(1, 2))[0]


# ---------------------------------------------------------------- teacher rows
def run_dir(voice):
    return os.path.join(os.environ.get("TITL_RUNS_DIR", "runs"), voice)


def pool_dir(voice):
    return os.environ.get("TITL_POOL_DIR") or os.path.join(run_dir(voice), "pool")


def save_wav(audio_tensor, path):
    """Save a 1D float [-1, 1] or int16 audio tensor as a 24 kHz 16-bit mono WAV file."""
    import wave

    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = audio_tensor.detach().cpu().flatten()
    if t.dtype == torch.int16:
        pcm = t.numpy().astype("<i2")
    else:
        pcm = (t.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).numpy().astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def load_rows(voice, train_rows=0, keep_audio=False):
    """Held-out rows and training rows of a voice's teacher corpus (gen_teacher.py). train_rows > 0 keeps only
    the first train_rows training rows. Training rows drop their audio unless keep_audio."""
    d = os.path.join(run_dir(voice), "teacher")
    held = torch.load(os.path.join(d, "heldout.pt"), weights_only=False)
    train = []
    for path in sorted(glob.glob(os.path.join(d, "shard_*.pt"))):
        rows = torch.load(path, weights_only=False)
        for r in rows:
            r.pop("d", None)
            if not keep_audio:
                r.pop("audio", None)
        train += rows
        if 0 < train_rows <= len(train):
            break
    if not train:
        raise FileNotFoundError(f"no teacher rows in {d}; run gen_teacher.py {voice} first")
    return held, train[:train_rows] if train_rows > 0 else train


def frames(r):
    """Decoder inputs of a whole sentence: phoneme features expanded to frames [512, F], F0 and N [2F]."""
    idx = torch.repeat_interleave(torch.arange(len(r["pred_dur"])), r["pred_dur"])
    return r["t_en"].float()[:, idx], r["F0"][: 2 * len(idx)], r["N"][: 2 * len(idx)]


def full(r):
    """frames(r) plus the teacher's waveform for the sentence."""
    asr, F0, N = frames(r)
    return asr, F0, N, r["audio"][: HOP * asr.shape[1]].float() / 32767


def crop(r, seg, start=None):
    """A crop of seg frames from a random (or the given) start; a shorter sentence is zero-padded at the end."""
    asr, F0, N = frames(r)
    Fr = asr.shape[1]
    if Fr <= seg:
        pad = seg - Fr
        return F.pad(asr, (0, pad)), F.pad(F0, (0, 2 * pad)), F.pad(N, (0, 2 * pad)), 0
    a = random.randint(0, Fr - seg) if start is None else start
    return asr[:, a : a + seg], F0[2 * a : 2 * (a + seg)], N[2 * a : 2 * (a + seg)], a


# ---------------------------------------------------------------- objective (Paradee's, unchanged)
_MEL = {}


def spec_losses(y, t):
    """Log-mel L1 (FFT 1024, hop 256, 80 bands) and the multi-resolution STFT term (FFT 512/1024/2048, hop FFT/4:
    spectral convergence plus log-magnitude L1)."""
    if y.device not in _MEL:
        _MEL[y.device] = torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, hop_length=256, n_mels=80).to(y.device)
    mel = _MEL[y.device]
    lm = F.l1_loss(torch.log(mel(y).clamp_min(1e-5)), torch.log(mel(t).clamp_min(1e-5)))
    ls = 0
    for n in (512, 1024, 2048):
        w = torch.hann_window(n, device=y.device)
        Y = torch.stft(y, n, n // 4, window=w, return_complex=True).abs()
        T = torch.stft(t, n, n // 4, window=w, return_complex=True).abs()
        ls = ls + (T - Y).norm() / T.norm().clamp_min(1e-5) + F.l1_loss(torch.log(Y.clamp_min(1e-5)), torch.log(T.clamp_min(1e-5)))
    return lm, ls / 3


# ---------------------------------------------------------------- checkpoints
def save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def log_json(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

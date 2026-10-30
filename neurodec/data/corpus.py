# SPDX-License-Identifier: GPL-3.0-or-later
"""Corpus manifests: one record per utterance (path, speaker, language, duration).

Supported layouts (`type` in the config):
  jvs        root/jvsNNN/{parallel100,nonpara30,...}/wav24kHz16bit/*.wav
  jsut       root/<subset>/wav/*.wav                     (single speaker)
  libritts   root/<subset>/<spk>/<chapter>/*.wav           (LibriTTS / LibriTTS-R)
  librispeech root/<subset>/<spk>/<chapter>/*.flac
  generic    root/**/<speaker>/*.{wav,flac}
  synth      probe signals from neurodec.data.synth (no files)
"""
import json
from dataclasses import asdict, dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from . import synth

SR = 8000


@dataclass
class Utterance:
    path: str
    speaker: str
    lang: str
    corpus: str
    seconds: float


def _speaker(p: Path, kind: str, root: Path) -> str:
    rel = p.relative_to(root).parts
    if kind == "jvs":
        return next(x for x in rel if x.startswith("jvs"))
    if kind == "jsut":
        return "jsut"
    if kind in ("libritts", "librispeech"):
        return rel[1] if len(rel) >= 3 else p.parent.parent.name
    return p.parent.name


def scan(entry: dict) -> list[Utterance]:
    kind, lang = entry["type"], entry["lang"]
    if kind == "synth":
        return synth_manifest(entry)
    root = Path(entry["root"]).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"corpus {entry['name']}: {root} not found (see scripts/download_corpora.sh)")
    pats = ["**/*.flac"] if kind == "librispeech" else ["**/*.wav", "**/*.flac"]
    files = sorted({f for pat in pats for f in root.glob(pat)})
    if kind == "jvs":
        files = [f for f in files if "wav24kHz16bit" in f.parts]
    out = []
    for f in files:
        info = sf.info(f)
        if info.duration < entry.get("min_seconds", 0.5):
            continue
        out.append(Utterance(str(f), f"{entry['name']}/{_speaker(f, kind, root)}", lang, entry["name"],
                             float(info.duration)))
    return out


def synth_manifest(entry: dict) -> list[Utterance]:
    """Pseudo-utterances `synth:<seed>`; each 'speaker' groups probes of one kind."""
    out = []
    for i in range(entry.get("count", 1000)):
        kind, x = synth.make_probe(np.random.default_rng(entry.get("seed", 0) * 1_000_003 + i))
        out.append(Utterance(f"synth:{entry.get('seed', 0)}:{i}", f"{entry['name']}/{kind}", entry["lang"], entry["name"], len(x) / SR))
    return out


def load_audio(utt: Utterance | dict) -> np.ndarray:
    path = utt.path if isinstance(utt, Utterance) else utt["path"]
    if path.startswith("synth:"):
        _, seed, i = path.split(":")
        return synth.make_probe(np.random.default_rng(int(seed) * 1_000_003 + int(i)))[1]
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def build_manifest(entries: list[dict], cache: Path | None = None) -> list[Utterance]:
    if cache and cache.exists():
        return [Utterance(**json.loads(l)) for l in cache.read_text().splitlines()]
    utts = [u for e in entries for u in scan(e)]
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("".join(json.dumps(asdict(u), ensure_ascii=False) + "\n" for u in utts))
    return utts


def speaker_genders(entries: list[dict]) -> dict[str, str]:
    """speaker id -> 'M' / 'F' from the corpora's own speaker tables (JVS gender_f0range.txt,
    LibriTTS SPEAKERS.txt; JSUT is one female speaker). Speakers without a table entry are absent."""
    g: dict[str, str] = {}
    for e in entries:
        root = Path(e.get("root", ""))
        kind, name = e.get("type"), e.get("name", e.get("type"))
        if kind == "jvs" and (root / "gender_f0range.txt").exists():
            for line in (root / "gender_f0range.txt").read_text(encoding="utf-8").splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 2:
                    g[f"{name}/{parts[0]}"] = parts[1]
        elif kind == "jsut":
            g[f"{name}/jsut"] = "F"
        elif kind in ("libritts", "librispeech"):
            for sp in list(root.glob("SPEAKERS.txt")) + list(root.glob("*/SPEAKERS.txt")):
                for line in sp.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = [x.strip() for x in line.split("|")]
                    if len(parts) >= 2 and parts[0].isdigit():
                        g[f"{name}/{parts[0]}"] = parts[1]
    return g

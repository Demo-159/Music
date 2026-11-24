"""Jingle Bells en cumbia — pista para cantar.

Genera:
  jingle_cumbia_guia.mp3     con la melodía en acordeón (para aprenderla)
  jingle_cumbia_karaoke.mp3  sin melodía en las partes cantadas (para cantar encima)
  letra.md / jingle_cumbia.lrc  letra con tiempos

Canción tradicional (J. Pierpont, 1857, dominio público). Audio 100 % sintetizado con numpy.
"""
import shutil
import subprocess
import wave

import numpy as np

SR = 44100
BPM = 108
SPB = 60 / BPM
START = 0.3

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi(n):
    return NAMES.index(n[:-1]) + 12 * (int(n[-1]) + 1)


def freq(n):
    return 440.0 * 2 ** ((midi(n) - 69) / 12)


def shift(n, s):
    m = midi(n) + s
    return f"{NAMES[m % 12]}{m // 12 - 1}"


G_SCALE = ["G", "A", "B", "C", "D", "E", "F#"]


def third_below(n):
    new = G_SCALE[(G_SCALE.index(n[:-1]) - 2) % 7]
    cands = [f"{new}{o}" for o in (int(n[-1]) - 1, int(n[-1])) if midi(f"{new}{o}") < midi(n)]
    return max(cands, key=midi)


def tvec(d):
    return np.arange(int(d * SR)) / SR


def T(beat):
    return START + beat * SPB


# ============================================================ instrumentos
def accordion(n, dur):
    f = freq(n)
    t = tvec(dur + 0.08)
    s = np.zeros_like(t)
    for det, amp in [(0, 1.0), (14, 0.8), (-1200, 0.35)]:  # lengüetas: central, "musette" desafinada, octava baja
        ff = f * 2 ** (det / 1200)
        ph = 2 * np.pi * ff * t + np.random.uniform(0, 6)
        for h in range(1, int(6000 / ff) + 1):
            s += amp * np.sin(np.pi * h * 0.32) / h * np.sin(h * ph)
    env = np.minimum(1, t / 0.012)
    off = t > dur
    env[off] *= np.exp(-(t[off] - dur) / 0.03)
    return s * env * 0.35


def keys(n, dur):
    """Teclado/órgano percusivo para el 'chucu' de la cumbia."""
    f = freq(n)
    t = tvec(dur + 0.05)
    s = sum(a * np.sin(2 * np.pi * f * h * t) for h, a in [(1, 1), (2, .6), (3, .35), (4, .25), (6, .1)])
    env = np.minimum(1, t / 0.003) * np.exp(-t * 9)
    off = t > dur
    env[off] *= np.exp(-(t[off] - dur) / 0.02)
    return s * env


def bass(n, dur):
    f = freq(n)
    t = tvec(dur + 0.05)
    s = np.sin(2 * np.pi * f * t) + 0.45 * np.sin(4 * np.pi * f * t) + 0.2 * np.sin(6 * np.pi * f * t)
    s = np.tanh(1.6 * s)
    env = np.minimum(1, t / 0.004) * np.exp(-t * 1.8)
    off = t > dur
    env[off] *= np.exp(-(t[off] - dur) / 0.02)
    return s * env


def bandnoise(n, lo, hi, seed=None):
    r = np.random.default_rng(seed)
    x = r.normal(0, 1, n)
    a = max(1, int(SR / hi))
    b = max(a + 1, int(SR / lo))
    return np.convolve(x, np.ones(a) / a, "same") - np.convolve(x, np.ones(b) / b, "same")


def kick():
    t = tvec(0.35)
    f = 48 + 90 * np.exp(-t * 35)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9) * np.minimum(1, t / 0.002)


def conga(kind):
    t = tvec(0.4)
    if kind == "open":       # tono abierto (tumbadora)
        f = 190 * (1 + 0.08 * np.exp(-t * 30))
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)
    if kind == "low":        # tumba grave
        f = 140 * (1 + 0.08 * np.exp(-t * 30))
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 8)
    if kind == "slap":       # golpe seco
        return (0.6 * np.sin(2 * np.pi * 330 * t) * np.exp(-t * 30)
                + 0.8 * bandnoise(len(t), 800, 5000) * np.exp(-t * 45))
    return 0.5 * np.sin(2 * np.pi * 210 * t) * np.exp(-t * 40)  # apagado


def guiro(long):
    d = 0.13 if long else 0.045
    t = tvec(d + 0.02)
    rate = 75 if long else 95
    ridges = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * rate * t))
    env = np.minimum(1, t / 0.005) * np.clip((d - t) / 0.015 + 1, 0, 1)
    return bandnoise(len(t), 2500, 9000) * ridges * env * (1.0 if long else 0.8)


def cowbell():
    t = tvec(0.3)
    s = sum(np.sign(np.sin(2 * np.pi * f * t)) * a for f, a in [(560, 1), (845, .7)])
    s = s - np.convolve(s, np.ones(8) / 8, "same")
    return s * np.exp(-t * 18) * 0.4


def crash():
    t = tvec(2.0)
    return bandnoise(len(t), 3000, 12000) * np.exp(-t * 2.2) * np.minimum(1, t / 0.003)


def _sleigh(seed):
    r = np.random.default_rng(seed)
    t = tvec(0.25)
    s = np.zeros_like(t)
    for _ in range(14):
        f0 = r.uniform(4500, 11000)
        dl = r.uniform(0, 0.025)
        tt = np.clip(t - dl, 0, None)
        s += np.sin(2 * np.pi * f0 * tt) * np.exp(-tt * r.uniform(25, 45)) * (t >= dl)
    return s / 6 + 0.5 * bandnoise(len(t), 3000, 12000, seed) * np.exp(-t * 40)


SLEIGH = [_sleigh(i) for i in range(8)]
KICK, CRASH = kick(), crash()
CONGA = {k: conga(k) for k in ("open", "low", "slap", "mute")}
GUIRO_L, GUIRO_S, COWBELL = guiro(True), guiro(False), cowbell()

# ============================================================ canción
CH = {  # acorde: (notas teclado, fundamental bajo, quinta bajo)
    "G": (["G4", "B4", "D5"], "G2", "D2"), "C": (["G4", "C5", "E5"], "C3", "G2"),
    "Am": (["A4", "C5", "E5"], "A2", "E2"), "D": (["F#4", "A4", "D5"], "D3", "A2"),
    "D7": (["F#4", "A4", "C5"], "D3", "A2"), "A7": (["G4", "A4", "C#5"], "A2", "E2"),
    "E7": (["G#4", "B4", "D5"], "E3", "B2"),
}

CHORUS = [("B4", 1), ("B4", 1), ("B4", 2), ("B4", 1), ("B4", 1), ("B4", 2),
          ("B4", 1), ("D5", 1), ("G4", 1.5), ("A4", 0.5), ("B4", 4),
          ("C5", 1), ("C5", 1), ("C5", 1.5), ("C5", 0.5), ("C5", 1), ("B4", 1), ("B4", 1), ("B4", 0.5), ("B4", 0.5),
          ("B4", 1), ("A4", 1), ("A4", 1), ("B4", 1), ("A4", 2), ("D5", 2),
          ("B4", 1), ("B4", 1), ("B4", 2), ("B4", 1), ("B4", 1), ("B4", 2),
          ("B4", 1), ("D5", 1), ("G4", 1.5), ("A4", 0.5), ("B4", 4),
          ("C5", 1), ("C5", 1), ("C5", 1.5), ("C5", 0.5), ("C5", 1), ("B4", 1), ("B4", 1), ("B4", 0.5), ("B4", 0.5),
          ("D5", 1), ("D5", 1), ("C5", 1), ("A4", 1), ("G4", 4)]
CHORUS_CH = ["G", "G", "G", "G", "C", "G", "A7", "D7", "G", "G", "G", "G", "C", "G", "D7", "G"]

VERSE = [("D4", 1), ("B4", 1), ("A4", 1), ("G4", 1), ("D4", 3), ("D4", .5), ("D4", .5),
         ("D4", 1), ("B4", 1), ("A4", 1), ("G4", 1), ("E4", 4),
         ("E4", 1), ("C5", 1), ("B4", 1), ("A4", 1), ("F#4", 4),
         ("D5", 1), ("D5", 1), ("C5", 1), ("A4", 1), ("B4", 4),
         ("D4", 1), ("B4", 1), ("A4", 1), ("G4", 1), ("D4", 4),
         ("D4", 1), ("B4", 1), ("A4", 1), ("G4", 1), ("E4", 3), ("E4", 1),
         ("E4", 1), ("C5", 1), ("B4", 1), ("A4", 1), ("D5", 1), ("D5", 1), ("D5", 1), ("D5", 1),
         ("E5", 1), ("D5", 1), ("C5", 1), ("A4", 1), ("G4", 2), ("D5", 2)]
VERSE_CH = ["G", "G", "G", "C", "Am", "D", "D7", "G", "G", "G", "G", "C", "Am", "D", "D7", "G"]

INTRO = CHORUS[26:]           # "jingle bells... jingle all the way / oh what fun... sleigh"
INTRO_CH = CHORUS_CH[8:]
INTER = [("G5", .5), ("D5", .5), ("B4", .5), ("D5", .5), ("G5", .5), ("B5", .5), ("A5", .5), ("G5", .5),
         ("E5", .5), ("C5", .5), ("G4", .5), ("C5", .5), ("E5", .5), ("G5", .5), ("F#5", .5), ("E5", .5),
         ("D5", .5), ("A4", .5), ("F#4", .5), ("A4", .5), ("D5", .5), ("F#5", .5), ("E5", .5), ("C5", .5),
         ("B4", 1), ("D5", 1), ("G5", 1), ("D5", .5), ("B4", .5)]
INTER_CH = ["G", "C", "D7", "G"]
OUTRO = [("B4", 1), ("D5", 1), ("G4", 1.5), ("A4", .5), ("B4", 4), ("B4", 1), ("D5", 1), ("G4", 1.5), ("A4", .5), ("B4", 2), ("D5", 2)]
OUTRO_CH = ["G", "G", "G", "D7"]

# secciones: (nombre, compases, acordes, melodía, ¿se canta?, energía)
PLAN = [("Intro", INTRO_CH, INTRO, False, 1),
        ("Estrofa 1", VERSE_CH, VERSE, True, 1),
        ("Coro", CHORUS_CH, CHORUS, True, 2),
        ("Puente (acordeón)", INTER_CH, INTER, False, 2),
        ("Estrofa 2", VERSE_CH, VERSE, True, 1),
        ("Coro", CHORUS_CH, CHORUS, True, 2),
        ("Coro final", CHORUS_CH, CHORUS, True, 3),
        ("Final", OUTRO_CH, OUTRO, False, 3)]

LYRICS = {
    "Estrofa 1": ["Dashing through the snow, in a one-horse open sleigh",
                  "O'er the fields we go, laughing all the way",
                  "Bells on bobtail ring, making spirits bright",
                  "What fun it is to ride and sing a sleighing song tonight, oh!"],
    "Estrofa 2": ["A day or two ago, I thought I'd take a ride",
                  "And soon Miss Fanny Bright was seated by my side",
                  "The horse was lean and lank, misfortune seemed his lot",
                  "He got into a drifted bank, and we, we got upset, oh!"],
    "Coro": ["Jingle bells, jingle bells, jingle all the way",
             "Oh, what fun it is to ride in a one-horse open sleigh, hey!",
             "Jingle bells, jingle bells, jingle all the way",
             "Oh, what fun it is to ride in a one-horse open sleigh!"],
}
LYRICS["Coro final"] = LYRICS["Coro"]


def render(guide):
    rng = np.random.default_rng(7)
    np.random.seed(11)
    total_beats = sum(4 * len(p[1]) for p in PLAN) + 8
    L = np.zeros(int((T(total_beats) + 3) * SR))
    R = np.zeros_like(L)

    def add(sig, t, amp=1.0, pan=0.0):
        i = int(t * SR)
        sig = sig[: len(L) - i]
        L[i:i + len(sig)] += sig * amp * np.cos((pan + 1) * np.pi / 4)
        R[i:i + len(sig)] += sig * amp * np.sin((pan + 1) * np.pi / 4)

    def groove(b, ch, energy, bar_in_sec):
        tones, root, fifth = CH[ch]
        # bajo cumbiero
        for off, n, d in [(0, root, 1.4), (2, fifth, 0.9), (3, root, 0.45), (3.5, fifth, 0.45)]:
            add(bass(n, d * SPB), T(b + off), 0.55)
        # bombo (tambora)
        for off in (0, 2):
            add(KICK, T(b + off), 0.55)
        # güiro: "chiii-ki-ki" en cada tiempo
        for beat in range(4):
            add(GUIRO_L, T(b + beat), 0.13 * rng.uniform(.85, 1), pan=0.45)
            add(GUIRO_S, T(b + beat + 0.5), 0.10 * rng.uniform(.85, 1), pan=0.45)
            add(GUIRO_S, T(b + beat + 0.75), 0.08 * rng.uniform(.85, 1), pan=0.45)
        # congas
        for off, k, a in [(0, "mute", .25), (1, "slap", .35), (1.5, "mute", .2), (2, "mute", .25),
                          (2.5, "mute", .2), (3, "open", .45), (3.5, "low", .45)]:
            add(CONGA[k], T(b + off) + rng.uniform(-.004, .004), a, pan=-0.35)
        # teclado: "chucu" a contratiempo
        for off in (0.5, 1.5, 2.5, 3.5):
            for n in tones:
                add(keys(n, 0.22 * SPB), T(b + off), 0.075, pan=0.25)
        # cascabeles navideños en corcheas
        for k in range(8):
            add(SLEIGH[rng.integers(8)], T(b + k * 0.5), (0.16 if k % 2 == 0 else 0.10) * (1 + 0.2 * (energy > 1)),
                pan=-0.55)
        if energy >= 2:
            for beat in range(4):
                add(COWBELL, T(b + beat), 0.10 if beat % 2 == 0 else 0.07, pan=0.6)
        if energy >= 3:  # más fiesta: congas extra
            add(CONGA["slap"], T(b + 3.75), 0.25, pan=-0.35)

    def melody(b, mel, lead_amp, thirds, octave=0):
        for n, d in mel:
            n2 = shift(n, octave)
            add(accordion(n2, d * SPB * 0.92), T(b), lead_amp, pan=0.1)
            if thirds:
                add(accordion(shift(third_below(n), octave), d * SPB * 0.92), T(b), lead_amp * 0.6, pan=0.15)
            b += d

    b = 0
    marks = []
    for name, chords, mel, sung, energy in PLAN:
        marks.append((name, T(b)))
        add(CRASH, T(b), 0.12 if energy < 3 else 0.18, pan=0.3)
        for i, ch in enumerate(chords):
            groove(b + 4 * i, ch, energy, i)
        if not sung:
            melody(b, mel, 0.32, thirds=True)
        elif guide:
            melody(b, mel, 0.22, thirds=(energy >= 2))
        if sung and name.startswith("Coro"):
            # "¡hey!": golpe de acordeón y crash al final de la primera mitad
            add(CRASH, T(b + 30), 0.12)
            for n in ("D5", "F#5", "A5"):
                add(accordion(n, 0.4 * SPB), T(b + 30), 0.14)
        b += 4 * len(chords)

    # remate final "¡cha-cha-chán!"
    for off, chs, a in [(0, ["G4", "B4", "D5", "G5"], .2), (0.75, ["G4", "B4", "D5", "G5"], .2)]:
        for n in chs:
            add(accordion(n, 0.3 * SPB), T(b + off), a)
            add(keys(n, 0.3 * SPB), T(b + off), 0.1)
    add(bass("G2", 0.3 * SPB), T(b), 0.6)
    add(bass("G2", 1.5 * SPB), T(b + 0.75), 0.6)
    add(KICK, T(b), 0.6)
    add(KICK, T(b + 0.75), 0.6)
    add(CRASH, T(b + 0.75), 0.25)
    for k in range(12):
        add(SLEIGH[k % 8], T(b + 0.75) + k * 0.035, 0.2 * (1 - k / 12), pan=-0.5)

    end = int((T(b + 0.75) + 2.6) * SR)
    L, R = L[:end], R[:end]

    # reverb ligera
    def reverb(x, seed):
        r = np.random.default_rng(seed)
        t = tvec(1.4)
        ir = r.normal(0, 1, len(t)) * np.exp(-t * 4.5)
        ir = np.convolve(ir, np.ones(6) / 6, "same")
        ir[: int(.012 * SR)] = 0
        ir /= np.sqrt(np.sum(ir ** 2))
        nfft = 1 << (len(x) + len(ir) - 1).bit_length()
        return np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)[: len(x)]

    oL, oR = L + 0.12 * reverb(L, 1), R + 0.12 * reverb(R, 2)
    pk = max(abs(oL).max(), abs(oR).max())
    oL, oR = (np.tanh(1.6 * c / pk) / np.tanh(1.6) for c in (oL, oR))
    return (np.stack([oL, oR], 1) * 0.9 * 32767).astype(np.int16), marks


def save(data, base):
    with wave.open(base + ".wav", "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", base + ".wav", "-b:a", "192k", base + ".mp3"],
                       check=True)
    print(base, f"{len(data) / SR:.1f} s")


def mmss(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


if __name__ == "__main__":
    guia, marks = render(guide=True)
    save(guia, "jingle_cumbia_guia")
    kar, _ = render(guide=False)
    save(kar, "jingle_cumbia_karaoke")

    md = ["# Jingle Bells — versión cumbia 🎄🪇", "",
          f"Tonalidad: **Sol mayor (G)** · Tempo: **{BPM} BPM** · Canción tradicional (dominio público)", "",
          "- `jingle_cumbia_guia.mp3` — con la melodía en acordeón para guiarte",
          "- `jingle_cumbia_karaoke.mp3` — sin melodía en las partes cantadas", ""]
    lrc = ["[ti:Jingle Bells (cumbia)]"]
    for name, t in marks:
        md.append(f"## {name} — `{mmss(t)}`")
        lines = LYRICS.get(name)
        if lines:
            span = 16 * 4 * SPB / len(lines)
            for i, line in enumerate(lines):
                lt = t + i * span
                md.append(f"- `{mmss(lt)}` {line}")
                lrc.append(f"[{int(lt // 60):02d}:{lt % 60:05.2f}]{line}")
        else:
            md.append("*(instrumental)*")
            lrc.append(f"[{int(t // 60):02d}:{t % 60:05.2f}]♪ {name} ♪")
        md.append("")
    open("letra.md", "w").write("\n".join(md))
    open("jingle_cumbia.lrc", "w").write("\n".join(lrc) + "\n")

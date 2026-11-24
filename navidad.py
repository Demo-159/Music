"""Popurrí navideño sintetizado: Intro + Jingle Bells + We Wish You a Merry Christmas.

Villancicos tradicionales (dominio público). Todo el audio se sintetiza con numpy.
Uso: python3 navidad.py  -> genera navidad.wav (y navidad.mp3 si hay ffmpeg)
"""
import subprocess
import shutil
import wave

import numpy as np

SR = 44100
rng = np.random.default_rng(25)

NOTE_IDX = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6,
            "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def midi(n):
    name, octv = n[:-1], int(n[-1])
    return NOTE_IDX[name] + 12 * (octv + 1)


def freq(n):
    m = midi(n) if isinstance(n, str) else n
    return 440.0 * 2 ** ((m - 69) / 12)


def shift(n, semis):
    m = midi(n) + semis
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[m % 12]}{m // 12 - 1}"


G_SCALE = ["G", "A", "B", "C", "D", "E", "F#"]


def third_below(n):
    """Tercera diatónica inferior en Sol mayor."""
    name, octv = n[:-1], int(n[-1])
    i = G_SCALE.index(name)
    j = i - 2
    new = G_SCALE[j % 7]
    cands = [f"{new}{o}" for o in (octv - 1, octv) if midi(f"{new}{o}") < midi(n)]
    return max(cands, key=midi)


# ---------------------------------------------------------------- mezcla
TOTAL = 140.0
L = np.zeros(int(SR * TOTAL))
R = np.zeros(int(SR * TOTAL))


def add(sig, t, amp=1.0, pan=0.0):
    i = int(t * SR)
    if i >= len(L):
        return
    sig = sig[: len(L) - i]
    gl = np.cos((pan + 1) * np.pi / 4)
    gr = np.sin((pan + 1) * np.pi / 4)
    L[i:i + len(sig)] += sig * amp * gl
    R[i:i + len(sig)] += sig * amp * gr


def tvec(dur):
    return np.arange(int(dur * SR)) / SR


# ---------------------------------------------------------------- instrumentos
def glock(n, amp=1.0):
    f = freq(n)
    t = tvec(2.2)
    s = np.zeros_like(t)
    for ratio, a, dec in [(1, 1, 2.2), (2.76, 0.45, 5), (5.40, 0.22, 9), (8.93, 0.1, 14)]:
        if f * ratio < SR / 2:
            s += a * np.sin(2 * np.pi * f * ratio * t) * np.exp(-t * dec)
    s *= np.minimum(1, t / 0.002)
    return s * amp


def celesta(n, amp=1.0):
    f = freq(n)
    t = tvec(2.5)
    s = (np.sin(2 * np.pi * f * t) * np.exp(-t * 1.8)
         + 0.35 * np.sin(2 * np.pi * f * 2 * t) * np.exp(-t * 4)
         + 0.15 * np.sin(2 * np.pi * f * 4 * t) * np.exp(-t * 8))
    s *= np.minimum(1, t / 0.004)
    return s * amp


def tubular(n, amp=1.0):
    f = freq(n)
    t = tvec(6.0)
    s = np.zeros_like(t)
    for ratio, a, dec in [(0.5, 0.25, 0.7), (1, 1, 0.9), (2.0, 0.5, 1.3),
                          (3.0, 0.35, 1.8), (4.2, 0.25, 2.6), (5.4, 0.15, 3.5)]:
        s += a * np.sin(2 * np.pi * f * ratio * t + rng.uniform(0, 6)) * np.exp(-t * dec)
    s *= np.minimum(1, t / 0.003)
    return s * amp


def piano(n, dur, amp=1.0):
    f = freq(n)
    rel = 0.25
    t = tvec(dur + rel)
    s = np.zeros_like(t)
    for h in range(1, 9):
        if f * h > 9000:
            break
        det = 1 + 0.0004 * h * h
        s += (1 / h ** 1.4) * np.sin(2 * np.pi * f * h * det * t) * np.exp(-t * (1.2 + 0.6 * h))
    env = np.minimum(1, t / 0.003)
    off = t > dur
    env[off] *= np.exp(-(t[off] - dur) / 0.08)
    return s * env * amp


def bass(n, dur, amp=1.0):
    f = freq(n)
    t = tvec(dur + 0.1)
    s = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t)
    env = np.minimum(1, t / 0.005) * np.exp(-t * 2.5)
    off = t > dur
    env[off] *= np.exp(-(t[off] - dur) / 0.03)
    return s * env * amp


def strings(n, dur, amp=1.0, attack=0.25):
    f = freq(n)
    rel = 0.6
    t = tvec(dur + rel)
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.2 * t) * np.minimum(1, t / 0.6)
    s = np.zeros_like(t)
    for d in (-0.004, 0.0, 0.0045):
        ph = 2 * np.pi * f * (1 + d) * np.cumsum(vib) / SR
        for h in range(1, 14):
            if f * h > 7000:
                break
            s += np.sin(h * ph + rng.uniform(0, 6)) / h ** 1.3
    env = np.minimum(1, t / attack)
    off = t > dur
    env[off] *= np.exp(-(t[off] - dur) / 0.18)
    return s * env * amp / 3


def _sleigh_variant():
    t = tvec(0.25)
    s = np.zeros_like(t)
    for _ in range(14):
        f0 = rng.uniform(4500, 11000)
        delay = rng.uniform(0, 0.025)
        tt = np.clip(t - delay, 0, None)
        s += np.sin(2 * np.pi * f0 * tt) * np.exp(-tt * rng.uniform(25, 45)) * (t >= delay)
    noise = rng.normal(0, 1, len(t)) * np.exp(-t * 40)
    noise = noise - np.convolve(noise, np.ones(4) / 4, "same")  # paso alto simple
    return (s / 6 + 0.5 * noise) * 0.8


SLEIGH = [_sleigh_variant() for _ in range(8)]


def sleigh(t, amp=1.0):
    add(SLEIGH[rng.integers(len(SLEIGH))], t + rng.uniform(-0.004, 0.004),
        amp * rng.uniform(0.8, 1.05), pan=rng.uniform(0.2, 0.45))


def sleigh_shake(t, dur, amp=1.0):
    n = int(dur * 28)
    for k in range(n):
        env = np.sin(np.pi * k / n)
        sleigh(t + k / 28, amp * (0.3 + 0.7 * env))


# ---------------------------------------------------------------- armonía
CHORDS = {
    "G": (["G3", "B3", "D4"], "G2", "D2"),
    "C": (["G3", "C4", "E4"], "C3", "G2"),
    "Em": (["G3", "B3", "E4"], "E2", "B2"),
    "A7": (["G3", "A3", "C#4", "E4"], "A2", "E2"),
    "D": (["F#3", "A3", "D4"], "D3", "A2"),
    "D7": (["F#3", "A3", "C4", "D4"], "D3", "A2"),
    "B7": (["F#3", "A3", "B3", "D#4"], "B2", "F#2"),
}


class Sec:
    def __init__(self, start, bpm):
        self.start, self.spb = start, 60.0 / bpm

    def t(self, beat):
        return self.start + beat * self.spb


def play_melody(sec, mel, start_beat, voices):
    b = start_beat
    for n, d in mel:
        for fn in voices:
            fn(sec.t(b), n, d * sec.spb)
        b += d
    return b


# ---------------------------------------------------------------- 1) INTRO
intro = Sec(0.4, 76)
intro_prog = ["G", "Em", "C", "D7"]
for bar, ch in enumerate(intro_prog):
    tones, root, _ = CHORDS[ch]
    b0 = bar * 4
    for n in tones:
        add(strings(n, 4 * intro.spb, attack=0.8), intro.t(b0), 0.11, pan=-0.3)
    add(strings(shift(root, 12), 4 * intro.spb, attack=0.8), intro.t(b0), 0.12, pan=0.0)
    arp = [shift(x, 12) for x in tones] + [shift(x, 24) for x in tones]
    arp = (arp + arp[::-1][1:])[:8]
    for k, n in enumerate(arp):
        add(celesta(n), intro.t(b0 + k * 0.5), 0.28, pan=0.35 - 0.1 * (k % 3))
# campanas de iglesia
for b, n in [(0, "G4"), (4, "D4"), (8, "C4"), (12, "D4")]:
    add(tubular(n), intro.t(b), 0.33, pan=-0.2)
# los cascabeles se acercan...
for k in range(16):
    sleigh(intro.t(12 + k * 0.25), 0.04 + 0.03 * k)

# ---------------------------------------------------------------- 2) JINGLE BELLS
JB_A = [("B4", 1), ("B4", 1), ("B4", 2), ("B4", 1), ("B4", 1), ("B4", 2),
        ("B4", 1), ("D5", 1), ("G4", 1.5), ("A4", 0.5), ("B4", 4)]
JB_B1 = [("C5", 1), ("C5", 1), ("C5", 1.5), ("C5", 0.5), ("C5", 1), ("B4", 1), ("B4", 1), ("B4", 0.5), ("B4", 0.5),
         ("B4", 1), ("A4", 1), ("A4", 1), ("B4", 1), ("A4", 2), ("D5", 2)]
JB_B2 = [("C5", 1), ("C5", 1), ("C5", 1.5), ("C5", 0.5), ("C5", 1), ("B4", 1), ("B4", 1), ("B4", 0.5), ("B4", 0.5),
         ("D5", 1), ("D5", 1), ("C5", 1), ("A4", 1), ("G4", 4)]
JB = JB_A + JB_B1 + JB_A + JB_B2
JB_CH = ["G", "G", "G", "G", "C", "G", "A7", "D7", "G", "G", "G", "G", "C", "G", "D7", "G"]

jb = Sec(intro.t(16), 138)


def jingle_pass(b0, full):
    # melodía
    def v_glock(t, n, d):
        add(glock(shift(n, 12) if full else n), t, 0.30 if full else 0.34, pan=0.15)

    def v_piano(t, n, d):
        add(piano(n, d * 0.9), t, 0.22, pan=-0.05)

    def v_str(t, n, d):
        add(strings(n, d * 0.95, attack=0.04), t, 0.10, pan=-0.25)

    def v_harm(t, n, d):
        add(piano(third_below(n), d * 0.9), t, 0.12, pan=-0.15)
        add(celesta(shift(third_below(n), 12)), t, 0.10, pan=0.4)

    voices = [v_glock, v_piano] + ([v_str, v_harm] if full else [])
    play_melody(jb, JB, b0, voices)

    # acompañamiento oom-pah
    for bar, ch in enumerate(JB_CH):
        tones, root, fifth = CHORDS[ch]
        bb = b0 + bar * 4
        add(bass(root, jb.spb * 0.9), jb.t(bb), 0.45)
        add(bass(fifth, jb.spb * 0.9), jb.t(bb + 2), 0.42)
        for beat in (1, 3):
            for n in tones:
                add(piano(n, jb.spb * 0.45), jb.t(bb + beat), 0.10, pan=-0.35)
        if full:
            for n in tones:
                add(strings(shift(n, 12), 4 * jb.spb, attack=0.3), jb.t(bb), 0.045, pan=0.3)
            if bar % 4 == 0:
                add(tubular(shift(root, 24) if ch != "G" else "G4"), jb.t(bb), 0.16, pan=-0.4)
        # cascabeles en corcheas con acento
        for k in range(8):
            sleigh(jb.t(bb + k * 0.5), (0.42 if k % 2 == 0 else 0.26) * (1.15 if full else 1))


jingle_pass(0, full=False)
jingle_pass(64, full=True)

# puente: glissando de glockenspiel + sacudida de cascabeles
bridge_t = jb.t(128)
scale_up = ["G4", "A4", "B4", "C5", "D5", "E5", "F#5", "G5", "A5", "B5", "C6", "D6", "E6", "F#6", "G6"]
for k, n in enumerate(scale_up):
    add(glock(n), bridge_t + k * 0.055, 0.2, pan=-0.4 + 0.06 * k)
sleigh_shake(bridge_t, 1.3, 0.35)
add(tubular("D4"), bridge_t, 0.25)

# ---------------------------------------------------------------- 3) WE WISH YOU A MERRY CHRISTMAS
WW = [("G4", 1), ("G4", 0.5), ("A4", 0.5), ("G4", 0.5), ("F#4", 0.5), ("E4", 1), ("E4", 1), ("E4", 1),
      ("A4", 1), ("A4", 0.5), ("B4", 0.5), ("A4", 0.5), ("G4", 0.5), ("F#4", 1), ("D4", 1), ("D4", 1),
      ("B4", 1), ("B4", 0.5), ("C5", 0.5), ("B4", 0.5), ("A4", 0.5), ("G4", 1), ("E4", 1), ("D4", 0.5), ("D4", 0.5),
      ("E4", 1), ("A4", 1), ("F#4", 1)]
WW_CH = [[(0, "G")], [(0, "C")], [(0, "A7")], [(0, "D")], [(0, "B7")], [(0, "Em")], [(0, "C"), (1, "D7")], [(0, "G")]]

ww = Sec(bridge_t + 1.5, 150)


def wish_pass(b0, full, last=False):
    # anacrusa
    add(glock("D5" if full else "D4"), ww.t(b0 - 1), 0.3)
    add(piano("D4", ww.spb * 0.9), ww.t(b0 - 1), 0.2)

    def v_glock(t, n, d):
        add(glock(shift(n, 12)), t, 0.30, pan=0.15)

    def v_piano(t, n, d):
        add(piano(n, d * 0.9), t, 0.22)

    def v_str(t, n, d):
        add(strings(shift(n, 12) if full else n, d * 0.95, attack=0.04), t, 0.10, pan=-0.25)

    def v_harm(t, n, d):
        add(piano(third_below(n), d * 0.9), t, 0.11, pan=-0.15)

    end = play_melody(ww, WW, b0, [v_glock, v_piano, v_str] + ([v_harm] if full else []))
    final_len = 7.5 if last else 3
    # nota final (Sol)
    add(glock("G5"), ww.t(end), 0.32, pan=0.15)
    add(piano("G4", final_len * ww.spb), ww.t(end), 0.22)
    add(strings("G4" if not full else "G5", final_len * ww.spb), ww.t(end), 0.10, pan=-0.25)

    for bar, changes in enumerate(WW_CH):
        bb = b0 + bar * 3
        for idx, (off, ch) in enumerate(changes):
            tones, root, fifth = CHORDS[ch]
            dur_beats = (changes[idx + 1][0] if idx + 1 < len(changes) else 3) - off
            if bar == 7 and last:
                dur_beats = final_len
            add(bass(root, ww.spb * 0.95), ww.t(bb + off), 0.45)
            if bar == 7:
                for n in tones:
                    add(piano(n, ww.spb * dur_beats), ww.t(bb), 0.10, pan=-0.35)
            else:
                for beat in range(off + 1, off + dur_beats) if off == 0 else [off]:
                    for n in tones:
                        add(piano(n, ww.spb * 0.5), ww.t(bb + beat), 0.10, pan=-0.35)
            if full:
                for n in tones:
                    add(strings(shift(n, 12), dur_beats * ww.spb, attack=0.25), ww.t(bb + off), 0.045, pan=0.3)
        if bar < 7:
            for k in range(3):
                sleigh(ww.t(bb + k), 0.42 if k == 0 else 0.28)
    return b0 + 24


nb = wish_pass(1, full=False)
nb = wish_pass(nb + 1, full=True, last=True)

# ---------------------------------------------------------------- 4) FINAL
end_t = ww.t(nb - 3)
for k, n in enumerate(["G6", "D6", "B5", "G5", "D5", "B4", "G4"]):
    add(celesta(n), end_t + 0.6 + k * 0.09, 0.22, pan=0.4 - 0.1 * k)
for k, n in enumerate(scale_up):
    add(glock(n), end_t + 1.6 + k * 0.05, 0.16, pan=-0.4 + 0.06 * k)
for k, n in enumerate(["G4", "D4", "B3", "G3"]):
    add(tubular(n), end_t + 2.6 + k * 0.7, 0.28, pan=-0.3 + 0.2 * k)
sleigh_shake(end_t + 1.6, 3.0, 0.3)
add(tubular("G3"), end_t, 0.25)

# ---------------------------------------------------------------- reverb + master
def reverb(x, seconds=2.4, seed=0):
    r = np.random.default_rng(seed)
    t = tvec(seconds)
    ir = r.normal(0, 1, len(t)) * np.exp(-t * 3.0)
    ir = np.convolve(ir, np.ones(6) / 6, "same")  # oscurecer un poco
    ir[: int(0.015 * SR)] = 0
    ir /= np.sqrt(np.sum(ir ** 2))
    n = len(x) + len(ir)
    nfft = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)[: len(x)]
    return y


last_sample = int((end_t + 9) * SR)
L, R = L[:last_sample], R[:last_sample]
wetL, wetR = reverb(L, seed=1), reverb(R, seed=2)
outL = L + 0.22 * wetL
outR = R + 0.22 * wetR
fade = int(2.5 * SR)
for ch in (outL, outR):
    ch[-fade:] *= np.linspace(1, 0, fade) ** 2
peak = max(np.abs(outL).max(), np.abs(outR).max())
outL, outR = np.tanh(1.3 * outL / peak) / np.tanh(1.3), np.tanh(1.3 * outR / peak) / np.tanh(1.3)
stereo = (np.stack([outL, outR], axis=1) * 0.89 * 32767).astype(np.int16)

with wave.open("navidad.wav", "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(stereo.tobytes())
print(f"navidad.wav: {len(stereo) / SR:.1f} s")

if shutil.which("ffmpeg"):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "navidad.wav",
                    "-b:a", "192k", "navidad.mp3"], check=True)
    print("navidad.mp3 listo")

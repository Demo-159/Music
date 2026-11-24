"""Voz cantada con MBROLA: convierte (tiempo, duración, nota, sílaba) en audio cantado.

Requiere: apt-get install mbrola mbrola-us1 mbrola-us2
"""
import os
import subprocess
import tempfile
import wave

import numpy as np

# sílaba -> (consonantes iniciales, vocal sostenida, consonantes finales)  [fonemas SAMPA de MBROLA us1/us2]
SYL = {
    "jin": (["dZ"], "I", ["N"]), "gle": (["g"], "@", ["l"]), "bells": (["b"], "E", ["l", "z"]),
    "all": ([], "O", ["l"]), "the": (["D"], "@", []), "way": (["w"], "EI", []),
    "oh": ([], "@U", []), "what": (["w"], "V", ["t"]), "fun": (["f"], "V", ["n"]),
    "it": ([], "I", ["t"]), "is": ([], "I", ["z"]), "to": (["t_h"], "u", []),
    "ride": (["r"], "AI", ["d"]), "in": ([], "I", ["n"]), "a": ([], "@", []),
    "one": (["w"], "V", ["n"]), "horse": (["h"], "O", ["r", "s"]), "o": ([], "@U", []),
    "pen": (["p"], "@", ["n"]), "sleigh": (["s", "l"], "EI", []), "hey": (["h"], "EI", []),
    "we": (["w"], "i", []), "wish": (["w"], "I", ["S"]), "you": (["j"], "u", []),
    "mer": (["m"], "E", []), "ry": (["r"], "i", []), "christ": (["k", "r"], "I", ["s"]),
    "mas": (["m"], "@", ["s"]), "and": ([], "{", ["n", "d"]), "hap": (["h"], "{", []),
    "py": (["p"], "i", []), "new": (["n"], "u", []), "year": (["j"], "i", ["r"]),
}

# duración natural de cada consonante (s)
CONS = {"dZ": .07, "N": .08, "g": .06, "l": .06, "b": .05, "z": .06, "D": .05, "w": .05, "t": .07,
        "f": .08, "n": .07, "t_h": .07, "r": .05, "d": .05, "h": .06, "s": .09, "p": .06, "S": .1,
        "j": .05, "m": .06, "k": .08}

LEGATO = 0.96


def _plan(notes):
    """notes: lista de (t, dur, hz, syl) -> lista de segmentos (fonema, ini, fin, hz)."""
    segs = []
    for i, (t, d, hz, syl) in enumerate(notes):
        on, vow, co = SYL[syl]
        on_d = [CONS[c] for c in on]
        co_d = [CONS[c] for c in co]
        k_on = min(1.0, 0.3 * d / max(sum(on_d), 1e-9)) if on else 1
        k_co = min(1.0, 0.3 * d / max(sum(co_d), 1e-9)) if co else 1
        cur = t - sum(on_d) * k_on
        for c, cd in zip(on, on_d):
            segs.append([c, cur, cur + cd * k_on, hz])
            cur += cd * k_on
        end = t + d * LEGATO
        v_end = end - sum(co_d) * k_co
        segs.append([vow, t, v_end, hz])
        cur = v_end
        for c, cd in zip(co, co_d):
            segs.append([c, cur, cur + cd * k_co, hz])
            cur += cd * k_co
    # resolver solapes y huecos
    out = []
    for s in segs:
        if out and s[1] < out[-1][2]:
            out[-1][2] = s[1]
        if out and s[1] - out[-1][2] > 0.005:
            out.append(["_", out[-1][2], s[1], None])
        out.append(s)
    return [s for s in out if s[2] - s[1] > 0.004]


def _pho(segs, cents, seed):
    r = np.random.default_rng(seed)
    rate = r.uniform(5.0, 5.8)
    lines = []

    def silence(ms):  # MBROLA falla con silencios muy largos: partirlos
        while ms > 0:
            lines.append(f"_ {min(ms, 500)}")
            ms -= 500

    if segs[0][1] > 0:
        silence(round(segs[0][1] * 1000))
    for ph, a, b, hz in segs:
        ms = round(b * 1000) - round(a * 1000)
        if ms <= 0:
            continue
        if hz is None:
            silence(ms)
            continue
        hz = hz * 2 ** ((cents - 12) / 1200)  # MBROLA sale ~12 cents alto: compensar
        pts = []
        dur = b - a
        n = max(2, int(dur / 0.03))
        for k in range(n + 1):
            x = k / n
            tt = x * dur
            vib = 0.0
            if dur > 0.25:
                vib = 0.022 * np.sin(2 * np.pi * rate * tt) * min(1, max(0, (tt - 0.12) / 0.3))
            # pequeño "scoop": entra un poco por debajo de la nota
            scoop = -0.03 * max(0, 1 - tt / 0.06)
            pts.append(f"{round(x * 100)} {hz * (1 + vib + scoop):.1f}")
        lines.append(f"{ph} {ms} " + " ".join(pts))
    lines.append("_ 300")
    return "\n".join(lines) + "\n"


def sing(notes, voice="us1", cents=0.0, seed=0, sr_out=44100):
    """Devuelve audio mono (float) a sr_out, con t=0 al inicio de la canción."""
    notes = sorted(notes)
    segs = _plan(notes)
    with tempfile.TemporaryDirectory() as td:
        pho = os.path.join(td, "v.pho")
        wav = os.path.join(td, "v.wav")
        open(pho, "w").write(_pho(segs, cents, seed))
        r = subprocess.run(["mbrola", "-e", f"/usr/share/mbrola/{voice}/{voice}", pho, wav],
                           capture_output=True, text=True)
        if r.returncode:
            open("/tmp/claude-0/fail.pho", "w").write(open(pho).read())
            raise RuntimeError(f"mbrola ({voice}) falló: {r.stderr[-500:]}")
        with wave.open(wav) as w:
            sr = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float) / 32768
    tt = np.arange(int(len(x) * sr_out / sr)) / sr_out
    y = np.interp(tt, np.arange(len(x)) / sr, x)
    return y / (np.sqrt(np.mean(y[np.abs(y) > 1e-3] ** 2)) + 1e-9)

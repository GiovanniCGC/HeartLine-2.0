"""
Flipcraft main theme — "Rooftop Gold"
Bright nu-disco / feel-good house, A major, 118 BPM, 32 bars, seamless loop.

Everything is composed by hand below (HOOK / VERSE / COUNTER / BASS / drums)
and rendered with additive synthesis so the timbres stay clean and unaliased.
"""
import numpy as np

SR = 44100
BPM = 118.0
SPB = 60.0 / BPM          # seconds per beat
S16 = SPB / 4.0           # seconds per sixteenth
BARS = 32
BAR16 = 16                # sixteenths per bar
TOTAL16 = BARS * BAR16
SWING = 0.055             # fraction of a 16th that odd 16ths are pushed late
TAIL = 3.0                # extra render time, folded back for a seamless loop

LOOP_LEN = int(round(BARS * 4 * SPB * SR))
N = LOOP_LEN + int(TAIL * SR)

rng = np.random.default_rng(20240730)


def t16(p):
    """Sixteenth index -> seconds, with swing on the off-16ths."""
    swing = SWING * S16 if (p % 2) == 1 else 0.0
    return p * S16 + swing


def midi(n):
    return 440.0 * (2.0 ** ((n - 69) / 12.0))


# ----------------------------------------------------------------- buses
L = np.zeros(N, dtype=np.float64)
R = np.zeros(N, dtype=np.float64)
# dry buses that feed the reverb
sendL = np.zeros(N, dtype=np.float64)
sendR = np.zeros(N, dtype=np.float64)
# The kick sits on its own bus so the sidechain can duck everything *else*
# against it. Ducking the kick along with the mix is what makes a pump sound
# limp - the trigger has to stay untouched.
kickL = np.zeros(N, dtype=np.float64)
kickR = np.zeros(N, dtype=np.float64)


def place(buf, start, sig):
    e = min(len(buf), start + len(sig))
    if e <= start or start < 0:
        return
    buf[start:e] += sig[: e - start]


def stereo(sig, pan=0.0, send=0.0):
    """pan -1..1"""
    lg = np.cos((pan + 1) * np.pi / 4)
    rg = np.sin((pan + 1) * np.pi / 4)
    return lg, rg


def emit(sig, start, pan=0.0, send=0.0):
    lg, rg = stereo(sig, pan)
    place(L, start, sig * lg)
    place(R, start, sig * rg)
    if send > 0:
        place(sendL, start, sig * lg * send)
        place(sendR, start, sig * rg * send)


def emit_kick(sig, start, send=0.0):
    place(kickL, start, sig)
    place(kickR, start, sig)
    if send > 0:
        place(sendL, start, sig * send)
        place(sendR, start, sig * send)


# ----------------------------------------------------------------- voices
def pluck(freq, dur, amp=1.0, bright=1.0, harm=26, odd_bias=0.0):
    """Additive pluck: higher harmonics decay faster (physical, never harsh)."""
    n = max(8, int(dur * SR))
    t = np.arange(n) / SR
    out = np.zeros(n)
    nyq = SR * 0.45
    for k in range(1, harm + 1):
        f = freq * k
        if f > nyq:
            break
        a = 1.0 / (k ** (1.35 - 0.25 * bright))
        if odd_bias and k % 2 == 0:
            a *= 1.0 - odd_bias
        dec = (3.0 + 2.2 * k) / max(0.35, bright)
        out += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 0.4)) * np.exp(-t * dec)
    # soft attack so it never clicks
    atk = int(0.004 * SR)
    out[:atk] *= np.linspace(0, 1, atk)
    return out * amp


def bass_voice(freq, dur, amp=1.0):
    n = max(8, int(dur * SR))
    t = np.arange(n) / SR
    out = np.sin(2 * np.pi * freq * t) * 1.0
    out += 0.42 * np.sin(2 * np.pi * freq * 2 * t) * np.exp(-t * 5.5)
    out += 0.18 * np.sin(2 * np.pi * freq * 3 * t) * np.exp(-t * 9.0)
    out += 0.09 * np.sin(2 * np.pi * freq * 4 * t) * np.exp(-t * 14.0)
    env = np.minimum(1.0, np.arange(n) / (0.006 * SR))
    rel = np.exp(-np.maximum(0, t - dur * 0.55) * 9.0)
    return out * env * rel * amp


def pad_voice(freqs, dur, amp=1.0, detune=6.5):
    """Detuned saw stack, gently lowpassed by harmonic rolloff."""
    n = max(8, int(dur * SR))
    t = np.arange(n) / SR
    out = np.zeros(n)
    nyq = SR * 0.45
    for f0 in freqs:
        for d in (-detune, 0.0, detune):
            f = f0 * (2 ** (d / 1200.0))
            ph = rng.uniform(0, 2 * np.pi)
            for k in range(1, 15):
                fk = f * k
                if fk > nyq:
                    break
                out += (1.0 / (k ** 1.5)) * np.sin(2 * np.pi * fk * t + ph * k)
    atk = int(0.10 * SR)
    env = np.ones(n)
    env[:atk] = np.linspace(0, 1, atk) ** 1.6
    rel = int(0.30 * SR)
    if n > rel:
        env[-rel:] *= np.linspace(1, 0, rel) ** 1.4
    return out * env * (amp / (len(freqs) * 3.0))


def kick(amp=1.0):
    n = int(0.40 * SR)
    t = np.arange(n) / SR
    f = 48 + 105 * np.exp(-t * 42)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t * 6.2)
    click = rng.normal(0, 1, n) * np.exp(-t * 320) * 0.28
    return (body + click) * amp


def clap(amp=1.0):
    n = int(0.34 * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    # three quick bursts + tail = real clap, not a single noise hit
    for off, g in ((0.0, 1.0), (0.011, 0.85), (0.022, 0.7)):
        s = int(off * SR)
        seg = rng.normal(0, 1, n - s) * np.exp(-np.arange(n - s) / SR * 145) * g
        out[s:] += seg
    out += rng.normal(0, 1, n) * np.exp(-t * 24) * 0.30
    # band-shape it
    out = np.convolve(out, np.array([1, -0.72]), mode="same")
    return out * amp


def hat(dur=0.045, amp=1.0, tone=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = rng.normal(0, 1, n) * np.exp(-t * (150 / max(0.3, tone)))
    out = np.convolve(out, np.array([1, -0.86]), mode="same")  # highpass-ish
    return out * amp


def shaker(amp=1.0):
    n = int(0.05 * SR)
    t = np.arange(n) / SR
    env = np.exp(-t * 90) * (1 - np.exp(-t * 900))
    out = rng.normal(0, 1, n) * env
    out = np.convolve(out, np.array([1, -0.9]), mode="same")
    return out * amp


# ----------------------------------------------------------------- harmony
#      A major.  vi - IV - I - V  : F#m -> D -> A -> E
CHORDS = [
    ("F#m", 42, [54, 57, 61], [54, 57, 61, 64]),
    ("D",   38, [50, 54, 57], [50, 54, 57, 61]),
    ("A",   45, [57, 61, 64], [57, 61, 64, 68]),
    ("E",   40, [52, 56, 59], [52, 56, 59, 63]),
]


def chord_at(bar):
    return CHORDS[bar % 4]


# ----------------------------------------------------------------- melodies
# The hook.  A rising signature leap (F#->C#), a clear peak on E6, and an
# unresolved C#6 at the end so the ear is pulled straight back to the top.
HOOK = [
    (0, 78, 2), (2, 81, 2), (4, 85, 3), (8, 83, 2), (10, 81, 2), (12, 78, 4),
    (16, 76, 2), (18, 78, 2), (20, 81, 3), (24, 78, 2), (26, 76, 2), (28, 74, 4),
    (32, 76, 2), (34, 81, 2), (36, 85, 3), (40, 83, 2), (42, 85, 2), (44, 88, 4),
    (48, 86, 2), (50, 85, 2), (52, 83, 3), (56, 81, 2), (58, 83, 2), (60, 85, 4),
]

# Verse: same harmony, lower and more conversational, keeps the loop breathing.
VERSE = [
    (0, 73, 3), (4, 76, 3), (8, 78, 2), (11, 76, 3), (14, 73, 2),
    (16, 74, 3), (20, 78, 3), (24, 81, 2), (27, 78, 3), (30, 76, 2),
    (32, 76, 3), (36, 81, 3), (40, 85, 2), (43, 83, 3), (46, 81, 2),
    (48, 80, 3), (52, 83, 3), (56, 85, 2), (59, 83, 2), (62, 80, 2),
]

# Sparse high answer over the second hook.
COUNTER = [
    (6, 90, 2), (14, 88, 2), (22, 86, 2), (30, 85, 2),
    (38, 90, 2), (46, 92, 4), (54, 90, 2), (62, 88, 2),
]


def play_line(notes, bar0, amp, pan, bright, send, harmony=None, hum=0.0035):
    for (p, nt, dur) in notes:
        pos = bar0 * BAR16 + p
        if pos >= TOTAL16:
            continue
        st = int((t16(pos) + rng.normal(0, hum)) * SR)
        length = dur * S16 * 1.9
        v = amp * rng.uniform(0.90, 1.06)
        # a touch louder on downbeats, like a real player
        if p % 4 == 0:
            v *= 1.07
        emit(pluck(midi(nt), length, v, bright), st, pan, send)
        if harmony is not None:
            emit(pluck(midi(nt + harmony), length, v * 0.42, bright * 0.9),
                 st, -pan * 0.85, send * 0.9)


# ----------------------------------------------------------------- arrangement
# The hook opens the track, so the game starts on the catchiest bar AND the
# loop seam lands on a downbeat with a kick and a crash over it - the join is
# masked by a transient instead of falling in a quiet gap.
#  0-7   HOOK
#  8-15  verse
# 16-23  HOOK reprise (harmony + high answer)
# 24-27  breakdown (pad + arp, no kick) - lets the loop breathe
# 28-31  build back -> straight into the hook again
def section(bar):
    if bar < 8:
        return "hookA"
    if bar < 16:
        return "verse"
    if bar < 24:
        return "hookB"
    if bar < 28:
        return "break"
    return "build"


for bar in range(BARS):
    sec = section(bar)
    name, broot, triad, tetrad = chord_at(bar)
    base = bar * BAR16

    # ---- drums
    full = sec in ("hookA", "hookB", "verse", "build")
    for beat in range(4):
        p = base + beat * 4
        if sec == "break":
            hit = False
        elif sec == "build":
            hit = True if bar < 31 else beat in (0, 2)
        else:
            hit = True
        if hit:
            emit_kick(kick(0.92 if beat == 0 else 0.86), int(t16(p) * SR), 0.02)

    if full:
        for beat in (1, 3):
            p = base + beat * 4
            emit(clap(0.30), int(t16(p) * SR), 0.0, 0.30)

    # hats: offbeat 8ths, open hat lifting into each 4-bar phrase
    for p16 in range(2, 16, 4):
        p = base + p16
        g = 0.13 if sec == "break" else 0.22
        emit(hat(0.05, g, 1.0), int(t16(p) * SR), 0.12, 0.10)
    if bar % 4 == 3 and sec != "break":
        emit(hat(0.20, 0.17, 2.2), int(t16(base + 14) * SR), -0.15, 0.20)

    # shaker 16ths with an accent pattern -> this is what makes it groove
    for p16 in range(16):
        if sec == "break" and p16 % 2:
            continue
        g = 0.070 if p16 % 4 == 0 else (0.045 if p16 % 2 == 0 else 0.030)
        emit(shaker(g * rng.uniform(0.8, 1.15)),
             int(t16(base + p16) * SR), rng.uniform(-0.35, 0.35), 0.06)

    # ---- bass: root with an octave lift on the & of 3
    bamp = 0.26 if sec == "break" else 0.40
    for (p16, nt, dur) in ((0, broot, 3), (6, broot, 2), (10, broot, 2),
                           (12, broot + 12, 2), (14, broot, 2)):
        if sec == "break" and p16 in (12, 14):
            continue
        emit(bass_voice(midi(nt), dur * S16 * 1.6, bamp),
             int(t16(base + p16) * SR), 0.0, 0.0)

    # ---- pad
    pamp = {"hookA": 0.24, "verse": 0.20, "hookB": 0.27,
            "break": 0.32, "build": 0.30}[sec]
    freqs = [midi(x) for x in (tetrad if sec in ("hookB", "break", "build")
                               else triad)]
    emit(pad_voice(freqs, 4 * SPB * 1.02, pamp), int(t16(base) * SR), 0.0, 0.34)

    # ---- arp: 16th plucks through the chord, ping-ponged
    if sec in ("verse", "hookB", "break", "build"):
        seq = tetrad + tetrad[-2:0:-1]
        aamp = {"verse": 0.115, "hookB": 0.085,
                "break": 0.145, "build": 0.115}[sec]
        for p16 in range(16):
            nt = seq[p16 % len(seq)] + 12
            emit(pluck(midi(nt), 0.22, aamp * rng.uniform(0.85, 1.1), 1.15),
                 int(t16(base + p16) * SR),
                 0.55 if p16 % 2 else -0.55, 0.26)

    # ---- build: snare roll tightening into the loop point
    if sec == "build":
        div = {28: 4, 29: 4, 30: 2, 31: 1}[bar]
        for p16 in range(0, 16, div):
            g = 0.10 + 0.16 * ((bar - 28) / 3.0) * (0.5 + p16 / 32.0)
            emit(clap(g * 0.55), int(t16(base + p16) * SR),
                 rng.uniform(-0.3, 0.3), 0.34)

# ---- crash on the loop point: marks the seam with a transient
crash = rng.normal(0, 1, int(1.6 * SR)) * np.exp(-np.arange(int(1.6 * SR)) / SR * 2.6)
crash = np.convolve(crash, np.array([1, -0.80]), mode="same") * 0.16
emit(crash, 0, 0.0, 0.45)

# ---- riser through the build, resolving exactly on the loop point
rstart = int(t16(28 * BAR16) * SR)
# Stops exactly on the loop point. If it ran into the tail, the fold below
# would smear its loudest moment back over the opening bar.
rlen = LOOP_LEN - rstart
rt = np.arange(rlen) / SR
rdur = rlen / SR
sweep = rng.normal(0, 1, rlen)
sweep = np.convolve(sweep, np.ones(3) / 3, mode="same")
renv = (rt / rdur) ** 2.2 * 0.11
emit(sweep * renv, rstart, 0.0, 0.30)

# ---- lead lines
play_line(HOOK, 0, 0.30, 0.10, 1.30, 0.30)                       # hook A
play_line(HOOK, 4, 0.30, -0.10, 1.35, 0.30)                      # hook A repeat
play_line(VERSE, 8, 0.21, 0.14, 1.05, 0.26)                      # verse
play_line(VERSE, 12, 0.21, -0.14, 1.05, 0.26)
play_line(HOOK, 16, 0.31, 0.08, 1.40, 0.30, harmony=4)           # hook B + 3rds
play_line(HOOK, 20, 0.31, -0.08, 1.40, 0.30, harmony=4)
play_line(COUNTER, 16, 0.115, -0.45, 1.5, 0.42)                  # high answer
play_line(COUNTER, 20, 0.115, 0.45, 1.5, 0.42)
# Breakdown: the hook's 3rd/4th bars, alone and soft. Those bars are written
# over A and E, so they have to start on a bar where bar % 4 == 2 for the
# harmony to line up - hence 26, not 24.
play_line([(p - 32, n, d) for (p, n, d) in HOOK if p >= 32],
          26, 0.22, 0.0, 1.05, 0.44)
# Build: the hook's opening phrase (over F#m and D) announcing the return,
# so it starts on a bar where bar % 4 == 0.
play_line([(p, n, d) for (p, n, d) in HOOK if p < 32],
          28, 0.20, 0.0, 1.15, 0.36)


# ----------------------------------------------------------------- delay
def ping_pong(a, b, time_s, fb=0.34, mix=0.26):
    d = int(time_s * SR)
    outa, outb = a.copy(), b.copy()
    ta, tb = a * mix, b * mix
    for i in range(4):
        ta = np.concatenate([np.zeros(d), ta[:-d]]) * fb
        tb = np.concatenate([np.zeros(d), tb[:-d]]) * fb
        outa += tb
        outb += ta
        ta, tb = tb, ta
    return outa, outb


# 3/16 delay -> the classic house shuffle tail
L, R = ping_pong(L, R, S16 * 3, fb=0.30, mix=0.20)


# ----------------------------------------------------------------- reverb
def make_ir(dur=1.9, decay=5.2, pre=0.012):
    n = int(dur * SR)
    t = np.arange(n) / SR
    ir = rng.normal(0, 1, n) * np.exp(-t * decay)
    # smooth the very front so it reads as a room, not a burst
    p = int(pre * SR)
    ir[:p] *= np.linspace(0, 1, p) ** 2
    # tilt darker
    ir = np.convolve(ir, np.ones(6) / 6, mode="same")
    return ir / np.sqrt(np.sum(ir ** 2))


def fftconv(x, h):
    n = 1
    while n < len(x) + len(h):
        n <<= 1
    return np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(h, n))[: len(x)]


irL = make_ir()
irR = make_ir()
L += fftconv(sendL, irL) * 0.90
R += fftconv(sendR, irR) * 0.90


# ----------------------------------------------------------------- sidechain
# Duck everything against the kick - the pump that makes it read as produced
# music rather than a sequence of notes. The kick bus is deliberately not in
# L/R yet, so it stays at full weight and is added back after the ducking.
DUCK_DEPTH = 0.46
duck = np.ones(N)
for bar in range(BARS + 2):
    for beat in range(4):
        p = bar * BAR16 + beat * 4
        st = int(t16(p) * SR)
        if st >= N:
            break
        dur = int(0.30 * SR)
        e = min(N, st + dur)
        seg = np.arange(e - st) / SR
        shape = 1.0 - DUCK_DEPTH * np.exp(-seg * 13.0)
        duck[st:e] = np.minimum(duck[st:e], shape)
L *= duck
R *= duck
print("sidechain duck: floor %.3f, recovers to %.3f by the next beat"
      % (duck.min(), duck[int(t16(4) * SR) - 1]))

# kick back in, unducked
L += kickL
R += kickR


# ----------------------------------------------------------------- seamless
# Fold the reverb/delay tail that ran past the loop point back over the start,
# so bar 1 already contains the decay of bar 32 and the seam is inaudible.
def fold(x):
    head = x[:LOOP_LEN].copy()
    tail = x[LOOP_LEN:]
    head[: len(tail)] += tail
    return head


L, R = fold(L), fold(R)


# ----------------------------------------------------------------- master
def soft_clip(x, drive=1.25):
    return np.tanh(x * drive) / np.tanh(drive)


peak = max(np.max(np.abs(L)), np.max(np.abs(R)))
L, R = L / peak * 0.92, R / peak * 0.92
L, R = soft_clip(L, 1.20), soft_clip(R, 1.20)

# gentle high shelf for air
def shelf(x, g=0.16):
    return x + g * (x - np.convolve(x, np.ones(5) / 5, mode="same"))


L, R = shelf(L), shelf(R)

# Kill DC, then take both edges to zero over 2 ms. Far too short to hear as a
# dip, but it guarantees the wrap from the last sample to the first is a
# continuous step rather than a click.
L -= np.mean(L)
R -= np.mean(R)
F = int(0.002 * SR)
w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, F))
for ch in (L, R):
    ch[:F] *= w
    ch[-F:] *= w[::-1]

# 0.87, not 0.95: Opus rings on decode and overshoots the source peak by a few
# percent. Leaving headroom here is what keeps the decoded file from clipping.
peak = max(np.max(np.abs(L)), np.max(np.abs(R)))
L, R = L / peak * 0.87, R / peak * 0.87

rms = np.sqrt(np.mean(L ** 2))
print("loop length: %.3f s (%d bars @ %.0f BPM)" % (LOOP_LEN / SR, BARS, BPM))
print("rms %.4f  peak %.4f  dc %.2e" % (rms, np.max(np.abs(L)), np.mean(L)))
# Seam: the step from the last sample back to the first, which is what a
# looping player actually plays. Compare it to a typical sample-to-sample
# step inside the track - if it is in the same range there is nothing to hear.
for nm, ch in (("L", L), ("R", R)):
    jump = abs(ch[0] - ch[-1])
    typical = np.mean(np.abs(np.diff(ch)))
    p95 = np.percentile(np.abs(np.diff(ch)), 95)
    print("  %s seam step %.5f | median inner step %.5f | p95 %.5f -> %s"
          % (nm, jump, typical, p95, "inaudible" if jump <= p95 else "CHECK"))

stereo_i = np.empty(LOOP_LEN * 2, dtype=np.float32)
stereo_i[0::2] = L
stereo_i[1::2] = R
stereo_i.tofile("theme.f32")
print("wrote raw f32 stereo, %d frames" % LOOP_LEN)

# ---------------------------------------------------------------------------
# To rebuild the embedded soundtrack:
#   pip install numpy
#   python3 tools/compose-theme.py
#   ffmpeg -y -f f32le -ar 44100 -ac 2 -i theme.f32 \
#       -c:a libopus -b:a 112k -vbr constrained -compression_level 10 \
#       -application audio theme.opus
# then base64 theme.opus into the flipcraftBgmDataUri assignment in
# flipcraft.html. Keep the 0.87 master ceiling: Opus overshoots on decode and
# a higher ceiling clips on playback.
# ---------------------------------------------------------------------------

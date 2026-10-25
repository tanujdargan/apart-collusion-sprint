"""Figure 1: schematic of the Signal Table setup, probe, and ablation. Run: python make_fig1.py"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
DK, MID, LT = "#222222", "#777777", "#EDEDED"
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 7, "mathtext.default": "regular"})

fig = plt.figure(figsize=(7, 3))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 70); ax.set_ylim(0, 30); ax.axis("off")

def box(x, y, w, h, ec=DK, fc="white", lw=0.8, ls="-", r=0.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                ec=ec, fc=fc, lw=lw, ls=ls))

def arrow(p, q, c=DK, ls="-", lw=0.9, cs="arc3"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, color=c,
                                 lw=lw, ls=ls, connectionstyle=cs, shrinkA=0, shrinkB=0))

def T(x, y, s, **k):
    k.setdefault("ha", "center"); k.setdefault("va", "center"); k.setdefault("color", DK)
    ax.text(x, y, s, **k)

def stack(x, y, w, n=5, h=1.9, gap=0.8, colors=None):
    for i in range(n):
        yy = y + i * (h + gap)
        box(x, yy, w, h, ec=MID, fc=(colors[i] if colors else LT), lw=0.6, r=0.3)
    return y + n * (h + gap) - gap

# panel titles
for x, s in [(0.5, "(a) Signal Table"), (33.5, "(b) Probe"), (51.5, "(c) Cut the wire")]:
    T(x, 28.8, s, ha="left", fontweight="bold", fontsize=8)

# ---------- (a)
box(0.5, 15.5, 14, 11.5, ec=BLUE, lw=1)
T(7.5, 25.9, "Sender (counter)", fontweight="bold", color=BLUE)
T(7.5, 22.6, "knows hidden state\nS $\\in$ {COLD,\nNEUTRAL, HOT}", linespacing=1.3)
T(7.5, 17.9, "scheme (private):\nCOLD$\\to$weather  NEUTRAL$\\to$food\nHOT$\\to$sports", fontsize=7, color=MID, linespacing=1.3) if False else None
T(7.5, 18.3, "private scheme:\nCOLD$\\to$weather\nNEUTRAL$\\to$food\nHOT$\\to$sports", color=MID, linespacing=1.25)

box(23, 18, 9, 9, ec=ORANGE, lw=1)
T(27.5, 25.5, "Receiver", fontweight="bold", color=ORANGE)
T(27.5, 21.6, "(big player)\nknows scheme,\nreads public chat", linespacing=1.3)

box(0.5, 8.3, 22, 5.8, fc=LT, ec=MID, r=1.2)
T(11.5, 11.2, "\u201cCan\u2019t believe how hard it\u2019s\nraining outside tonight!\u201d", style="italic", linespacing=1.4)
T(22.4, 15.0, "public table talk", ha="right", color=MID)
arrow((7.5, 15.5), (7.5, 14.1))
ax.plot([22.5, 25.5], [11.2, 11.2], color=DK, lw=0.9)
arrow((25.5, 11.2), (25.5, 18))

box(23, 5.6, 9, 5.0)
T(27.5, 8.1, "DECODE: COLD\nBET: $0", fontweight="bold", linespacing=1.3)
arrow((30.0, 18), (30.0, 10.6))

box(0.5, 0.8, 22, 4.6, ec=MID, ls="--")
T(11.5, 3.1, "Pit boss: monitor reads\npublic chat only", color=MID, linespacing=1.3)
arrow((11.5, 8.3), (11.5, 5.4), c=MID, ls="--")

# ---------- (b)
Y0 = 10.5
T(41.5, 26.4, "Receiver residual stream", fontsize=7)
T(41.5, 24.9, "(over partner-message span)", color=MID)
stack(35, Y0, 13)
for i in range(5):
    T(35.8, Y0 + i * 2.7 + 0.95, f"L{i+1}", ha="left", color=MID)
box(35, 3.9, 13, 4.2, ec=ORANGE, lw=1)
T(41.5, 6.0, "linear probe\ndirection $d$", color=ORANGE, linespacing=1.2)
arrow((41.5, Y0), (41.5, 8.1))
arrow((41.5, 3.9), (41.5, 2.6))
T(41.5, 1.2, "collusion score  $h\\cdot d$", fontweight="bold")

# ---------- (c)
T(56.5, 26.4, "Same stream, $d$ removed", fontsize=7)
T(56.5, 24.9, "(receiver only)", color=MID)
stack(52, Y0, 9)
ax.add_patch(Rectangle((55.6, Y0 - 0.5), 1.8, 13.2, fc=ORANGE, ec="none", alpha=0.85))
T(56.5, 8.3, "$h \\leftarrow h-(h\\cdot d)\\,d$\nat every layer", color=ORANGE, linespacing=1.4)

box(62.2, 17.2, 7.6, 6.4)
T(66, 20.4, "Detector\nstill fires?", fontweight="bold", linespacing=1.3)
box(62.2, 9.5, 7.6, 6.4)
T(66, 12.7, "Collusion\nstill works?", fontweight="bold", linespacing=1.3)

T(51.5, 5.7, "Directions tested", ha="left", color=MID)
for i, (c, a, b) in enumerate([(BLUE, "$d_{detect}$", "detector"),
                               (GREEN, "$d_{state}$", "positive control"),
                               (MID, "random", "negative control")]):
    yy = 4.1 - i * 1.4
    ax.add_patch(Rectangle((51.5, yy - 0.5), 1.6, 1.0, fc=c, ec="none"))
    T(53.8, yy, a, ha="left")
    T(59.3, yy, b, ha="left", color=MID)

# panel separators
for x in (32.4, 50.6):
    ax.plot([x, x], [0.8, 29], color="#CCCCCC", lw=0.6)

here = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(here, "fig1_setup.png"), dpi=300)
fig.savefig(os.path.join(here, "fig1_setup.pdf"))

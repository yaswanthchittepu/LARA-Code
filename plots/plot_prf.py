"""Pareto frontier plot: E[c] (harmfulness) vs E[r] (helpfulness) across lambdas.

X-axis: E[c] — expected cost (higher = more harmful)
Y-axis: E[r] — expected reward (higher = more helpful)

A vertical red dashed line marks the harmfulness threshold (TAU).
The region to the right of TAU is shaded with increasing red intensity.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# ---- user config ----
import argparse

_p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_p.add_argument("--json_file", default="./results_lara/prf_data_llama3.1-8b.json",
                help="JSON written by scripts/make_prf_data.py")
_p.add_argument("--model", default="Llama", help="Top-level key in the JSON")
_p.add_argument("--size", default="8B", help="Label used in the title and output file name")
_a = _p.parse_args()
MODEL, SIZE, JSON_FILE = _a.model, _a.size, _a.json_file

# ---- load data ----
with open(JSON_FILE) as f:
    all_data = json.load(f)

cfg      = all_data[MODEL]
TAU      = cfg["tau"]
LAM_STAR = cfg["lam_star"]
lambdas  = np.array([p["lam"] for p in cfg["points"]])
er       = np.array([p["E_r"] for p in cfg["points"]])
ec       = np.array([p["E_c"] for p in cfg["points"]])

plt.rcParams.update({
    "font.size":        14,
    "axes.titlesize":   16,
    "axes.labelsize":   15,
    "xtick.labelsize":  13,
    "ytick.labelsize":  13,
    "legend.fontsize":  12,
    "axes.linewidth":   1.2,
})

# ---- plot ----
fig, ax = plt.subplots(figsize=(8, 5))

y_lo, y_hi = er.min() - 0.1, er.max() + 0.1
x_max = ec.max() + 0.15

# red danger zone gradient (right of TAU = more harmful), stronger
n_steps = 200
xs = np.linspace(TAU, x_max, n_steps)
for i in range(len(xs) - 1):
    alpha = 0.10 + 0.60 * ((xs[i] - TAU) / (x_max - TAU))
    ax.axvspan(xs[i], xs[i + 1], ymin=0, ymax=1,
               color="red", alpha=alpha, linewidth=0)

# threshold line
ax.axvline(TAU, color="red", linestyle="--", linewidth=2.0, label=f"threshold τ={TAU}")

# scatter — normal points
mask_star = lambdas == LAM_STAR
sc = ax.scatter(ec[~mask_star], er[~mask_star], c=lambdas[~mask_star],
                cmap="viridis", s=90, zorder=5)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("λ", fontsize=14)
cbar.ax.tick_params(labelsize=12)

# highlight lambda*
ax.scatter(ec[mask_star], er[mask_star],
           s=300, zorder=6, marker="*",
           color="gold", edgecolors="black", linewidths=1.0,
           label=f"λ*={LAM_STAR} (dual opt)")

for lam, r, c in zip(lambdas, er, ec):
    weight = "bold" if lam == LAM_STAR else "normal"
    size   = 10 if lam == LAM_STAR else 9
    ax.annotate(f"λ={lam}", (c, r),
                textcoords="offset points", xytext=(6, 5),
                fontsize=size, color="black", fontweight=weight)

ax.set_xlabel("E[c]  (harmfulness →)", fontsize=15)
ax.set_ylabel("E[r]  (helpfulness →)", fontsize=15)
ax.set_title(f"Helpfulness vs Harmfulness across λ — {MODEL}-{SIZE} (BoN)", fontsize=16)
ax.set_xlim(ec.min() - 0.1, x_max)
ax.set_ylim(y_lo, y_hi)
ax.tick_params(axis="both", which="major", length=5, width=1.2)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"pareto_plot_{MODEL}_{SIZE}.pdf", dpi=300, bbox_inches="tight")
plt.savefig(f"pareto_plot_{MODEL}_{SIZE}.png", dpi=300, bbox_inches="tight")
print(f"Saved pareto_plot_{MODEL}_{SIZE}.pdf / pareto_plot_{MODEL}_{SIZE}.png")
plt.show()

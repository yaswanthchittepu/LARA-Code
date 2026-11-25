"""GPT helpfulness vs harmlessness figure.

X-axis: harmlessness (higher = better), Y-axis: helpfulness (higher = better), both mean GPT ratings.
Reads the JSON written by scripts/collect_gpt_ratings.py:  <json_dir>/gpt_data_<model>.json

    python plots/plot_gpt_ratings.py --mode llama --json_dir results_lara
"""
import argparse
import json
import re

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--mode", default="llama", choices=["llama", "qwen", "both"])
ap.add_argument("--json_dir", default="./results_lara")
ap.add_argument("--out_name", default=None, help="output file stem (default: pareto_gpt_<mode>)")
args = ap.parse_args()

MODELS = {"llama": ("llama3.2-3b", "Llama-3.2-3B"), "qwen": ("qwen2.5-3b", "Qwen-2.5-3B")}
COLORS = {"BoN": "#1f77b4", "ARGS": "#ff7f0e", "PARGS": "#2ca02c", "CDQ": "#9467bd",
          "SFT": "#d62728", "SafeDPO": "#e377c2", "Safe-RLHF": "#8c564b"}
DISPLAY = {"BoN": "BoN-20", "ARGS": "ARGS", "PARGS": "PARGS", "CDQ": "CD-Q"}
MARKERS = {"SFT": "D", "SafeDPO": "s", "Safe-RLHF": "^"}


def plot_method(ax, name, entry):
    pts = sorted(entry["points"], key=lambda p: p["lam"])
    xs, ys = [p["harmlessness"] for p in pts], [p["helpfulness"] for p in pts]
    lam_star = entry.get("lam_star")
    ax.plot(xs, ys, color=COLORS[name], linewidth=0.8, alpha=0.5, zorder=3)
    if lam_star is not None and not any(abs(p["lam"] - lam_star) < 1e-6 for p in pts):
        print(f"WARNING: {DISPLAY[name]}: no point at lambda*={lam_star:g}; star not drawn", flush=True)
    for i, p in enumerate(pts):
        if lam_star is not None and abs(p["lam"] - lam_star) < 1e-6:
            ax.scatter(xs[i], ys[i], color=COLORS[name], s=220, marker="*", edgecolors="black", linewidths=0.8, zorder=6)
        elif i == 0:
            ax.scatter(xs[i], ys[i], color=COLORS[name], s=60, marker="o", zorder=8)
        else:
            ax.scatter(xs[i], ys[i], color="white", s=60, marker="o", zorder=8, edgecolors=COLORS[name], linewidths=1.5)


def draw(ax, key, title):
    with open(f"{args.json_dir}/gpt_data_{key}.json") as f:
        data = json.load(f)[key]
    for name in DISPLAY:
        if name in data["methods"]:
            plot_method(ax, name, data["methods"][name])
    for name, m in data["baselines"].items():
        if name in MARKERS:
            ax.scatter(m["harmlessness"], m["helpfulness"], color=COLORS[name], marker=MARKERS[name], s=110, zorder=7)
    ax.set_xlabel("Harmlessness →", fontsize=13)
    ax.set_ylabel("Helpfulness →", fontsize=13)
    ax.set_title(f"Helpfulness vs Harmlessness (GPT Evaluations) — {title}", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.25)


plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.1})
modes = ["llama", "qwen"] if args.mode == "both" else [args.mode]
fig, axes = plt.subplots(1, len(modes), figsize=(7 * len(modes), 5), squeeze=False)
for ax, m in zip(axes[0], modes):
    draw(ax, MODELS[m][0], MODELS[m][1])

handles = [mlines.Line2D([], [], color=COLORS[n], marker="*", linestyle="None", markersize=10, label=DISPLAY[n]) for n in DISPLAY]
handles += [mlines.Line2D([], [], color=COLORS[n], marker=MARKERS[n], linestyle="None", markersize=8, label=n) for n in MARKERS]
handles += [mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=7, label="λ=0"),
            mlines.Line2D([], [], color="black", marker="*", linestyle="None", markersize=12, label="λ* (dual opt)"),
            mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=7, markerfacecolor="none", label="λ_max")]
plt.tight_layout()
plt.subplots_adjust(bottom=0.28)
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10, bbox_to_anchor=(0.5, 0.0), frameon=True)
out = args.out_name or f"pareto_gpt_{args.mode}"
plt.savefig(f"{out}.pdf", dpi=300, bbox_inches="tight")
plt.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
print(f"Saved {out}.pdf / {out}.png")

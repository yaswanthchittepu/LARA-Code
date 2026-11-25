"""2x2 Pareto frontier figure.

Columns: BoN (left) | RGTG methods (right)
Rows:    Llama (top) | Qwen (bottom)

X-axis: E[c]  (harmfulness, lower is better → left)
Y-axis: E[r]  (helpfulness, higher is better → up)

All lambda points per method are plotted and connected by a line.
Lambda* point is highlighted with a star marker.
Baselines (SFT, SafeDPO, Safe-RLHF) are individual scatter points.
"""

import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# =============================================================================
# ---- config ----
# =============================================================================
import argparse
_p = argparse.ArgumentParser(description="2x2 Pareto frontier figure")
_p.add_argument("--eval_mode", default="gold", choices=["trained", "gold"], help="suffix of the input JSON name")
_p.add_argument("--mode", default="llama", choices=["llama", "qwen", "both"])
_p.add_argument("--out_name", default=None, help="output file stem (default: pareto_new_<mode>)")
_p.add_argument("--json_dir", default="./results_lara", help="where scripts/collect_results.py wrote the JSONs")
_a = _p.parse_args()
EVAL_MODE, MODE = _a.eval_mode, _a.mode

JSON_FILES = {
    "llama3.2-3b": f"{_a.json_dir}/pareto_data_llama3.2-3b_{EVAL_MODE}.json",
    "qwen2.5-3b":  f"{_a.json_dir}/pareto_data_qwen2.5-3b_{EVAL_MODE}.json",
}

# lam_star per method per model — requires judgment, kept as config
LAM_STAR = {
    "llama3.2-3b": {"BoN": 0.6,  "ARGS": 0.6,  "PARGS": 0.6,  "CDQ": 0.6},
    "qwen2.5-3b":  {"BoN": 2.0,  "ARGS": 2.0,  "PARGS": 2.0,  "CDQ": 2.0},
}

# =============================================================================
# ---- consistent colors/markers ----
# =============================================================================
COLORS = {
    "BoN":       "#1f77b4",   # blue
    "ARGS":      "#ff7f0e",   # orange
    "PARGS":     "#2ca02c",   # green
    "CDQ":       "#9467bd",   # purple
    "SFT":       "#d62728",   # red
    "SafeDPO":   "#e377c2",   # pink
    "Safe-RLHF": "#8c564b",   # brown
}
MARKERS = {
    "SFT":       "D",
    "SafeDPO":   "s",
    "Safe-RLHF": "^",
}

# =============================================================================
# ---- load data from JSON ----
# =============================================================================

def load_model_data(model_name):
    """Return (methods, baselines) for one model.

    methods:   {algo: [(lam, E_r, E_c), ...]}
    baselines: {name: (E_r, E_c)}
    """
    with open(JSON_FILES[model_name]) as f:
        raw = json.load(f)
    data = raw[model_name]

    methods = {
        algo: [(p["lam"], p["E_r"], p["E_c"]) for p in entry["points"]]
        for algo, entry in data.get("methods", {}).items()
    }
    # lambda* recorded by scripts/collect_results.py overrides the LAM_STAR constants above
    for algo, entry in data.get("methods", {}).items():
        if "lam_star" in entry:
            LAM_STAR[model_name][algo] = entry["lam_star"]

    baselines = {}
    bon_points = {}
    for name, entry in data.get("baselines", {}).items():
        m = re.match(r"BoN-\d+-lam-([0-9.]+)", name)
        if m:
            bon_points[float(m.group(1))] = (entry["E_r"], entry["E_c"])
        elif name in ("SFT", "SafeDPO", "Safe-RLHF"):
            baselines[name] = (entry["E_r"], entry["E_c"])

    if bon_points:
        methods["BoN"] = [(lam, er, ec) for lam, (er, ec) in sorted(bon_points.items())]

    return methods, baselines


if MODE in ("llama", "both"):
    METHODS_LLAMA, BASELINES_LLAMA = load_model_data("llama3.2-3b")
if MODE in ("qwen", "both"):
    METHODS_QWEN,  BASELINES_QWEN  = load_model_data("qwen2.5-3b")

# =============================================================================
# ---- helpers ----
# =============================================================================

def plot_method_line(ax, data, lam_star, color, label):
    """Plot all lambda points connected by line; star on lam*."""
    if not data:
        return
    lams = [d[0] for d in data]
    ers  = [d[1] for d in data]
    ecs  = [-d[2] for d in data]   # negate: -E[c] = harmlessness

    ax.plot(ecs, ers, color=color, linewidth=0.8, alpha=0.5, zorder=3)

    # lambda* is continuous (calibrated): it is only marked if the method was actually run at lambda*
    # (scripts/bon_select.py --calib_json adds it automatically; token-level decoders need a run with --lam <lambda*>).
    if not any(abs(l - lam_star) < 1e-6 for l in lams):
        print(f"WARNING: {label}: no point at lambda*={lam_star:g} (swept: {lams}); star not drawn", flush=True)
    for lam, er, ec in zip(lams, ers, ecs):
        if abs(lam - lam_star) < 1e-6:
            ax.scatter(ec, er, color=color, s=220, marker="*",
                       edgecolors="black", linewidths=0.8, zorder=6, label=label)
        elif lam == lams[0]:   # lambda=0: filled circle
            ax.scatter(ec, er, color=color, s=60, marker="o", zorder=8)
        else:                   # lambda_max: open circle
            ax.scatter(ec, er, color="white", s=60, marker="o", zorder=8,
                       edgecolors=color, linewidths=1.5)


def plot_baselines(ax, baselines):
    for name, (er, ec) in baselines.items():
        ax.scatter(-ec, er, color=COLORS[name], marker=MARKERS[name],
                   s=110, zorder=7, label=name)


def finalize(ax, title):
    ax.set_xlabel("-E[c]  (harmlessness →)", fontsize=13)
    ax.set_ylabel("E[r]  (helpfulness →)", fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="both", labelsize=11, length=4, width=1.1)
    ax.grid(True, alpha=0.25)


# =============================================================================
# ---- figure ----
# =============================================================================

plt.rcParams.update({
    "font.size":      12,
    "axes.linewidth": 1.1,
})

if MODE == "both":
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_llama, ax_qwen = axes
else:
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    if MODE == "llama":
        ax_llama = ax
    else:
        ax_qwen = ax

DISPLAY = {"BoN": "BoN-20", "ARGS": "ARGS", "PARGS": "PARGS", "CDQ": "CD-Q"}

if MODE in ("llama", "both"):
    for algo, label in DISPLAY.items():
        plot_method_line(ax_llama, METHODS_LLAMA.get(algo, []),
                         LAM_STAR["llama3.2-3b"][algo], COLORS[algo], label)
    plot_baselines(ax_llama, BASELINES_LLAMA)
    finalize(ax_llama, "Llama-3.2-3B")

if MODE in ("qwen", "both"):
    for algo, label in DISPLAY.items():
        plot_method_line(ax_qwen, METHODS_QWEN.get(algo, []),
                         LAM_STAR["qwen2.5-3b"][algo], COLORS[algo], label)
    plot_baselines(ax_qwen, BASELINES_QWEN)
    finalize(ax_qwen, "Qwen-2.5-3B")

# ---- shared legend ----
legend_handles = []
for name, color in [("BoN-20", COLORS["BoN"]), ("ARGS", COLORS["ARGS"]),
                    ("PARGS", COLORS["PARGS"]), ("CD-Q", COLORS["CDQ"])]:
    legend_handles.append(mlines.Line2D([], [], color=color, marker="*", linestyle="None",
                                        markersize=10, label=name))

for name in ["SFT", "SafeDPO", "Safe-RLHF"]:
    legend_handles.append(
        mlines.Line2D([], [], color=COLORS[name], marker=MARKERS[name],
                      linestyle="None", markersize=8, label=name)
    )

legend_handles += [
    mlines.Line2D([], [], color="black", marker="o", linestyle="None",
                  markersize=7, label="λ=0"),
    mlines.Line2D([], [], color="black", marker="*", linestyle="None",
                  markersize=12, label="λ* (dual opt)"),
    mlines.Line2D([], [], color="black", marker="o", linestyle="None",
                  markersize=7, markerfacecolor="none", label="λ_max"),
]

plt.suptitle("Helpfulness vs Harmlessness (Model Evaluations)", fontsize=15, y=1.01)
plt.tight_layout()
plt.subplots_adjust(bottom=0.28)
fig.legend(handles=legend_handles, loc="lower center", ncol=5,
           fontsize=10, bbox_to_anchor=(0.5, 0.0), frameon=True)
out_name = _a.out_name or f"pareto_new_{MODE}"
plt.savefig(f"{out_name}.pdf", dpi=300, bbox_inches="tight")
plt.savefig(f"{out_name}.png", dpi=300, bbox_inches="tight")
print(f"Saved {out_name}.pdf / {out_name}.png")
plt.show()

"""Results figures: the working resolution, the memory size, shared or separate memories, and defect maps."""
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.acquisition.visa import read, split
from src.evaluation.latency import MEMORY_SIZES, released
from src.evaluation.run import ground_truth, maps
from src.paths import CACHE, FIGURES, LATENCY_BUDGET_MS, PRODUCTS, REPORTS
from src.visualization import style

SHADES = [style.SHADES[k] for k in ("faint", "muted", "soft", "base", "strong")]


def load(name):
    return json.loads((REPORTS / f"{name}.json").read_text())


def bars(axis, values, labels, colors, fmt, top):
    axis.set_xlim(-0.5, len(values) - 0.5)
    axis.set_ylim(0, top)
    style.rounded_bars(axis, range(len(values)), values, colors)
    for i, v in enumerate(values):
        axis.annotate(fmt(v), (i, v), xytext=(0, 5), textcoords="offset points", ha="center",
                      va="bottom", fontsize=8, color=style.NEUTRAL["ink"])
    axis.set_xticks(range(len(values)), labels)


def budget(axis):
    axis.axhline(LATENCY_BUDGET_MS, color=style.GREY, linewidth=1, linestyle=(0, (2, 3)), zorder=1)
    axis.annotate("Budget", (axis.get_xlim()[1], LATENCY_BUDGET_MS), xytext=(-2, 4), textcoords="offset points",
                  ha="right", va="bottom", fontsize=7.5, color=style.NEUTRAL["secondary"])


def highlight(rows, chosen):
    """The released setting in the base shade, the others in a quieter one."""
    return [style.SHADES["base"] if r == chosen else style.SHADES["faint"] for r in rows]


def resolution():
    style.apply()
    rows = load("resolution")["settings"]
    chosen = released(load("latency"))["resolution"]
    labels = [f"{r['resolution']} px" for r in rows]
    colors = highlight([r["resolution"] for r in rows], chosen)
    small = [np.mean([p["caught_by_size"]["small"]["caught"] for p in r["products"].values()]) for r in rows]
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 2.9), gridspec_kw={"wspace": 0.22})
    bars(axes[0], [r["mean"]["aupro"] for r in rows], labels, colors, lambda v: f"{v:.2f}", 1.0)
    axes[0].set_title("Defect location (AUPRO)")
    bars(axes[1], small, labels, colors, lambda v: f"{v:.0%}", 1.0)
    axes[1].set_title("Small defects caught")
    axes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ms = [r["p95_ms"] for r in rows]
    bars(axes[2], ms, labels, colors, lambda v: f"{v:.0f} ms", max(max(ms), LATENCY_BUDGET_MS) * 1.2)
    budget(axes[2])
    axes[2].set_title("Time per part")
    for a in axes:
        a.set_xlabel("Long side")
    style.save(figure, FIGURES, "resolution")


def memory():
    style.apply()
    rows = load("memory")["settings"]
    chosen = released(load("latency"))["memory"]
    labels = [f"{r['memory']:,}" for r in rows]
    colors = highlight([r["memory"] for r in rows], chosen)
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 2.9), gridspec_kw={"wspace": 0.22})
    bars(axes[0], [r["mean"]["auroc"] for r in rows], labels, colors, lambda v: f"{v:.3f}", 1.0)
    axes[0].set_title("Part decision (AUROC)")
    bars(axes[1], [r["mean"]["escape_rate"] for r in rows], labels, colors, lambda v: f"{v:.0%}",
         max(r["mean"]["escape_rate"] for r in rows) * 1.3 + 0.01)
    axes[1].set_title("Defective parts let through")
    axes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ms = [r["p95_ms"] for r in rows]
    bars(axes[2], ms, labels, colors, lambda v: f"{v:.0f} ms", max(max(ms), LATENCY_BUDGET_MS) * 1.2)
    budget(axes[2])
    axes[2].set_title("Time per part")
    for a in axes:
        a.set_xlabel("Normal patches in memory")
    style.save(figure, FIGURES, "memory")


def shared():
    style.apply()
    rows = load("shared")["models"]
    labels = ["One per board", f"Shared\n{rows[1]['memory']:,}", f"Shared\n{rows[2]['memory']:,}"]
    colors = [style.SHADES["base"], style.SHADES["muted"], style.SHADES["faint"]]
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), gridspec_kw={"wspace": 0.22})
    bars(axes[0], [r["mean"]["auroc"] for r in rows], labels, colors, lambda v: f"{v:.3f}", 1.0)
    axes[0].set_title("Part decision (AUROC)")
    escape = [r["mean"]["escape_rate"] for r in rows]
    bars(axes[1], escape, labels, colors, lambda v: f"{v:.0%}", max(escape) * 1.3 + 0.01)
    axes[1].set_title("Defective parts let through")
    axes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ms = [r["p95_ms"] for r in rows]
    bars(axes[2], ms, labels, colors, lambda v: f"{v:.0f} ms", max(max(ms), LATENCY_BUDGET_MS) * 1.2)
    budget(axes[2])
    axes[2].set_title("Time per part")
    style.save(figure, FIGURES, "shared")


def examples():
    """One defective test board of each type, with its defect map and the annotated defect outlined."""
    style.apply()
    setting = released(load("latency"))
    r, chosen = setting["resolution"], MEMORY_SIZES.index(setting["memory"])
    thresholds = {p: v["from_normals"]["threshold"] for p, v in load("threshold")["products"].items()}
    ramp = LinearSegmentedColormap.from_list("", [style.SHADES[k] for k in ("faint", "soft", "base", "strong")])
    figure, axes = plt.subplots(1, len(PRODUCTS), figsize=(10.5, 2.4), gridspec_kw={"wspace": 0.06})
    for axis, product in zip(axes, PRODUCTS):
        cache = dict(np.load(CACHE / f"{product}-{r}.npz"))
        truth, test = ground_truth(product), split(product)[2]
        scores = cache["test"][:, chosen]
        defective = np.flatnonzero(truth["defective"])
        caught = [i for i in defective if scores[i] >= thresholds[product]]
        i = caught[len(caught) // 2] if caught else defective[0]  # a typical caught board, not the easiest
        defect_map = maps(cache["grids"][i:i + 1, chosen], truth, r)[0]
        image = cv2.resize(cv2.cvtColor(read(test[i].image), cv2.COLOR_BGR2GRAY), truth["size"])
        axis.imshow(image, cmap="gray", alpha=0.55)
        shown = np.ma.masked_less(defect_map, thresholds[product])
        axis.imshow(shown, cmap=ramp, alpha=0.85, vmin=thresholds[product], vmax=max(defect_map.max(), thresholds[product] + 1e-6))
        axis.contour(truth["masks"][i], levels=[0.5], colors=[style.NEUTRAL["ink"]], linewidths=0.8)
        axis.set_title(f"{product.upper()}  ·  {test[i].defect.replace(',', ', ')}", fontsize=8.5)
        axis.axis("off")
    style.save(figure, FIGURES, "examples")


if __name__ == "__main__":
    resolution()
    memory()
    shared()
    examples()

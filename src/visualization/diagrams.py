"""Documentation diagrams: the model as a file, and the model architecture.

The layout: a soft frame with a title, a symmetric row of raised
cards, each with an icon, a name and a short description, and small arrows
between them.
"""

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path as Shape
from PIL import Image

from src.paths import MODELS, ROOT
from src.visualization import style

DOCS = ROOT / "docs"
STRONG, BASE, SOFT, MUTED, FAINT = (style.SHADES[k] for k in ("strong", "base", "soft", "muted", "faint"))
FRAME, CARD, ARROW = style.NEUTRAL["background"], style.NEUTRAL["surface"], style.NEUTRAL["tertiary"]

CARD_W, CARD_H, GAP = 2.6, 3.4, 0.41  # portrait cards, like the UAV figure
MARGIN = 0.4  # around the row of cards, equal to CARD_Y
CARD_Y = 0.4
CARD_RADIUS, FRAME_RADIUS = 0.27, 0.38  # in proportion to the UAV figure's cards and frame
HEIGHT = CARD_Y + CARD_H + 1.0  # room above the cards for the title
ICON_Y = CARD_Y + 2.25
NAME_Y, SUBTITLE_Y = CARD_Y + 1.05, CARD_Y + 0.65
MIDDLE = CARD_Y + CARD_H / 2


# Drawing primitives ---------------------------------------------------------

INK = style.NEUTRAL["ink"]


def rounded(axis, x, y, w, h, color, radius=0.12, z=3, alpha=1.0):
    axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                  facecolor=color, edgecolor="none", alpha=alpha, zorder=z))


def outline(axis, x, y, w, h, color, radius=0.1, lw=1.6, z=3, dash=None):
    axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", facecolor="none",
                                  edgecolor=color, linewidth=lw, linestyle=dash or "-", zorder=z))


def board(axis, cx, cy, color, s=1.0, z=4):
    """A circuit board from above: the board, two chips and the traces between them."""
    rounded(axis, cx - 0.6 * s, cy - 0.44 * s, 1.2 * s, 0.88 * s, color, 0.1 * s, z=z)
    for x, y, w, h in ((-0.42, 0.02, 0.36, 0.28), (0.12, -0.3, 0.3, 0.24)):
        rounded(axis, cx + x * s, cy + y * s, w * s, h * s, CARD, 0.04 * s, z=z + 1)
    for path in (((-0.24, 0.02), (-0.24, -0.18), (0.12, -0.18)), ((-0.06, 0.16), (0.27, 0.16), (0.27, -0.06))):
        xs, ys = zip(*path)
        axis.plot([cx + x * s for x in xs], [cy + y * s for y in ys], color=STRONG, linewidth=1.4 * s,
                  solid_capstyle="round", solid_joinstyle="round", zorder=z + 1)


POINTS = np.random.default_rng(3).normal(0, 1, (26, 2)) * [0.34, 0.24]   # one fixed cloud of normal patches


# Icons, each centred on (cx, cy) ------------------------------------------

def tile(axis, x, y, size, color, alpha=1.0, z=4, edge=None):
    """A square patch, the unit the model reads and remembers."""
    axis.add_patch(FancyBboxPatch((x - size / 2, y - size / 2), size, size,
                                  boxstyle=f"round,pad=0,rounding_size={size * 0.24}", facecolor=color,
                                  edgecolor=edge or "none", linewidth=1.8 if edge else 0, alpha=alpha, zorder=z))


def icon_image(axis, cx, cy):
    """Board images arriving one after another, the newest in front."""
    for dx, dy, alpha in ((-0.3, -0.3, 0.3), (-0.16, -0.16, 0.55)):
        rounded(axis, cx - 0.5 + dx, cy - 0.34 + dy, 1.0, 0.7, BASE, 0.1, z=3, alpha=alpha)
    rounded(axis, cx - 0.5, cy - 0.34, 1.0, 0.7, BASE, 0.1, z=4)
    chip(axis, cx, cy, STRONG, 1.35)


def chip(axis, cx, cy, color, s=1.0, z=5):
    """A packaged component seen from above: its body, two rows of pins and the pin-one mark."""
    rounded(axis, cx - 0.22 * s, cy - 0.14 * s, 0.44 * s, 0.28 * s, color, 0.05 * s, z=z)
    for k in range(4):
        x = cx + (-0.15 + k * 0.1) * s
        for y in (cy + 0.15 * s, cy - 0.22 * s):
            rounded(axis, x - 0.025 * s, y, 0.05 * s, 0.07 * s, color, 0.015 * s, z=z)
    axis.add_patch(Circle((cx - 0.14 * s, cy + 0.06 * s), 0.03 * s, facecolor=BASE, edgecolor="none", zorder=z + 1))


def icon_features(axis, cx, cy):
    """The image read as a grid of patches, each described by its own features."""
    shades = [FAINT, MUTED, SOFT, BASE]
    rng = np.random.default_rng(1)
    for i in range(4):
        for j in range(4):
            rounded(axis, cx - 0.5 + j * 0.26, cy + 0.26 - i * 0.26, 0.22, 0.22, shades[rng.integers(4)], 0.05, z=4)
    outline(axis, cx - 0.5 + 2 * 0.26 - 0.03, cy + 0.26 - 1 * 0.26 - 0.03, 0.28, 0.28, STRONG, 0.07, lw=1.8, z=5)


def icon_memory(axis, cx, cy):
    """A store of normal patches: the few kept in front, the many they stand for fading behind."""
    cx -= 0.1
    for k, (dx, dy, alpha) in enumerate(((0.2, 0.2, 0.25), (0.1, 0.1, 0.5))):
        rounded(axis, cx - 0.5 + dx, cy - 0.42 + dy, 0.84, 0.72, FAINT, 0.1, z=3 + k, alpha=alpha)
    rounded(axis, cx - 0.5, cy - 0.42, 0.84, 0.72, MUTED, 0.1, z=5)
    kept = {(0, 0): STRONG, (0, 2): BASE, (1, 1): STRONG, (2, 0): BASE, (2, 2): BASE, (1, 0): SOFT, (0, 1): SOFT,
            (1, 2): SOFT, (2, 1): STRONG}
    for (i, j), color in kept.items():
        tile(axis, cx - 0.33 + j * 0.25, cy + 0.17 - i * 0.24, 0.18, color, z=6)


def icon_comparison(axis, cx, cy):
    """A new patch measured against the memory; the nearest normal patch decides its score."""
    q = (cx - 0.48, cy)
    targets = [(cx + 0.42, cy + 0.36), (cx + 0.42, cy), (cx + 0.42, cy - 0.36)]
    for k, (x, y) in enumerate(targets):
        near = k == 1
        axis.plot([q[0] + 0.14, x - 0.13], [q[1], y], color=BASE if near else FAINT,
                  linewidth=2.2 if near else 1.3, linestyle="-" if near else (0, (2, 2)), solid_capstyle="round", zorder=4)
        tile(axis, x, y, 0.24, BASE if near else MUTED, alpha=1.0 if near else 0.7, z=5)
    tile(axis, *q, 0.3, STRONG, z=6)


def icon_calibration(axis, cx, cy):
    """Part scores against the threshold set from good parts: the scores below it are held back."""
    base_y, line_y = cy - 0.42, cy + 0.0
    for i, (h, color) in enumerate(((0.62, BASE), (0.3, FAINT), (0.84, BASE), (0.6, BASE))):
        rounded(axis, cx - 0.48 + i * 0.27, base_y, 0.2, h, color, 0.08, z=4)
    axis.plot([cx - 0.62, cx + 0.62], [line_y, line_y], color=SOFT, linewidth=1.5, linestyle=(0, (1.5, 2.5)), zorder=3)


def icon_output(axis, cx, cy):
    """The board with the defect map laid over it and the defect framed: where it is, and that it is there."""
    board(axis, cx - 0.04, cy - 0.04, MUTED, 0.95)
    dx, dy = cx + 0.2, cy + 0.04
    for r, color, alpha in ((0.17, SOFT, 0.55), (0.11, BASE, 0.9), (0.055, STRONG, 1.0)):
        axis.add_patch(Circle((dx, dy), r, facecolor=color, edgecolor="none", alpha=alpha, zorder=7))
    half, reach, r = 0.24, 0.09, 0.05
    for sx in (-1, 1):
        for sy in (-1, 1):
            t = np.radians(np.linspace(0, 90, 16))
            corner = np.c_[half - r + r * np.cos(t), half - r + r * np.sin(t)]
            path = np.r_[[[half, half - reach]], corner, [[half - reach, half]]]
            axis.plot(dx + sx * path[:, 0], dy + sy * path[:, 1], color=STRONG, linewidth=2,
                      solid_capstyle="round", zorder=8)


def diagram(title: str, cards: list[tuple], name: str) -> None:
    style.apply()
    width = len(cards) * CARD_W + (len(cards) - 1) * GAP + 2 * MARGIN
    figure, axis = plt.subplots(figsize=(width / 1.45, HEIGHT / 1.45))
    axis.set_xlim(0, width)
    axis.set_ylim(0, HEIGHT)
    axis.set_aspect("equal")
    axis.axis("off")

    rounded(axis, 0, 0, width, HEIGHT, FRAME, FRAME_RADIUS, z=1)
    axis.text(width / 2, HEIGHT - 0.5, title, ha="center", va="center", fontsize=13,
              fontweight="medium", color=style.NEUTRAL["ink"], zorder=6)

    for index, (name_text, subtitle, draw) in enumerate(cards):
        x = MARGIN + index * (CARD_W + GAP)
        centre = x + CARD_W / 2
        rounded(axis, x, CARD_Y, CARD_W, CARD_H, CARD, CARD_RADIUS, z=2)
        draw(axis, centre, ICON_Y)
        axis.text(centre, NAME_Y, name_text, ha="center", va="center", fontsize=10,
                  fontweight="medium", color=style.NEUTRAL["ink"], zorder=6)
        axis.text(centre, SUBTITLE_Y, subtitle, ha="center", va="center", fontsize=7.4,
                  color=style.NEUTRAL["secondary"], zorder=6)
        if index:
            mid = x - GAP / 2
            axis.add_patch(Polygon([[mid - 0.07, MIDDLE - 0.12], [mid - 0.07, MIDDLE + 0.12], [mid + 0.09, MIDDLE]],
                                   facecolor=ARROW, edgecolor="none", zorder=3))

    style.save(figure, DOCS, name, panel=False)


def architecture() -> None:
    diagram("Model Architecture", [
        ("Input", "Board image", icon_image),
        ("Features", "ResNet-18 patches", icon_features),
        ("Memory", "Normal patches", icon_memory),
        ("Comparison", "Nearest normal patch", icon_comparison),
        ("Calibration", "Threshold from normals", icon_calibration),
        ("Output", "Defect located", icon_output),
    ], "architecture")


# The model as a file -------------------------------------------------------

def fill(axis, patch, colors, angle=90, z=3):
    """Fill a patch with a linear gradient; angle 90 runs bottom to top."""
    axis.add_patch(patch)
    patch.set_facecolor("none")
    patch.set_edgecolor("none")
    x0, y0, x1, y1 = patch.get_path().transformed(patch.get_patch_transform()).get_extents().bounds
    x1, y1 = x0 + x1, y0 + y1
    t = np.radians(angle)
    gx, gy = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
    ramp = gx * np.cos(t) + gy * np.sin(t)
    ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min())
    image = axis.imshow(ramp, extent=(x0, x1, y0, y1), origin="lower", zorder=z,
                        cmap=LinearSegmentedColormap.from_list("", colors), interpolation="bicubic")
    image.set_clip_path(patch)


def corner(points, centre, radius, start, end):
    angles = np.linspace(np.radians(start), np.radians(end), 12)
    points += [(centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)) for a in angles]


def page(x, y, w, h, fold, radius):
    """A document outline with rounded corners and a folded top-right corner."""
    points = []
    corner(points, (x + radius, y + radius), radius, 180, 270)
    corner(points, (x + w - radius, y + radius), radius, 270, 360)
    points += [(x + w, y + h - fold), (x + w - fold, y + h)]
    corner(points, (x + radius, y + h - radius), radius, 90, 180)
    return PathPatch(Shape(points + [points[0]], closed=True))


def squircle(x, y, size, radius):
    return FancyBboxPatch((x, y), size, size, boxstyle=f"round,pad=0,rounding_size={radius}")


def onnx_mark() -> np.ndarray:
    """The ONNX mark from docs/onnx.png, without its wordmark, recoloured in the accent.

    Lines take the brightest shade. Each facet keeps its original shading, mapped
    from the deepest shade for the darkest facet to the base shade for the lightest.
    """
    source = np.asarray(Image.open(DOCS / "onnx.png").convert("RGBA")).astype(float) / 255
    mark = source[:575]                                                        # above the wordmark
    rows, columns = np.where(mark[..., 3] > 0.05)
    mark = mark[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
    lightness, alpha = mark[..., :3].mean(axis=2), mark[..., 3]
    line = lightness < 0.5
    facets = LinearSegmentedColormap.from_list("", [MUTED, SOFT, BASE])
    shade = np.clip((lightness - 0.6) / 0.38, 0, 1)                           # facet greys span about 0.6 to 0.98
    image = np.empty_like(mark)
    image[..., :3] = np.where(line[..., None], matplotlib.colors.to_rgb(STRONG), facets(shade)[..., :3])
    image[..., 3] = np.where(line, alpha, alpha * 0.9)
    return image


def caption() -> str:
    card = json.loads((MODELS / "model.json").read_text())
    models = card["models"]
    files = models[0]["file"] if len(models) == 1 else f"{models[0]['file']} to {models[-1]['file']}"
    parameters = card["architecture"]["backbone_parameters"] / 1e6
    memory = card["architecture"]["memory"].split(" ")[0]
    each = " each" if len(models) > 1 else ""
    return f"{files}  \u00b7  {parameters:.1f} million parameters  \u00b7  {int(memory):,} normal patches  \u00b7  {models[0]['size_mb']} MB{each}"


def model_file() -> None:
    """The released model shown as what it is: a single ONNX file."""
    style.apply()
    width, height = 6 * CARD_W + 5 * GAP + 2 * MARGIN, 7.2
    figure, axis = plt.subplots(figsize=(width / 1.45, height / 1.45))
    axis.set_xlim(0, width)
    axis.set_ylim(0, height)
    axis.set_aspect("equal")
    axis.axis("off")
    rounded(axis, 0, 0, width, height, FRAME, FRAME_RADIUS, z=1)

    w, h, fold = 2.55, 3.4, 0.62                                              # document proportions, 3:4
    x, y = width / 2 - w / 2, 2.5
    cx = width / 2

    for spread, alpha in ((0.32, 0.04), (0.2, 0.06), (0.1, 0.09), (0.03, 0.14)):   # soft lift
        shadow = page(x - spread, y - spread - 0.12, w + 2 * spread, h + 2 * spread, fold + spread, 0.2 + spread)
        shadow.set(facecolor="black", edgecolor="none", alpha=alpha, zorder=2)
        axis.add_patch(shadow)

    sheet = page(x, y, w, h, fold, 0.2)
    fill(axis, sheet, ["#1D1D20", "#2A2A2E", "#38383D"], z=3)                # lit from above
    rim = page(x, y, w, h, fold, 0.2)
    rim.set(facecolor="none", edgecolor="#4A4A50", linewidth=1, zorder=4)
    axis.add_patch(rim)
    flap = PathPatch(Shape([(x + w - fold, y + h), (x + w - fold, y + h - fold + 0.1),
                            (x + w - fold + 0.1, y + h - fold), (x + w, y + h - fold),
                            (x + w - fold, y + h)], closed=True))
    fill(axis, flap, ["#3E3E44", "#5A5A61"], angle=45, z=4)

    mark = onnx_mark()
    size = 1.3                                                                 # about half the page width
    aspect = mark.shape[0] / mark.shape[1]
    label = 0.3                                                                # cap height of the label
    block = size * aspect + 0.36 + label                                       # mark, gap, label
    top = y + h * 0.47 + block / 2                                             # centred, just below the middle
    axis.imshow(mark, extent=(cx - size / 2, cx + size / 2, top - size * aspect, top),
                zorder=6, interpolation="lanczos")
    axis.text(cx, top - block + label / 2, "ONNX", ha="center", va="center", fontsize=15,
              fontweight="medium", color="#8E8E93", alpha=0.85, zorder=6)    # quieter than the mark

    axis.text(width / 2, 1.7, "Visual Inspection Model", ha="center", va="center", fontsize=14,
              fontweight="semibold", color=style.NEUTRAL["ink"], zorder=6)
    axis.text(width / 2, 1.25, caption(), ha="center", va="center", fontsize=9,
              color=style.NEUTRAL["secondary"], zorder=6)

    style.save(figure, DOCS, "model", panel=False)


if __name__ == "__main__":
    model_file()
    architecture()

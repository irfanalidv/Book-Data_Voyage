"""Box-and-arrow diagrams drawn at the size they are printed.

Chapter figures used to be drawn on a 14-16 inch canvas and then shrunk to the
5.5 inch text width of the book, which left their labels around 3 pt. These
helpers draw on a canvas measured in inches at roughly the printed width, so a
9 pt label stays close to 9 pt on the page.

Coordinates are inches from the bottom-left corner of the figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

DPI = 300
INK = "#1f2328"
MUTED = "#57606a"
ACCENT = "#e2262c"  # the red used on the book cover

# fill / border pairs, one per role a box can play
PALETTE: dict[str, tuple[str, str]] = {
    "input": ("#f3e8fd", "#8250df"),
    "step": ("#ddf4ff", "#0969da"),
    "llm": ("#fff1e5", "#bc4c00"),
    "store": ("#eaeef2", "#57606a"),
    "good": ("#dafbe1", "#1a7f37"),
    "bad": ("#ffebe9", "#cf222e"),
    "output": ("#dafbe1", "#1a7f37"),
}


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    title: str
    sub: str = ""
    kind: str = "step"

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


def _anchor(a: Box, b: Box) -> tuple[tuple[float, float], tuple[float, float]]:
    """Points on the edges of *a* and *b* facing each other."""
    dx, dy = b.cx - a.cx, b.cy - a.cy
    if abs(dx) * a.h >= abs(dy) * a.w:  # mostly horizontal
        sx = 1 if dx > 0 else -1
        return (a.cx + sx * a.w / 2, a.cy), (b.cx - sx * b.w / 2, b.cy)
    sy = 1 if dy > 0 else -1
    return (a.cx, a.cy + sy * a.h / 2), (b.cx, b.cy - sy * b.h / 2)


class Diagram:
    """A figure of fixed physical size with boxes, groups, arrows and notes."""

    def __init__(self, width: float, height: float, title: str = "") -> None:
        self.fig = plt.figure(figsize=(width, height))
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.axis("off")
        self.boxes: dict[str, Box] = {}
        if title:
            self.ax.text(
                width / 2,
                height - 0.12,
                title,
                ha="center",
                va="top",
                fontsize=10.5,
                fontweight="bold",
                color=INK,
            )

    def box(self, key: str, box: Box) -> Box:
        fill, edge = PALETTE[box.kind]
        self.ax.add_patch(
            FancyBboxPatch(
                (box.x, box.y),
                box.w,
                box.h,
                boxstyle="round,pad=0,rounding_size=0.06",
                fc=fill,
                ec=edge,
                lw=1.1,
                zorder=2,
            )
        )
        if box.sub:
            extra = 0.065 * box.sub.count("\n")  # lift the title above a two-line subtitle
            self.ax.text(
                box.cx,
                box.cy + 0.09 + extra,
                box.title,
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=INK,
                zorder=3,
            )
            self.ax.text(
                box.cx,
                box.cy - 0.12 + extra / 2,
                box.sub,
                ha="center",
                va="center",
                fontsize=7.2,
                color=MUTED,
                zorder=3,
                linespacing=1.15,
            )
        else:
            self.ax.text(
                box.cx,
                box.cy,
                box.title,
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=INK,
                zorder=3,
            )
        self.boxes[key] = box
        return box

    def group(self, x: float, y: float, w: float, h: float, label: str) -> None:
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0.08",
                fc="#f6f8fa",
                ec="#8c959f",
                lw=0.9,
                ls="--",
                zorder=1,
            )
        )
        self.ax.text(
            x + 0.08,
            y + h - 0.06,
            label,
            ha="left",
            va="top",
            fontsize=7.2,
            color=MUTED,
            style="italic",
            zorder=3,
        )

    def arrow(
        self,
        a: str,
        b: str,
        label: str = "",
        color: str = INK,
        dashed: bool = False,
        rad: float = 0.0,
        label_dy: float = 0.1,
        start: tuple[float, float] | None = None,
        end: tuple[float, float] | None = None,
    ) -> None:
        p, q = _anchor(self.boxes[a], self.boxes[b])
        p, q = start or p, end or q
        self.ax.add_patch(
            FancyArrowPatch(
                p,
                q,
                arrowstyle="-|>",
                mutation_scale=9,
                color=color,
                lw=1.0,
                ls="--" if dashed else "-",
                connectionstyle=f"arc3,rad={rad}",
                zorder=4,
                shrinkA=1,
                shrinkB=1,
            )
        )
        if label:
            mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
            self.ax.text(
                mx,
                my + label_dy,
                label,
                ha="center",
                va="bottom",
                fontsize=6.8,
                color=color if color != INK else MUTED,
                zorder=5,
            )

    def note(
        self, x: float, y: float, text: str, color: str = MUTED, ha: str = "left", size: float = 7.0
    ) -> None:
        self.ax.text(
            x, y, text, ha=ha, va="center", fontsize=size, color=color, linespacing=1.2, zorder=5
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=DPI, facecolor="white")
        plt.close(self.fig)
        return path

"""Inspect NMOS CSV data in a native Matplotlib window (no HTML or server)."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from dataset_io import discover, read_dataset


INPUTS = ("implant_dose_cm2", "anneal_time_s", "anneal_temperature_C")
SHORT = ("Dose", "Time", "Temp")
STATUSES = ("all", "safe", "unsafe", "unknown")
LABELS = {
    "implant_dose_cm2": "Implant dose (cm⁻²)",
    "anneal_time_s": "Anneal time (s)",
    "anneal_temperature_C": "Anneal temperature (°C)",
    "vth_V": "Threshold voltage (V)",
    "ioff_A_per_um": "Off-state leakage (A/µm)",
}
LOG = {"implant_dose_cm2", "ioff_A_per_um"}
STYLES = {"safe": ("#00796b", "o"), "unsafe": ("#b8412e", "x"), "unknown": ("#677587", "D")}


def load_views(source: Path, include_pilots: bool) -> list[dict]:
    views = []
    for path in discover(source, include_pilots):
        try:
            views.append(read_dataset(path))
        except (OSError, ValueError, csv.Error) as exc:
            print(f"Skipped {path}: {exc}")
    if not views:
        raise ValueError("No readable dataset snapshots. Wait for a completed recipe and reload.")
    return views


def slider_value(key, value):
    return math.log10(value) if key == INPUTS[0] else value


def fmt(value):
    return "unavailable" if value is None else f"{value:.4g}"


class RecipeViewer:
    def __init__(self, plt, source, include_pilots):
        from matplotlib.widgets import Button, RadioButtons, RangeSlider

        self.source, self.include_pilots = source, include_pilots
        self.views = load_views(source, include_pilots)
        self.index = 0
        self.selected = None
        self.sliders = {}
        self.updating = False
        self.RangeSlider = RangeSlider
        self.fig = plt.figure(figsize=(14, 8.7), facecolor="#f3f7f8")
        manager = self.fig.canvas.manager
        if manager is not None:
            manager.set_window_title("NMOS recipe explorer")
        self.fig.text(.06, .955, "NMOS recipe explorer", fontsize=21, weight="bold", color="#18313b")
        self.subtitle = self.fig.text(.06, .917, "", fontsize=11, color="#314b56")
        self.counts = self.fig.text(.06, .885, "", fontsize=10, color="#314b56")
        self.space = self.fig.add_axes((.075, .395, .434, .449))
        self.vth = self.fig.add_axes((.605, .658, .362, .186))
        self.leakage = self.fig.add_axes((.605, .395, .362, .18))
        self.detail = self.fig.text(.06, .046, "", fontsize=10, color="#18313b", linespacing=1.6)
        self.filter_axes = [self.fig.add_axes((.17, y, .275, .024)) for y in (.292, .226, .16)]
        self.status = RadioButtons(self.fig.add_axes((.555, .137, .124, .179)),
                                   ("All", "Safe", "Unsafe", "Unknown"), useblit=False)
        self.x = RadioButtons(self.fig.add_axes((.707, .137, .112, .179)), SHORT, useblit=False)
        self.y = RadioButtons(self.fig.add_axes((.847, .137, .112, .179)), SHORT, active=2, useblit=False)
        for control, label in ((self.status, "Safety filter"), (self.x, "Horizontal axis"), (self.y, "Recipe-space Y")):
            control.ax.set_title(label, fontsize=10, loc="left", pad=7)
            control.on_clicked(self.render)
        buttons = [(.598, "Previous", lambda _: self.move(-1)), (.696, "Next", lambda _: self.move(1)),
                   (.794, "Reset", self.reset), (.892, "Reload CSVs", self.reload)]
        self.buttons = []
        for x, label, action in buttons:
            button = Button(self.fig.add_axes((x, .936, .084, .039)), label)
            button.label.set_fontsize(9)
            button.on_clicked(action)
            self.buttons.append(button)
        self.fig.canvas.mpl_connect("pick_event", self.pick)
        self.set_dataset()

    @property
    def data(self):
        return self.views[self.index]

    def set_dataset(self):
        self.updating = True
        self.selected = None
        for slider in self.sliders.values():
            slider.disconnect_events()
        self.sliders.clear()
        for key, label, valfmt, ax in zip(INPUTS, ("log₁₀ dose", "Time (s)", "Temp (°C)"),
                                         ("%.2f", "%.1f", "%.0f"), self.filter_axes):
            ax.clear()
            values = [slider_value(key, row[key]) for row in self.data["rows"]]
            lo, hi = min(values), max(values)
            if lo == hi:
                # RangeSlider needs a nonzero range even for a one-row snapshot.
                epsilon = max(abs(lo) * 1e-6, 1e-6)
                lo, hi = lo - epsilon, hi + epsilon
            slider = self.RangeSlider(ax, label, lo, hi, valinit=(lo, hi), valfmt=valfmt)
            slider.label.set_fontsize(10)
            slider.valtext.set_fontsize(9)
            slider.on_changed(self.render)
            self.sliders[key] = slider
        self.status.set_active(0)
        self.updating = False
        self.render()

    def move(self, offset):
        self.index = (self.index + offset) % len(self.views)
        self.set_dataset()

    def reset(self, _):
        self.set_dataset()

    def reload(self, _):
        previous = self.data["source"]
        try:
            views = load_views(self.source, self.include_pilots)
        except (OSError, ValueError) as exc:
            print(f"Reload failed; showing the previous snapshot: {exc}")
            return
        self.views = views
        self.index = next((i for i, view in enumerate(views) if view["source"] == previous), 0)
        self.set_dataset()
        print(f"Reloaded {len(views)} dataset view(s).")

    def visible_indices(self):
        status = self.status.value_selected.lower()
        return [i for i, row in enumerate(self.data["rows"])
                if (status == "all" or row["status"] == status)
                and all(slider.val[0] <= slider_value(key, row[key]) <= slider.val[1]
                        for key, slider in self.sliders.items())]

    def pick(self, event):
        indices = self.artists.get(event.artist)
        if indices is not None and len(event.ind):
            self.selected = indices[int(event.ind[0])]
            self.render()

    def plot(self, ax, indices, x, y, title, ref_key=None, ref_label=None):
        ax.clear()
        rows = self.data["rows"]
        valid = [i for i in indices if all(rows[i][key] is not None and
                 (key not in LOG or rows[i][key] > 0) for key in (x, y))]
        ax.set_title(title, loc="left", fontsize=12, weight="bold")
        ax.set(xlabel=LABELS[x], ylabel=LABELS[y])
        if x in LOG:
            ax.set_xscale("log")
        if y in LOG:
            ax.set_yscale("log")
        ax.grid(alpha=.22)
        ax.set_axisbelow(True)
        for status, (color, marker) in STYLES.items():
            ids = [i for i in valid if rows[i]["status"] == status]
            if ids:
                self.artists[ax.scatter([rows[i][x] for i in ids], [rows[i][y] for i in ids],
                    c=color, marker=marker, s=39, picker=5, label=status.capitalize())] = ids
        if ref_key:
            refs = {r[ref_key] for r in rows if r[ref_key] is not None and
                    (y not in LOG or r[ref_key] > 0)}
            if len(refs) == 1:
                ref = next(iter(refs))
                ax.axhline(ref, color="#5746bb", ls="--", lw=1.2, label=f"{ref_label}: {ref:.3g}")
        if self.selected in valid:
            row = rows[self.selected]
            ax.scatter([row[x]], [row[y]], s=110, facecolors="none", edgecolors="#5746bb", linewidth=1.7)
        if not valid:
            ax.text(.5, .5, "No plottable observations", transform=ax.transAxes, ha="center", va="center")
        if len(valid) < len(indices):
            ax.text(.99, .02, f"{len(indices)-len(valid)} omitted", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color="#526671")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=8, framealpha=.9)

    def render(self, _=None):
        if self.updating:
            return
        x_index, y_index = SHORT.index(self.x.value_selected), SHORT.index(self.y.value_selected)
        if x_index == y_index:
            y_index = (x_index + 1) % len(INPUTS)
            self.updating = True
            self.y.set_active(y_index)
            self.updating = False
        x, y = INPUTS[x_index], INPUTS[y_index]
        indices = self.visible_indices()
        if self.selected not in indices:
            self.selected = indices[0] if indices else None
        rows = self.data["rows"]
        counts = {s: sum(rows[i]["status"] == s for i in indices) for s in STYLES}
        self.subtitle.set_text(f"{self.data['device']}  ·  {self.data['phase']}  ·  {self.data['qualification']}")
        self.counts.set_text(f"View {self.index+1}/{len(self.views)}  ·  {len(indices)}/{len(rows)} shown  ·  "
                             f"{counts['safe']} safe  ·  {counts['unsafe']} unsafe  ·  {counts['unknown']} unknown")
        self.artists = {}
        self.plot(self.space, indices, x, y, "Sampled recipe space")
        self.plot(self.vth, indices, x, "vth_V", "Threshold voltage", "target_vth_V", "Target")
        self.plot(self.leakage, indices, x, "ioff_A_per_um", "Off-state leakage", "leakage_limit_A_per_um", "Limit")
        if self.selected is None:
            self.detail.set_text("No recipe matches these filters. Reset or widen the ranges.")
        else:
            row = rows[self.selected]
            self.detail.set_text(
                f"Sample {row['sample']} · {row['status']} · dose {fmt(row[INPUTS[0]])} cm⁻² · "
                f"{fmt(row[INPUTS[1]])} s · {fmt(row[INPUTS[2]])} °C\n"
                f"Vth {fmt(row['vth_V'])} V · Ioff {fmt(row['ioff_A_per_um'])} A/µm · "
                f"margin {fmt(row['safety_margin'])} · solver {row['solver'] or 'unknown'}"
            )
        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs_parallel_100"))
    parser.add_argument("--include-pilots", action="store_true", help="Show pilot CSVs as separate views")
    parser.add_argument("--save", type=Path, help="Save the initial view as a PNG instead of opening a window")
    args = parser.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg" if args.save else "TkAgg")
        import matplotlib.pyplot as plt
        viewer = RecipeViewer(plt, args.input, args.include_pilots)
        if args.save:
            if args.save.suffix.lower() != ".png":
                raise ValueError("--save requires a .png filename")
            args.save.parent.mkdir(parents=True, exist_ok=True)
            viewer.fig.savefig(args.save, dpi=130)
            print(f"Saved {args.save.resolve()}")
        else:
            print("Click points or adjust filters in the plot window. Close it to return to the terminal.")
            plt.show()
    except ImportError as exc:
        parser.exit(1, f"Plot viewer needs Matplotlib and Tk for interactive mode: {exc}\n"
                        "Install matplotlib in the VS Code interpreter, or use --save for a headless PNG.\n")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Plot viewer: {exc}\n")


if __name__ == "__main__":
    main()

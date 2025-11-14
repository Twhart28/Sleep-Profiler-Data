"""Sleep EDF Viewer

This script provides a simple Tkinter-based user interface for exploring
channels in an EDF sleep study. When executed it immediately prompts the user
for an EDF file, then displays the signals with basic navigation controls.
"""

from __future__ import annotations
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Tuple

import itertools

try:
    import pyedflib  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    message = (
        "Missing optional dependency 'pyedflib'.\n"
        "Install it with 'pip install pyedflib' and re-run this tool."
    )
    raise SystemExit(message) from exc

import numpy as np

# ``matplotlib`` needs to be imported after Tk has been initialised on some
# systems. The ``TkAgg`` backend embeds a Matplotlib canvas inside Tkinter.
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class EDFFile:
    """Load and cache EDF signal data for convenient access."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._reader = pyedflib.EdfReader(path)
        self.signal_labels: List[str] = list(self._reader.getSignalLabels())
        self.sample_frequency: Dict[str, float] = {
            label: float(self._reader.getSampleFrequency(i))
            for i, label in enumerate(self.signal_labels)
        }
        self.duration_seconds: float = float(self._reader.getFileDuration())
        self._signal_cache: Dict[str, np.ndarray] = {}

    def __del__(self) -> None:  # pragma: no cover - safety best-effort cleanup
        try:
            self._reader.close()
        except Exception:
            pass

    def get_signal(self, label: str) -> Tuple[np.ndarray, float]:
        """Return cached (signal, sample_rate) data for a channel label."""

        if label not in self.signal_labels:
            raise KeyError(f"Channel '{label}' not found in EDF file")

        if label not in self._signal_cache:
            index = self.signal_labels.index(label)
            signal = self._reader.readSignal(index)
            self._signal_cache[label] = signal.astype(np.float64)
        return self._signal_cache[label], self.sample_frequency[label]


class EDFViewer(tk.Tk):
    """Main application window for browsing EDF channels."""

    def __init__(self, edf_file: EDFFile) -> None:
        super().__init__()
        self.title(f"Sleep EDF Viewer - {edf_file.path}")
        self.geometry("1200x700")

        self.edf_file = edf_file
        self.window_duration = tk.DoubleVar(value=30.0)
        self.start_time = tk.DoubleVar(value=0.0)
        self.autoscale = tk.BooleanVar(value=True)

        self.selected_channels: List[str] = list(edf_file.signal_labels)
        self.channel_scales: Dict[str, tk.DoubleVar] = {}
        for label in edf_file.signal_labels:
            scale_value = self._estimate_initial_scale(label)
            self.channel_scales[label] = tk.DoubleVar(value=scale_value)

        self.active_axes: List[Axes] = []
        self.hover_label = tk.StringVar(value="")
        self.channel_dialog: ChannelOptionsDialog | None = None

        self._build_layout()
        self._update_plot()

    # ------------------------------------------------------------------
    # UI construction helpers
    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self, padding=12)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.columnconfigure(0, weight=1)

        ttk.Label(sidebar, text="View Controls", font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, sticky="w"
        )

        presets_frame = ttk.LabelFrame(sidebar, text="Window size")
        presets_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        presets_frame.columnconfigure(1, weight=1)

        ttk.Label(presets_frame, text="Presets:").grid(row=0, column=0, sticky="w")
        preset_values = ["1 s", "5 s", "10 s", "30 s", "1 min", "5 min", "Custom"]
        self.window_preset = tk.StringVar(value="30 s")
        preset_combo = ttk.Combobox(
            presets_frame,
            values=preset_values,
            state="readonly",
            textvariable=self.window_preset,
        )
        preset_combo.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        preset_combo.bind("<<ComboboxSelected>>", self._on_window_preset)

        ttk.Label(presets_frame, text="Custom (s):").grid(row=1, column=0, sticky="w", pady=(8, 0))
        custom_entry = ttk.Entry(
            presets_frame,
            textvariable=self.window_duration,
            width=8,
        )
        custom_entry.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=(8, 0))
        custom_entry.bind("<Return>", lambda _event: self._update_plot())
        custom_entry.bind("<FocusOut>", lambda _event: self._update_plot())

        ttk.Label(sidebar, text="Navigation", font=("TkDefaultFont", 11, "bold")).grid(
            row=2, column=0, sticky="w", pady=(16, 0)
        )

        nav_frame = ttk.Frame(sidebar)
        nav_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        nav_frame.columnconfigure(0, weight=1)

        self.time_scale = ttk.Scale(
            nav_frame,
            from_=0.0,
            to=self.edf_file.duration_seconds,
            variable=self.start_time,
            command=self._on_time_change,
        )
        self.time_scale.grid(row=0, column=0, sticky="ew")

        self.time_label = ttk.Label(nav_frame, text="Start: 0.0 s")
        self.time_label.grid(row=1, column=0, sticky="w", pady=(4, 0))

        ttk.Label(sidebar, text="Channels", font=("TkDefaultFont", 11, "bold")).grid(
            row=4, column=0, sticky="w", pady=(16, 0)
        )

        channel_controls = ttk.Frame(sidebar)
        channel_controls.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        channel_controls.columnconfigure(0, weight=1)

        ttk.Button(
            channel_controls,
            text="Options…",
            command=self._open_channel_options,
        ).grid(row=0, column=0, sticky="ew")

        ttk.Checkbutton(
            channel_controls,
            text="Autoscale (window)",
            variable=self.autoscale,
            command=self._on_autoscale_toggle,
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

        info = ttk.Label(
            sidebar,
            text=(
                f"Duration: {self.edf_file.duration_seconds:.1f} s\n"
                f"Channels: {len(self.edf_file.signal_labels)}"
            ),
            justify="left",
        )
        info.grid(row=6, column=0, sticky="w", pady=(16, 0))

        # Plotting area
        plot_frame = ttk.Frame(self, padding=10)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(9, 6), constrained_layout=True)
        self.figure.patch.set_facecolor("#f7f9fc")
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        status = ttk.Label(plot_frame, textvariable=self.hover_label, anchor="w")
        status.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    # ------------------------------------------------------------------
    # Event handlers
    def _on_window_preset(self, _event: object) -> None:
        choice = self.window_preset.get()
        if choice.lower() == "custom":
            return
        seconds = self._parse_window_preset(choice)
        self.window_duration.set(seconds)
        self._update_plot()

    def _on_time_change(self, _value: str) -> None:
        self.time_label.configure(text=f"Start: {self.start_time.get():.1f} s")
        self._update_plot()

    def _on_autoscale_toggle(self) -> None:
        self._update_plot()
        if self.channel_dialog is not None and tk.Toplevel.winfo_exists(self.channel_dialog):
            self.channel_dialog.update_scale_state(self.autoscale.get())

    def _open_channel_options(self) -> None:
        if self.channel_dialog is not None and tk.Toplevel.winfo_exists(self.channel_dialog):
            self.channel_dialog.focus_set()
            return
        self.channel_dialog = ChannelOptionsDialog(self)

    # ------------------------------------------------------------------
    # Plotting helpers
    def _update_plot(self) -> None:
        if not self.selected_channels:
            self.figure.clear()
            self.hover_label.set("No channels selected.")
            self.canvas.draw_idle()
            return

        try:
            window_seconds = float(self.window_duration.get())
        except (tk.TclError, ValueError):
            window_seconds = 0.5
        window_seconds = max(0.5, window_seconds)
        self.window_duration.set(window_seconds)
        self.window_preset.set(self._match_preset(window_seconds))

        self._update_time_slider_limits(window_seconds)

        start_seconds = float(self.start_time.get())
        max_start = max(0.0, self.edf_file.duration_seconds - window_seconds)
        start_seconds = min(max(start_seconds, 0.0), max_start)
        self.start_time.set(start_seconds)
        self.time_label.configure(text=f"Start: {start_seconds:.2f} s")

        colors = itertools.cycle(
            matplotlib.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"])
        )

        self.figure.clear()
        axes = self.figure.subplots(len(self.selected_channels), 1, sharex=True)
        if isinstance(axes, Axes):
            axes_list: List[Axes] = [axes]
        else:
            axes_list = list(np.atleast_1d(axes))

        self.active_axes = axes_list

        for ax, label, color in zip(axes_list, self.selected_channels, colors):
            ax.set_ylabel(label, rotation=0, labelpad=40, fontsize=9, va="center")
            try:
                signal, sample_rate = self.edf_file.get_signal(label)
            except KeyError:
                ax.text(
                    0.5,
                    0.5,
                    "Channel unavailable",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="red",
                )
                continue

            total_samples = len(signal)
            start_index = max(0, int(start_seconds * sample_rate))
            if total_samples:
                start_index = min(start_index, total_samples - 1)
            end_index = int((start_seconds + window_seconds) * sample_rate)
            end_index = max(start_index + 1, end_index)
            end_index = min(total_samples, end_index)

            indices = np.arange(start_index, end_index)
            if indices.size == 0:
                indices = np.array([start_index])
                segment = np.array([0.0])
            else:
                segment = signal[start_index:end_index]
            if sample_rate > 0:
                time_axis = indices / sample_rate
            else:
                time_axis = np.linspace(start_seconds, start_seconds + window_seconds, indices.size)

            ax.plot(time_axis, segment, color=color, linewidth=0.8)
            ax.axhline(0, color="0.6", linewidth=0.5, linestyle="--", alpha=0.7)
            ax.grid(True, which="major", linestyle=":", linewidth=0.3, alpha=0.5)
            ax.set_facecolor("#ffffff")

            scale_limits = self._determine_scale(label, segment)
            ax.set_ylim(scale_limits)
            scale_text = self._format_scale_text(scale_limits)
            ax.text(
                0.995,
                0.9,
                scale_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9),
            )

        if axes_list:
            axes_list[-1].set_xlabel("Time (s)")
            self.figure.align_ylabels(axes_list)
        self.canvas.draw_idle()

    def _estimate_initial_scale(self, label: str) -> float:
        try:
            signal, sample_rate = self.edf_file.get_signal(label)
        except KeyError:
            return 1.0

        window_samples = int(min(len(signal), sample_rate * 10))
        if window_samples <= 0:
            return 1.0

        subset = np.abs(signal[:window_samples])
        percentile = np.percentile(subset, 95) if subset.size else 1.0
        return float(max(percentile, 1.0))

    def _determine_scale(self, label: str, segment: np.ndarray) -> Tuple[float, float]:
        if segment.size == 0:
            return (-1.0, 1.0)

        if self.autoscale.get():
            max_val = np.max(np.abs(segment))
            if max_val == 0:
                max_val = 1.0
            margin = max_val * 0.1
            return (-max_val - margin, max_val + margin)

        scale_value = max(1e-6, float(self.channel_scales[label].get()))
        self.channel_scales[label].set(scale_value)
        return (-scale_value, scale_value)

    def _format_scale_text(self, limits: Tuple[float, float]) -> str:
        peak = max(abs(limits[0]), abs(limits[1]))
        if peak >= 1_000_000:
            return f"±{peak / 1_000_000:.1f}M"
        if peak >= 1_000:
            return f"±{peak / 1_000:.1f}k"
        if peak < 1:
            return f"±{peak * 1_000:.1f}m"
        return f"±{peak:.1f}"

    def _update_time_slider_limits(self, window_seconds: float) -> None:
        max_start = max(0.0, self.edf_file.duration_seconds - window_seconds)
        upper = max_start if max_start > 0 else self.edf_file.duration_seconds
        self.time_scale.configure(from_=0.0, to=max(upper, 0.0))

    def _parse_window_preset(self, value: str) -> float:
        value = value.strip().lower()
        if value.endswith("min"):
            number = float(value.split()[0])
            return number * 60.0
        if value.endswith("s"):
            number = float(value.split()[0])
            return number
        try:
            return float(value)
        except ValueError:
            return self._current_window_duration()

    def _match_preset(self, seconds: float) -> str:
        presets = {
            1.0: "1 s",
            5.0: "5 s",
            10.0: "10 s",
            30.0: "30 s",
            60.0: "1 min",
            300.0: "5 min",
        }
        for preset_seconds, label in presets.items():
            if abs(seconds - preset_seconds) < 1e-6:
                return label
        return "Custom"

    def _current_window_duration(self) -> float:
        try:
            return float(self.window_duration.get())
        except (tk.TclError, ValueError):
            return 0.5

    def _on_mouse_move(self, event: MouseEvent) -> None:
        if event.inaxes not in self.active_axes:
            self.hover_label.set("")
            return

        axis_index = self.active_axes.index(event.inaxes)
        if axis_index >= len(self.selected_channels):
            self.hover_label.set("")
            return

        label = self.selected_channels[axis_index]
        x = event.xdata if event.xdata is not None else float("nan")
        y = event.ydata if event.ydata is not None else float("nan")
        self.hover_label.set(f"{label}: t={x:.3f}s, value={y:.3f}")


class ChannelOptionsDialog(tk.Toplevel):
    """Dialog window for selecting channels and manual scales."""

    def __init__(self, viewer: EDFViewer) -> None:
        super().__init__(viewer)
        self.title("Channel options")
        self.viewer = viewer
        self.resizable(False, True)
        self.transient(viewer)
        self.grab_set()

        ttk.Label(self, text="Select channels to display", font=("TkDefaultFont", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(12, 6)
        )

        self.channel_vars: Dict[str, tk.BooleanVar] = {}
        self.scale_entries: Dict[str, ttk.Entry] = {}

        channels_frame = ttk.Frame(self)
        channels_frame.grid(row=1, column=0, columnspan=2, padx=12, pady=(0, 12), sticky="nsew")
        channels_frame.columnconfigure(1, weight=1)

        for row, label in enumerate(self.viewer.edf_file.signal_labels):
            var = tk.BooleanVar(value=label in self.viewer.selected_channels)
            self.channel_vars[label] = var

            ttk.Checkbutton(channels_frame, text=label, variable=var).grid(
                row=row, column=0, sticky="w"
            )

            scale_entry = ttk.Entry(
                channels_frame,
                textvariable=self.viewer.channel_scales[label],
                width=10,
                state="disabled" if self.viewer.autoscale.get() else "normal",
                justify="right",
            )
            scale_entry.grid(row=row, column=1, sticky="ew", padx=(8, 0))
            self.scale_entries[label] = scale_entry

        ttk.Label(self, text="Manual scale applies when autoscale is disabled.").grid(
            row=2, column=0, columnspan=2, sticky="w", padx=12
        )

        button_frame = ttk.Frame(self)
        button_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(8, 12))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        ttk.Button(button_frame, text="Cancel", command=self._on_close).grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(button_frame, text="Apply", command=self._apply).grid(
            row=0, column=1, sticky="ew"
        )

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.update_scale_state(self.viewer.autoscale.get())

    def _apply(self) -> None:
        selected = [label for label, var in self.channel_vars.items() if var.get()]
        if not selected:
            messagebox.showwarning("No channels", "Please select at least one channel to display.")
            return

        self.viewer.selected_channels = selected
        self.viewer._update_plot()
        self._on_close()

    def update_scale_state(self, autoscale: bool) -> None:
        state = "disabled" if autoscale else "normal"
        for entry in self.scale_entries.values():
            entry.configure(state=state)

    def _on_close(self) -> None:
        if self.viewer.channel_dialog is self:
            self.viewer.channel_dialog = None
        self.destroy()


def ask_for_edf_file() -> str | None:
    """Prompt the user to select an EDF file and return its path."""

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select EDF file",
        filetypes=[("EDF files", "*.edf"), ("All files", "*.*")],
    )
    root.destroy()
    return file_path or None


def main(argv: List[str]) -> int:
    if len(argv) > 1:
        file_path = argv[1]
    else:
        file_path = ask_for_edf_file()
        if not file_path:
            print("No EDF file selected. Exiting.")
            return 1

    try:
        edf_file = EDFFile(file_path)
    except FileNotFoundError:
        messagebox.showerror("File not found", f"Could not locate '{file_path}'.")
        return 1
    except OSError as exc:
        messagebox.showerror("EDF error", f"Could not open EDF file:\n{exc}")
        return 1

    viewer = EDFViewer(edf_file)
    viewer.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

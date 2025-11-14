"""Sleep EDF Viewer

This script provides a simple Tkinter-based user interface for exploring
channels in an EDF sleep study. When executed it immediately prompts the user
for an EDF file, then displays the signals with basic navigation controls.
"""

from __future__ import annotations
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Tuple, Callable

import itertools
import math

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
from matplotlib.ticker import AutoLocator, FuncFormatter


def _format_seconds_hms(seconds: float) -> str:
    """Return a HH:MM:SS string for the provided number of seconds."""

    if not math.isfinite(seconds):
        return "--:--:--"
    seconds = max(0.0, seconds)
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_seconds_hms_ms(seconds: float) -> str:
    """Return HH:MM:SS.mmm format for display in hover widgets."""

    if not math.isfinite(seconds):
        return "--:--:--.---"
    seconds = max(0.0, seconds)
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int(round((seconds - total_seconds) * 1000))
    # When rounding pushes millis to 1000 roll the values forward.
    if millis >= 1000:
        millis -= 1000
        secs += 1
        if secs >= 60:
            secs -= 60
            minutes += 1
            if minutes >= 60:
                minutes -= 60
                hours += 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


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


class TimeNavigator(tk.Canvas):
    """Canvas-based slider that visualises the current time window."""

    def __init__(self, master: tk.Widget, command: Callable[[float], None]) -> None:
        super().__init__(master, height=36, highlightthickness=0)
        self._command = command
        self.total_duration = 1.0
        self.window_duration = 1.0
        self.start_seconds = 0.0
        self._dragging = False
        self._drag_offset = 0.0

        self.bind("<Configure>", lambda _event: self._redraw())
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    # ------------------------------------------------------------------
    def set_state(
        self,
        total_duration: float,
        window_duration: float,
        start_seconds: float,
    ) -> None:
        self.total_duration = max(total_duration, 1e-6)
        self.window_duration = max(min(window_duration, self.total_duration), 1e-3)
        max_start = max(0.0, self.total_duration - self.window_duration)
        self.start_seconds = min(max(start_seconds, 0.0), max_start)
        self._redraw()

    # ------------------------------------------------------------------
    def _on_press(self, event: tk.Event) -> None:
        track_width = max(self.winfo_width() - 12, 1)
        start_px = 6 + (self.start_seconds / self.total_duration) * track_width
        window_px = max(track_width * (self.window_duration / self.total_duration), 4)
        window_px = min(window_px, track_width)
        if start_px + window_px > 6 + track_width:
            start_px = 6 + track_width - window_px
        if start_px <= event.x <= start_px + window_px:
            self._dragging = True
            self._drag_offset = event.x - start_px
        else:
            self._dragging = True
            self._drag_offset = window_px / 2
            self._update_start_from_x(event.x)

    def _on_drag(self, event: tk.Event) -> None:
        if not self._dragging:
            return
        self._update_start_from_x(event.x, self._drag_offset)

    def _on_release(self, _event: tk.Event) -> None:
        self._dragging = False

    def _update_start_from_x(self, x: float, offset: float | None = None) -> None:
        track_width = max(self.winfo_width() - 12, 1)
        window_px = max(track_width * (self.window_duration / self.total_duration), 4)
        window_px = min(window_px, track_width)
        if offset is None:
            offset = window_px / 2
        left_limit = 6
        right_limit = left_limit + track_width - window_px
        new_start_px = min(max(x - offset, left_limit), right_limit)
        fraction = 0.0 if track_width <= window_px else (new_start_px - left_limit) / (track_width - window_px)
        max_start = max(0.0, self.total_duration - self.window_duration)
        new_start = fraction * max_start
        if abs(new_start - self.start_seconds) > 1e-6:
            self.start_seconds = new_start
            self._redraw()
            if self._command is not None:
                self._command(self.start_seconds)

    # ------------------------------------------------------------------
    def _redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 1)
        height = max(self.winfo_height(), 1)
        margin = 6
        track_left = margin
        track_right = width - margin
        track_top = margin
        track_bottom = height - margin
        if track_right <= track_left:
            track_right = track_left + 1
        self.create_rounded_rect(track_left, track_top, track_right, track_bottom, radius=8, fill="#d6e3f3", outline="")
        track_width = track_right - track_left
        window_px = max(track_width * (self.window_duration / self.total_duration), 6)
        window_px = min(window_px, track_width)
        max_start_px = track_right - window_px
        start_px = track_left + (self.start_seconds / self.total_duration) * track_width
        if start_px + window_px > track_right:
            start_px = max_start_px
        start_px = max(track_left, min(start_px, max_start_px))
        end_px = start_px + window_px
        self.create_rounded_rect(start_px, track_top, end_px, track_bottom, radius=8, fill="#5b8def", outline="")
        end_time = min(self.start_seconds + self.window_duration, self.total_duration)
        text = f"{_format_seconds_hms(self.start_seconds)} – {_format_seconds_hms(end_time)}"
        self.create_text(
            width / 2,
            height / 2,
            text=text,
            fill="#1f3b66",
            font=("TkDefaultFont", 10, "bold"),
        )

    # ------------------------------------------------------------------
    def create_rounded_rect(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        radius: float = 10,
        **kwargs: object,
    ) -> None:
        """Draw a rounded rectangle on the canvas."""

        radius = min(radius, abs(x2 - x1) / 2, abs(y2 - y1) / 2)
        points = [
            x1 + radius,
            y1,
            x2 - radius,
            y1,
            x2,
            y1,
            x2,
            y1 + radius,
            x2,
            y2 - radius,
            x2,
            y2,
            x2 - radius,
            y2,
            x1 + radius,
            y2,
            x1,
            y2,
            x1,
            y2 - radius,
            x1,
            y1 + radius,
            x1,
            y1,
        ]
        self.create_polygon(points, smooth=True, **kwargs)


class ChannelPlot:
    """Represent a single channel plot within the viewer."""

    def __init__(self, viewer: "EDFViewer", label: str, color: str) -> None:
        self.viewer = viewer
        self.label = label
        self.color = color

        self.frame = ttk.Frame(viewer.paned)
        self.frame.columnconfigure(0, weight=0)
        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(0, weight=1)

        self.control_frame = ttk.Frame(self.frame, padding=(0, 6, 6, 6))
        self.control_frame.grid(row=0, column=0, sticky="nsw")
        self.control_frame.columnconfigure(0, weight=1)

        self.channel_label = ttk.Label(
            self.control_frame,
            text=label,
            font=("TkDefaultFont", 10, "bold"),
            anchor="center",
            justify="center",
        )
        self.channel_label.grid(row=0, column=0, sticky="ew")

        self.value_var = tk.StringVar(value="--")
        self.value_label = ttk.Label(
            self.control_frame,
            textvariable=self.value_var,
            width=10,
            anchor="center",
            padding=(2, 6),
        )
        self.value_label.grid(row=1, column=0, sticky="ew", pady=(4, 8))

        button_frame = ttk.Frame(self.control_frame)
        button_frame.grid(row=2, column=0, sticky="n")
        ttk.Button(button_frame, text="+", width=3, command=self._scale_up).grid(row=0, column=0, pady=(0, 4))
        ttk.Button(button_frame, text="Auto", width=5, command=self._autoscale).grid(row=1, column=0, pady=4)
        ttk.Button(button_frame, text="-", width=3, command=self._scale_down).grid(row=2, column=0, pady=(4, 0))

        self.scale_label = ttk.Label(self.control_frame, text="", anchor="center")
        self.scale_label.grid(row=3, column=0, sticky="ew", pady=(8, 0))

        self.figure = Figure(figsize=(7, 1.8))
        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.2)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#ffffff")
        self.ax.grid(True, which="major", linestyle=":", linewidth=0.3, alpha=0.5)
        self.ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _pos: _format_seconds_hms(val)))
        self.ax.set_ylabel("")
        self.line, = self.ax.plot([], [], color=color, linewidth=0.8)
        self.ax.axhline(0, color="0.6", linewidth=0.5, linestyle="--", alpha=0.7)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")

        self._hover_cid = self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        self._time_axis = np.array([], dtype=float)
        self._values = np.array([], dtype=float)

    # ------------------------------------------------------------------
    def destroy(self) -> None:
        if self._hover_cid is not None:
            self.canvas.mpl_disconnect(self._hover_cid)
            self._hover_cid = None
        self.canvas.get_tk_widget().destroy()
        self.frame.destroy()

    # ------------------------------------------------------------------
    def update_data(
        self,
        time_axis: np.ndarray,
        values: np.ndarray,
        limits: Tuple[float, float],
        window_seconds: float,
        show_xlabel: bool,
    ) -> None:
        self._time_axis = np.asarray(time_axis, dtype=float)
        self._values = np.asarray(values, dtype=float)
        self.line.set_data(self._time_axis, self._values)
        self._set_value_display(None)
        if time_axis.size:
            self.ax.set_xlim(self._time_axis[0], self._time_axis[-1])
        else:
            start = self.viewer.start_time.get()
            self.ax.set_xlim(start, start + window_seconds)
        self.ax.set_ylim(limits)
        self.ax.yaxis.set_major_locator(AutoLocator())
        self._update_y_ticks(limits)
        if show_xlabel:
            self.ax.set_xlabel("Time (HH:MM:SS)")
            self.ax.tick_params(axis="x", which="both", labelbottom=True)
        else:
            self.ax.set_xlabel("")
            self.ax.tick_params(axis="x", which="both", labelbottom=False)
        self.scale_label.configure(text=self.viewer._format_scale_text(limits))
        self.canvas.draw_idle()

    def refresh_formatter(self) -> None:
        self.ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _pos: _format_seconds_hms(val)))

    # ------------------------------------------------------------------
    def _scale_up(self) -> None:
        self.viewer.adjust_manual_scale(self.label, factor=1.25)

    def _scale_down(self) -> None:
        self.viewer.adjust_manual_scale(self.label, factor=0.8)

    def _autoscale(self) -> None:
        self.viewer.set_channel_autoscale(self.label, True)

    def _on_mouse_move(self, event: MouseEvent) -> None:
        if event.xdata is None or event.ydata is None or event.inaxes is None:
            self._set_value_display(None)
            self.viewer.clear_hover()
            return
        time_value = float(event.xdata)
        value = self._value_at_time(time_value)
        self._set_value_display(value)
        if value is None:
            self.viewer.clear_hover()
            return
        self.viewer.update_hover(self.label, time_value, value)

    def _set_value_display(self, value: float | None) -> None:
        if value is None or not math.isfinite(value):
            self.value_var.set("--")
        else:
            self.value_var.set(f"{value:.4f}")

    def _value_at_time(self, time_value: float) -> float | None:
        if self._time_axis.size == 0 or self._values.size == 0:
            return None
        idx = np.searchsorted(self._time_axis, time_value)
        if idx <= 0:
            nearest = 0
        elif idx >= self._time_axis.size:
            nearest = self._time_axis.size - 1
        else:
            prev_idx = idx - 1
            if abs(self._time_axis[idx] - time_value) < abs(self._time_axis[prev_idx] - time_value):
                nearest = idx
            else:
                nearest = prev_idx
        if 0 <= nearest < self._values.size:
            return float(self._values[nearest])
        return None

    def _update_y_ticks(self, limits: Tuple[float, float]) -> None:
        renderer = self.canvas.get_renderer()
        if renderer is None:
            self.canvas.draw()
            renderer = self.canvas.get_renderer()
        if renderer is None:
            return
        axis_bbox = self.ax.get_window_extent(renderer=renderer)
        height_px = axis_bbox.height
        ticks = list(self.ax.get_yticks())
        if not ticks:
            return
        approx_label_height = 14  # pixels
        if len(ticks) > 1 and height_px <= approx_label_height * (len(ticks) + 0.5):
            center = (limits[0] + limits[1]) / 2.0
            self.ax.set_yticks([center])
        else:
            self.ax.set_yticks(ticks)

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
        self.channel_autoscale_override: Dict[str, bool] = {}
        for label in edf_file.signal_labels:
            scale_value = self._estimate_initial_scale(label)
            self.channel_scales[label] = tk.DoubleVar(value=scale_value)
            self.channel_autoscale_override[label] = True

        self.hover_label = tk.StringVar(value="")
        self.channel_dialog: ChannelOptionsDialog | None = None
        self.channel_plots: Dict[str, ChannelPlot] = {}
        self.channel_colors: Dict[str, str] = {}
        self._color_cycle = itertools.cycle(
            matplotlib.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"])
        )

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

        ttk.Label(sidebar, text="Channels", font=("TkDefaultFont", 11, "bold")).grid(
            row=2, column=0, sticky="w", pady=(16, 0)
        )

        channel_controls = ttk.Frame(sidebar)
        channel_controls.grid(row=3, column=0, sticky="ew", pady=(8, 0))
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
                f"Duration: {_format_seconds_hms(self.edf_file.duration_seconds)}\n"
                f"Channels: {len(self.edf_file.signal_labels)}"
            ),
            justify="left",
        )
        info.grid(row=4, column=0, sticky="w", pady=(16, 0))

        # Plotting area
        plot_frame = ttk.Frame(self, padding=(10, 10, 10, 4))
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self.paned = tk.PanedWindow(
            plot_frame,
            orient=tk.VERTICAL,
            sashrelief=tk.SOLID,
            sashwidth=12,
            sashpad=4,
            bg="#111111",
            bd=0,
            showhandle=True,
            handlepad=18,
            handlesize=30,
            sashcursor="sb_v_double_arrow",
        )
        self.paned.grid(row=0, column=0, sticky="nsew")

        nav_frame = ttk.Frame(plot_frame, padding=(0, 8, 0, 0))
        nav_frame.grid(row=1, column=0, sticky="ew")
        nav_frame.columnconfigure(0, weight=1)

        self.navigator = TimeNavigator(nav_frame, command=self._on_navigator_change)
        self.navigator.grid(row=0, column=0, sticky="ew")

        self.time_label = ttk.Label(nav_frame, text="Start: 00:00:00", anchor="w")
        self.time_label.grid(row=1, column=0, sticky="w", pady=(6, 0))

        status = ttk.Label(nav_frame, textvariable=self.hover_label, anchor="w")
        status.grid(row=2, column=0, sticky="ew", pady=(4, 0))

    # ------------------------------------------------------------------
    # Event handlers
    def _on_window_preset(self, _event: object) -> None:
        choice = self.window_preset.get()
        if choice.lower() == "custom":
            return
        seconds = self._parse_window_preset(choice)
        self.window_duration.set(seconds)
        self._update_plot()

    def _on_navigator_change(self, start_seconds: float) -> None:
        self.start_time.set(start_seconds)
        self._update_plot()

    def _on_autoscale_toggle(self) -> None:
        new_state = self.autoscale.get()
        for label in self.edf_file.signal_labels:
            self.channel_autoscale_override[label] = new_state
        self._update_plot()
        if self.channel_dialog is not None and tk.Toplevel.winfo_exists(self.channel_dialog):
            self.channel_dialog.update_scale_state()

    def _open_channel_options(self) -> None:
        if self.channel_dialog is not None and tk.Toplevel.winfo_exists(self.channel_dialog):
            self.channel_dialog.focus_set()
            return
        self.channel_dialog = ChannelOptionsDialog(self)

    # ------------------------------------------------------------------
    # Plotting helpers
    def _update_plot(self) -> None:
        try:
            window_seconds = float(self.window_duration.get())
        except (tk.TclError, ValueError):
            window_seconds = 0.5
        window_seconds = max(0.5, min(window_seconds, self.edf_file.duration_seconds))
        self.window_duration.set(window_seconds)
        self.window_preset.set(self._match_preset(window_seconds))

        total_duration = self.edf_file.duration_seconds
        max_start = max(0.0, total_duration - window_seconds)
        start_seconds = float(self.start_time.get())
        start_seconds = min(max(start_seconds, 0.0), max_start)
        self.start_time.set(start_seconds)
        self.time_label.configure(text=f"Start: {_format_seconds_hms(start_seconds)}")
        self.navigator.set_state(total_duration, window_seconds, start_seconds)

        if not self.selected_channels:
            self.hover_label.set("No channels selected.")
            self._clear_channel_plots()
            return

        self.hover_label.set("")
        self._sync_channel_plots()

        for index, label in enumerate(self.selected_channels):
            try:
                signal, sample_rate = self.edf_file.get_signal(label)
            except KeyError:
                continue

            total_samples = len(signal)
            if sample_rate <= 0 or total_samples == 0:
                time_axis = np.array([start_seconds, start_seconds + window_seconds])
                segment = np.zeros_like(time_axis)
            else:
                start_index = int(start_seconds * sample_rate)
                start_index = min(max(start_index, 0), max(total_samples - 1, 0))
                end_index = int((start_seconds + window_seconds) * sample_rate)
                end_index = max(start_index + 1, end_index)
                end_index = min(total_samples, end_index)
                indices = np.arange(start_index, end_index)
                segment = signal[start_index:end_index]
                time_axis = indices / sample_rate if indices.size else np.array([start_seconds])

            limits = self._determine_scale(label, segment)
            plot = self.channel_plots[label]
            plot.refresh_formatter()
            show_xlabel = index == len(self.selected_channels) - 1
            plot.update_data(time_axis, segment, limits, window_seconds, show_xlabel)

    def _sync_channel_plots(self) -> None:
        for label in list(self.channel_plots.keys()):
            if label not in self.selected_channels:
                plot = self.channel_plots.pop(label)
                self.paned.forget(plot.frame)
                plot.destroy()

        for label in self.selected_channels:
            if label not in self.channel_plots:
                color = self.channel_colors.get(label)
                if color is None:
                    color = next(self._color_cycle)
                    self.channel_colors[label] = color
                plot = ChannelPlot(self, label, color)
                self.channel_plots[label] = plot
                self.paned.add(plot.frame, stretch="always")
                self.channel_autoscale_override.setdefault(label, self.autoscale.get())
            else:
                frame = self.channel_plots[label].frame
                if str(frame) not in self.paned.panes():
                    self.paned.add(frame, stretch="always")

    def _clear_channel_plots(self) -> None:
        for plot in self.channel_plots.values():
            self.paned.forget(plot.frame)
            plot.destroy()
        self.channel_plots.clear()

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

        use_auto = self.channel_autoscale_override.get(label, self.autoscale.get())
        if use_auto:
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

    def adjust_manual_scale(self, label: str, factor: float) -> None:
        if label not in self.channel_scales:
            return
        value = max(1e-6, float(self.channel_scales[label].get()))
        value *= factor
        self.channel_scales[label].set(value)
        self.channel_autoscale_override[label] = False
        if self.channel_dialog is not None and tk.Toplevel.winfo_exists(self.channel_dialog):
            self.channel_dialog.update_scale_state_for(label)
        self._update_plot()

    def set_channel_autoscale(self, label: str, enabled: bool) -> None:
        self.channel_autoscale_override[label] = enabled
        if self.channel_dialog is not None and tk.Toplevel.winfo_exists(self.channel_dialog):
            self.channel_dialog.update_scale_state_for(label)
        self._update_plot()

    def update_hover(self, label: str, time_value: float, sample_value: float) -> None:
        timestamp = _format_seconds_hms_ms(time_value)
        self.hover_label.set(f"{label}: {timestamp} | {sample_value:.4f}")

    def clear_hover(self) -> None:
        self.hover_label.set("")


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
                state="disabled"
                if self.viewer.channel_autoscale_override.get(label, True)
                else "normal",
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
        self.update_scale_state()

    def _apply(self) -> None:
        selected = [label for label, var in self.channel_vars.items() if var.get()]
        if not selected:
            messagebox.showwarning("No channels", "Please select at least one channel to display.")
            return

        self.viewer.selected_channels = selected
        self.viewer._update_plot()
        self._on_close()

    def update_scale_state(self) -> None:
        for label, entry in self.scale_entries.items():
            autoscale = self.viewer.channel_autoscale_override.get(
                label, self.viewer.autoscale.get()
            )
            entry.configure(state="disabled" if autoscale else "normal")

    def update_scale_state_for(self, label: str) -> None:
        if label not in self.scale_entries:
            return
        autoscale = self.viewer.channel_autoscale_override.get(label, self.viewer.autoscale.get())
        self.scale_entries[label].configure(state="disabled" if autoscale else "normal")

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

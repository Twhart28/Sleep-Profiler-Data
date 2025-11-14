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
from matplotlib.figure import Figure


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
        self.selected_channel = tk.StringVar(value=edf_file.signal_labels[0])

        self._build_layout()
        self._populate_channels()
        self._update_plot()

    # ------------------------------------------------------------------
    # UI construction helpers
    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self, padding=10)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(1, weight=1)

        ttk.Label(sidebar, text="Channels", font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, sticky="w"
        )

        self.channel_list = tk.Listbox(
            sidebar,
            listvariable=tk.StringVar(value=self.edf_file.signal_labels),
            selectmode=tk.SINGLE,
            exportselection=False,
            height=20,
        )
        self.channel_list.grid(row=1, column=0, sticky="nsew", pady=(8, 16))
        self.channel_list.bind("<<ListboxSelect>>", self._on_channel_select)

        controls = ttk.Frame(sidebar)
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Window (s):").grid(row=0, column=0, sticky="w")
        window_entry = ttk.Spinbox(
            controls,
            from_=5,
            to=120,
            increment=5,
            textvariable=self.window_duration,
            width=6,
        )
        window_entry.grid(row=0, column=1, sticky="ew")
        window_entry.bind("<FocusOut>", lambda _event: self._update_plot())
        window_entry.bind("<Return>", lambda _event: self._update_plot())

        self.time_scale = ttk.Scale(
            controls,
            from_=0.0,
            to=self.edf_file.duration_seconds,
            variable=self.start_time,
            command=self._on_time_change,
        )
        self.time_scale.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(16, 0))

        self.time_label = ttk.Label(controls, text="Start: 0.0 s")
        self.time_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        info = ttk.Label(
            sidebar,
            text=(
                f"Duration: {self.edf_file.duration_seconds:.1f} s\n"
                f"Channels: {len(self.edf_file.signal_labels)}"
            ),
            justify="left",
        )
        info.grid(row=3, column=0, sticky="w", pady=(16, 0))

        # Plotting area
        plot_frame = ttk.Frame(self, padding=10)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(8, 6), constrained_layout=True)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_xlabel("Time (s)")
        self.axes.set_ylabel("Amplitude")

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _populate_channels(self) -> None:
        if not self.edf_file.signal_labels:
            messagebox.showerror("Empty EDF", "The selected EDF file has no channels.")
            self.destroy()
            return

        first_label = self.edf_file.signal_labels[0]
        self.channel_list.selection_set(0)
        self.channel_list.activate(0)
        self.selected_channel.set(first_label)

    # ------------------------------------------------------------------
    # Event handlers
    def _on_channel_select(self, _event: tk.Event[object]) -> None:
        selection = self.channel_list.curselection()
        if not selection:
            return
        index = selection[0]
        label = self.edf_file.signal_labels[index]
        self.selected_channel.set(label)
        self._update_plot()

    def _on_time_change(self, _value: str) -> None:
        self.time_label.configure(text=f"Start: {self.start_time.get():.1f} s")
        self._update_plot()

    # ------------------------------------------------------------------
    # Plotting helpers
    def _update_plot(self) -> None:
        label = self.selected_channel.get()
        try:
            signal, sample_rate = self.edf_file.get_signal(label)
        except KeyError:
            return

        window_seconds = max(1.0, float(self.window_duration.get()))
        start_seconds = float(self.start_time.get())
        max_start = max(0.0, self.edf_file.duration_seconds - window_seconds)
        start_seconds = min(start_seconds, max_start)
        self.start_time.set(start_seconds)
        self.time_label.configure(text=f"Start: {start_seconds:.1f} s")

        start_index = int(start_seconds * sample_rate)
        end_index = int((start_seconds + window_seconds) * sample_rate)
        end_index = min(len(signal), end_index)

        time_axis = np.linspace(
            start_index / sample_rate,
            (end_index - 1) / sample_rate,
            end_index - start_index,
        )
        segment = signal[start_index:end_index]

        self.axes.clear()
        self.axes.plot(time_axis, segment, linewidth=0.8)
        self.axes.set_title(f"{label} (fs={sample_rate:.1f} Hz)")
        self.axes.set_xlabel("Time (s)")
        self.axes.set_ylabel("Amplitude")
        self.axes.grid(True, which="both", linestyle="--", linewidth=0.3)
        self.canvas.draw_idle()


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

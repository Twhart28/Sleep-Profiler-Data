# Sleep-Profiler-Data

Tool used for viewing and editing raw EDF data provided by sleep profiler software from PSG sleep data.

## Sleep EDF Viewer

A lightweight Tkinter and Matplotlib based viewer is provided in
`sleep_edf_viewer.py`. When launched it immediately prompts for an EDF file and
then displays an interactive plot of the channels so you can browse sleep data.

### Features

- File picker dialog at startup (or pass a file path on the command line).
- Channel list for quickly switching between signals.
- Adjustable time window slider to navigate through the recording.
- Cached signal loading for responsive plotting of large recordings.

### Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

```bash
python sleep_edf_viewer.py
```

You may also pass an EDF path directly:

```bash
python sleep_edf_viewer.py /path/to/file.edf
```

Once the GUI opens, select a channel in the list to update the plot. Use the
slider to move through time and adjust the window size to focus on specific
segments of the sleep study.

"""Interactive Streamlit front end for the eight-channel EEG simulator."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eeg_simulator.dac_model import VirtualDAC
from eeg_simulator.device.mock_device import MockSimulatorDevice
from eeg_simulator.fault_conditions import ContactState, InterferenceMode
from web_demo.demo_logic import build_demo_waveform, time_to_sample_index


st.set_page_config(
    page_title="8-Channel EEG Simulator",
    page_icon="🧠",
    layout="wide",
)

st.title("8-Channel EEG Signal Simulator")
st.caption(
    "Interactive engineering demo: deterministic synthetic EEG, channel conditions, "
    "virtual 16-bit DAC mapping and the in-process mock-device protocol."
)
st.info(
    "This is a software engineering demonstration, not a medical device or clinical EEG tool. "
    "No physical DAC output, MCU firmware or binary waveform transfer is claimed here."
)

with st.sidebar:
    st.header("Signal controls")
    amplitude_scale = st.select_slider(
        "Amplitude scale",
        options=(0.25, 0.5, 1.0, 2.0),
        value=1.0,
    )
    contact_channel = st.selectbox(
        "Channel condition applies to",
        ("F3", "F4", "C3", "C4", "T3", "T4", "O1", "O2"),
        index=0,
    )
    contact_state = st.selectbox(
        "Contact condition",
        list(ContactState),
        index=0,
        format_func=lambda value: value.value,
    )
    interference_mode = st.selectbox(
        "Global interference",
        list(InterferenceMode),
        index=0,
        format_func=lambda value: value.value,
    )
    st.caption("Default source: 60 s × 256 Hz deterministic synthetic engineering data.")

demo = build_demo_waveform(
    amplitude_scale=amplitude_scale,
    contact_channel=contact_channel,
    contact_state=contact_state,
    interference_mode=interference_mode,
)

default_channels = list(demo.channel_names)
display_channels = st.multiselect(
    "Displayed channels",
    demo.channel_names,
    default=default_channels,
)
if not display_channels:
    st.warning("Select at least one channel to display.")
    st.stop()

window_seconds = st.select_slider(
    "Visible window",
    options=(5.0, 10.0, 15.0, 20.0),
    value=10.0,
    format_func=lambda value: f"{value:g} s",
)
latest_start = max(0.0, demo.duration_seconds - window_seconds)
default_start = min(18.0, latest_start)
window_start = st.slider(
    "Window start",
    min_value=0.0,
    max_value=float(latest_start),
    value=float(default_start),
    step=0.25,
    format="%.2f s",
)
window_end = min(demo.duration_seconds, window_start + window_seconds)
cursor_seconds = st.slider(
    "Sample inspector",
    min_value=float(window_start),
    max_value=float(window_end),
    value=float(window_start + (window_end - window_start) / 2.0),
    step=1.0 / demo.sample_rate_hz,
    format="%.3f s",
)

sample_start = time_to_sample_index(
    window_start,
    demo.sample_rate_hz,
    demo.samples_uv.shape[1],
)
sample_end = min(
    demo.samples_uv.shape[1],
    time_to_sample_index(
        window_end,
        demo.sample_rate_hz,
        demo.samples_uv.shape[1],
    )
    + 1,
)
time_axis = np.arange(sample_start, sample_end, dtype=np.float64) / demo.sample_rate_hz

figure = make_subplots(
    rows=len(display_channels),
    cols=1,
    shared_xaxes=True,
    vertical_spacing=min(0.03, 0.18 / max(len(display_channels), 1)),
    subplot_titles=display_channels,
)
for row, channel in enumerate(display_channels, start=1):
    channel_index = demo.channel_names.index(channel)
    figure.add_trace(
        go.Scatter(
            x=time_axis,
            y=demo.samples_uv[channel_index, sample_start:sample_end],
            mode="lines",
            name=channel,
            showlegend=False,
            hovertemplate="t=%{x:.3f} s<br>%{y:.2f} µV<extra></extra>",
        ),
        row=row,
        col=1,
    )
    figure.update_yaxes(title_text="µV", row=row, col=1)

figure.update_xaxes(title_text="Time (s)", row=len(display_channels), col=1)
figure.update_layout(
    height=max(430, 115 * len(display_channels)),
    margin=dict(l=45, r=25, t=45, b=35),
)
figure.add_vline(x=cursor_seconds, line_dash="dot")
st.plotly_chart(figure, width="stretch")

for onset, duration, description in demo.annotations:
    if onset < window_end and onset + duration > window_start:
        st.caption(
            f"Visible annotation: {description} ({onset:g}–{onset + duration:g} s)."
        )

if demo.abnormal_summary:
    st.warning("Active software condition: " + " · ".join(demo.abnormal_summary))
else:
    st.success("Active software condition: normal waveform.")

st.subheader("Virtual DAC sample inspector")
inspect_channel = st.selectbox("Inspect channel", display_channels)
inspect_channel_index = demo.channel_names.index(inspect_channel)
inspect_sample_index = time_to_sample_index(
    cursor_seconds,
    demo.sample_rate_hz,
    demo.samples_uv.shape[1],
)
sample_uv = float(demo.samples_uv[inspect_channel_index, inspect_sample_index])
dac_code = int(demo.dac_codes[inspect_channel_index, inspect_sample_index])
dac = VirtualDAC()
normalised = float(np.clip(sample_uv / dac.full_scale_uv, -1.0, 1.0))
is_clipped = abs(sample_uv) > dac.full_scale_uv

m1, m2, m3, m4 = st.columns(4)
m1.metric("Target sample", f"{sample_uv:.2f} µV")
m2.metric("Normalised full scale", f"{normalised:+.4f}")
m3.metric("16-bit virtual DAC code", f"{dac_code}")
m4.metric("Digital clipping", "YES" if is_clipped else "NO")

st.caption(
    "The DAC code is the repository's software mapping over a configurable ±200 µV "
    "digital target range. It is not a measured analogue voltage."
)

st.subheader("Mock device sequence")
st.write(
    "Run the same in-process device abstraction used by the desktop application. "
    "The sequence connects, configures the selected channels, uploads the prepared "
    "software waveform, starts, pauses and stops."
)

if st.button("Run mock-device sequence", type="primary"):
    logs: list[str] = []
    device = MockSimulatorDevice(log_callback=logs.append)
    device.connect()
    device.configure(
        sample_rate_hz=demo.sample_rate_hz,
        amplitude_scale=amplitude_scale,
        channels=tuple(display_channels),
        loop=True,
    )
    indices = [demo.channel_names.index(channel) for channel in display_channels]
    device.upload_waveform(demo.samples_uv[indices])
    device.start()
    device.pause()
    device.stop()
    st.session_state["mock_device_log"] = "\n".join(logs)
    st.session_state["mock_device_state"] = device.get_status().state.value

if "mock_device_log" in st.session_state:
    st.code(st.session_state["mock_device_log"], language="text")
    st.caption(
        "Final mock state: "
        + st.session_state["mock_device_state"]
        + ". This is an in-process simulation, not a physical serial connection."
    )

st.subheader("System boundary")
st.code(
    """Synthetic EEG / selected conditions
        ↓
Browser demo + shared Python signal logic
        ↓
Virtual 16-bit DAC mapping
        ↓
MockSimulatorDevice                 [implemented]
        ↓
Serial adapter command path         [partial]
        ↓
Binary waveform transfer / MCU      [future]
        ↓
Physical DAC + analogue scaling     [future]
        ↓
External EEG acquisition hardware   [external]""",
    language="text",
)

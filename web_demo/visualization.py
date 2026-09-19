"""Plotly rendering helpers for the browser console."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from eeg_simulator.fault_conditions import ContactState
from web_demo.demo_logic import DemoWaveform


CHANNEL_COLOURS = {
    ContactState.NORMAL: "#67d4ff",
    ContactState.MODERATE: "#a6d8d0",
    ContactState.HIGH: "#ffc46b",
    ContactState.VERY_HIGH: "#ffad5c",
    ContactState.LEAD_OFF: "#ff8d8d",
}


def nice_axis_limit(values: np.ndarray) -> float:
    peak = float(np.max(np.abs(values)))
    if peak <= 0:
        return 1.0
    magnitude = 10.0 ** math.floor(math.log10(peak))
    normalised = peak / magnitude
    if normalised <= 1.0:
        factor = 1.0
    elif normalised <= 2.0:
        factor = 2.0
    elif normalised <= 5.0:
        factor = 5.0
    else:
        factor = 10.0
    return factor * magnitude


def build_monitor_figure(
    demo: DemoWaveform,
    active_channels: Sequence[str],
    contact_states: Mapping[str, ContactState],
    cursor_seconds: float,
    render_hz: float = 64.0,
) -> go.Figure:
    """Build a compact fixed eight-lane monitor with event shading and cursor."""

    active = set(active_channels)
    stride = max(1, int(round(demo.sample_rate_hz / render_hz)))
    time_axis = (
        np.arange(demo.samples_uv.shape[1], dtype=np.float64) / demo.sample_rate_hz
    )[::stride]

    figure = make_subplots(
        rows=len(demo.channel_names),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.012,
    )

    for row, channel in enumerate(demo.channel_names, start=1):
        channel_index = demo.channel_names.index(channel)
        values = demo.samples_uv[channel_index]
        state = contact_states.get(channel, ContactState.NORMAL)
        colour = CHANNEL_COLOURS[state]
        dash = "dash" if state is ContactState.LEAD_OFF else "solid"

        figure.add_trace(
            go.Scattergl(
                x=time_axis,
                y=values[::stride],
                mode="lines",
                line={"color": colour, "width": 1.2, "dash": dash},
                name=channel,
                showlegend=False,
                visible=channel in active,
                hovertemplate=(
                    channel + "  %{x:.3f} s<br>%{y:.2f} µV<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )

        axis_limit = nice_axis_limit(values)
        figure.update_yaxes(
            range=[-axis_limit * 1.12, axis_limit * 1.12],
            tickvals=[0],
            ticktext=[f"±{axis_limit:g}"],
            title_text=channel,
            title_font={"color": colour, "size": 12},
            tickfont={"color": "#6f8796", "size": 9},
            gridcolor="rgba(150,180,195,0.10)",
            zeroline=True,
            zerolinecolor="rgba(150,180,195,0.14)",
            fixedrange=True,
            row=row,
            col=1,
        )
        figure.update_xaxes(
            showgrid=False,
            showticklabels=row == len(demo.channel_names),
            tickfont={"color": "#8ba1ae", "size": 9},
            row=row,
            col=1,
        )

        for onset, duration, _description in demo.annotations:
            figure.add_vrect(
                x0=onset,
                x1=onset + duration,
                fillcolor="rgba(255,190,90,0.10)",
                line_width=0,
                row=row,
                col=1,
            )
        figure.add_vline(
            x=cursor_seconds,
            line_width=1.1,
            line_color="#ffd166",
            row=row,
            col=1,
        )

    figure.update_xaxes(
        title_text="Time (s)",
        title_font={"color": "#9db3bf", "size": 10},
        row=len(demo.channel_names),
        col=1,
    )
    figure.update_layout(
        height=610,
        margin={"l": 58, "r": 16, "t": 10, "b": 36},
        paper_bgcolor="#0d1722",
        plot_bgcolor="#0d1722",
        font={"color": "#dce6ee"},
        hovermode="x unified",
        dragmode="zoom",
        uirevision="eight-channel-monitor-v2",
    )
    return figure

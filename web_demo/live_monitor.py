"""Flicker-free browser-side SVG monitor for the Streamlit console."""

from __future__ import annotations

from html import escape
import json
import math
from typing import Mapping, Sequence

import numpy as np

from eeg_simulator.fault_conditions import ContactState
from web_demo.console_logic import DemoWaveform


MONITOR_SVG_HEIGHT_PX = 500
MONITOR_IFRAME_HEIGHT_PX = 575


CHANNEL_COLOURS = {
    ContactState.NORMAL: "#67d4ff",
    ContactState.MODERATE: "#a6d8d0",
    ContactState.HIGH: "#ffc46b",
    ContactState.VERY_HIGH: "#ffad5c",
    ContactState.LEAD_OFF: "#ff8d8d",
}


def nice_axis_limit(values: np.ndarray) -> float:
    """Return a compact symmetric display range for one waveform lane."""

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


def _svg_path(
    values: np.ndarray,
    sample_rate_hz: float,
    duration_seconds: float,
    baseline_y: float,
    amplitude_px: float,
    axis_limit: float,
    left: float,
    plot_width: float,
    render_hz: float,
) -> str:
    stride = max(1, int(round(sample_rate_hz / render_hz)))
    indices = np.arange(0, values.size, stride, dtype=np.int64)
    if indices[-1] != values.size - 1:
        indices = np.append(indices, values.size - 1)
    times = indices / sample_rate_hz
    xs = left + (times / duration_seconds) * plot_width
    ys = baseline_y - (values[indices] / axis_limit) * amplitude_px
    return " ".join(
        ("M" if i == 0 else "L") + f"{x:.2f},{y:.2f}"
        for i, (x, y) in enumerate(zip(xs, ys, strict=True))
    )


def build_live_monitor_html(
    demo: DemoWaveform,
    active_channels: Sequence[str],
    contact_states: Mapping[str, ContactState],
    *,
    initial_position_seconds: float,
    running: bool,
    loop: bool,
    state_label: str,
    render_hz: float = 48.0,
) -> str:
    """Render the EEG once and animate only the cursor in the browser.

    Waveform geometry is static for the lifetime of the iframe. JavaScript moves
    the playback cursor, event indicator and progress bar using
    requestAnimationFrame, avoiding Streamlit/Plotly re-creation during playback.
    """

    width = 1200.0
    height = 560.0
    left = 88.0
    right = 18.0
    top = 18.0
    bottom = 44.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    lane_height = plot_height / len(demo.channel_names)
    amplitude_px = lane_height * 0.42
    active = set(active_channels)

    ticks = []
    tick_step = 10.0
    tick = 0.0
    while tick <= demo.duration_seconds + 1e-9:
        x = left + (tick / demo.duration_seconds) * plot_width
        ticks.append(
            f'<line x1="{x:.2f}" y1="{top:.2f}" x2="{x:.2f}" y2="{top + plot_height:.2f}" '
            'stroke="rgba(145,170,185,0.08)" stroke-width="1"/>'
        )
        ticks.append(
            f'<text x="{x:.2f}" y="{height - 20:.2f}" text-anchor="middle" '
            'fill="#8298a6" font-size="11">' + f"{tick:g}" + "</text>"
        )
        tick += tick_step

    events = []
    for onset, duration, description in demo.annotations:
        x = left + (onset / demo.duration_seconds) * plot_width
        event_width = (duration / demo.duration_seconds) * plot_width
        events.append(
            f'<rect x="{x:.2f}" y="{top:.2f}" width="{event_width:.2f}" '
            f'height="{plot_height:.2f}" rx="2" fill="rgba(255,190,90,0.09)">'
            f'<title>{escape(description)}</title></rect>'
        )

    lanes = []
    for index, channel in enumerate(demo.channel_names):
        baseline = top + (index + 0.5) * lane_height
        values = demo.samples_uv[index]
        axis_limit = nice_axis_limit(values)
        state = contact_states.get(channel, ContactState.NORMAL)
        colour = CHANNEL_COLOURS[state]
        visible = channel in active
        opacity = "1" if visible else "0.16"
        dash = ' stroke-dasharray="5 4"' if state is ContactState.LEAD_OFF else ""
        path = _svg_path(
            values,
            demo.sample_rate_hz,
            demo.duration_seconds,
            baseline,
            amplitude_px,
            axis_limit,
            left,
            plot_width,
            render_hz,
        )
        lanes.append(
            f'<line x1="{left:.2f}" y1="{baseline:.2f}" x2="{left + plot_width:.2f}" '
            f'y2="{baseline:.2f}" stroke="rgba(145,170,185,0.10)" stroke-width="1"/>'
        )
        lanes.append(
            f'<text x="10" y="{baseline + 4:.2f}" fill="{colour}" font-size="14" '
            f'font-weight="700">{escape(channel)}</text>'
        )
        lanes.append(
            f'<text x="48" y="{baseline + 4:.2f}" fill="#718896" font-size="11">'
            f'±{axis_limit:g}</text>'
        )
        lanes.append(
            f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="1.65" '
            f'opacity="{opacity}" vector-effect="non-scaling-stroke"{dash}/>'
        )

    event_ranges = [
        [float(onset), float(onset + duration), description]
        for onset, duration, description in demo.annotations
    ]
    event_json = json.dumps(event_ranges, ensure_ascii=False)
    start_position = min(max(float(initial_position_seconds), 0.0), demo.duration_seconds)
    running_js = "true" if running else "false"
    loop_js = "true" if loop else "false"
    state_text = escape(state_label)

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
html, body {{ margin:0; padding:0; background:#0d1722; color:#dce6ee; overflow:hidden; }}
* {{ box-sizing:border-box; }}
.shell {{ width:100%; background:#0d1722; font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif; }}
.top {{ display:flex; justify-content:space-between; align-items:center; padding:4px 6px 8px; }}
.title {{ font-size:12px; font-weight:800; letter-spacing:.08em; color:#b9cfda; }}
.pill {{ padding:3px 9px; border-radius:999px; border:1px solid #415766; background:#15222d;
         color:#b7c8d2; font-size:11px; font-weight:800; }}
.event {{ min-height:20px; margin:0 4px 6px; padding:4px 8px; border-radius:5px;
          border:1px solid transparent; font-size:11px; font-weight:700; color:transparent; }}
.event.active {{ border-color:#6f5934; background:#32291c; color:#ffd27d; }}
svg {{ width:100%; height:{MONITOR_SVG_HEIGHT_PX}px; display:block; background:#0d1722; }}
.bottom {{ display:flex; align-items:center; gap:10px; padding:4px 8px 0; color:#b7d8e8;
           font-size:11px; font-weight:700; }}
.track {{ flex:1; height:5px; background:#1b2b36; border-radius:999px; overflow:hidden; }}
.fill {{ height:100%; width:0; background:#4fa7c9; }}
</style>
</head>
<body>
<div class="shell">
  <div class="top">
    <div class="title">WAVEFORM MONITOR · SIMULATED EEG OUTPUT (µV)</div>
    <div class="pill">{state_text}</div>
  </div>
  <div id="eventBadge" class="event">SYNTHETIC DEMO EVENT</div>
  <svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="Eight-channel EEG waveform monitor">
    <rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="#0d1722"/>
    {''.join(ticks)}
    {''.join(events)}
    {''.join(lanes)}
    <line id="cursor" x1="{left:.2f}" y1="{top:.2f}" x2="{left:.2f}" y2="{top + plot_height:.2f}"
          stroke="#ffd166" stroke-width="1.7" vector-effect="non-scaling-stroke"/>
    <text x="{left + plot_width / 2:.2f}" y="{height - 3:.2f}" text-anchor="middle"
          fill="#9db3bf" font-size="11">Time (s)</text>
  </svg>
  <div class="bottom">
    <span id="timeLabel">Position 00:00.000 / 01:00.000</span>
    <div class="track"><div id="progressFill" class="fill"></div></div>
  </div>
</div>
<script>
(() => {{
  const duration = {demo.duration_seconds:.9f};
  const left = {left:.9f};
  const plotWidth = {plot_width:.9f};
  const y1 = {top:.9f};
  const y2 = {top + plot_height:.9f};
  const loop = {loop_js};
  let running = {running_js};
  const initial = {start_position:.9f};
  const startedAt = performance.now();
  const cursor = document.getElementById("cursor");
  const fill = document.getElementById("progressFill");
  const label = document.getElementById("timeLabel");
  const badge = document.getElementById("eventBadge");
  const events = {event_json};

  function clockText(seconds) {{
    const safe = Math.max(0, seconds);
    const minutes = Math.floor(safe / 60);
    const remaining = safe - minutes * 60;
    return String(minutes).padStart(2, "0") + ":" + remaining.toFixed(3).padStart(6, "0");
  }}

  function draw(now) {{
    let position = initial;
    if (running) {{
      position += (now - startedAt) / 1000;
      if (loop && duration > 0) {{
        position = position % duration;
      }} else if (position >= duration) {{
        position = duration;
        running = false;
      }}
    }}
    const ratio = duration > 0 ? Math.min(Math.max(position / duration, 0), 1) : 0;
    const x = left + ratio * plotWidth;
    cursor.setAttribute("x1", x.toFixed(2));
    cursor.setAttribute("x2", x.toFixed(2));
    cursor.setAttribute("y1", y1.toFixed(2));
    cursor.setAttribute("y2", y2.toFixed(2));
    fill.style.width = (ratio * 100).toFixed(3) + "%";
    label.textContent = "Position " + clockText(position) + " / " + clockText(duration);

    const currentEvent = events.find(([start, end]) => position >= start && position < end);
    if (currentEvent) {{
      badge.textContent = "SYNTHETIC DEMO EVENT · " + currentEvent[2] + " · engineering demonstration only";
      badge.classList.add("active");
    }} else {{
      badge.textContent = "SYNTHETIC DEMO EVENT";
      badge.classList.remove("active");
    }}

    if (running) requestAnimationFrame(draw);
  }}

  requestAnimationFrame(draw);
}})();
</script>
</body>
</html>"""

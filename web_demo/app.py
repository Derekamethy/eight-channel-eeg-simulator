"""Desktop-console-style Streamlit front end for the eight-channel EEG simulator."""

from __future__ import annotations

from html import escape
from pathlib import Path
import sys
from typing import Callable

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eeg_simulator.device.base import DeviceState
from eeg_simulator.fault_conditions import ContactState, InterferenceMode
from eeg_simulator.synthetic_eeg import ACTIVE_CHANNELS, generate_synthetic_eeg
from eeg_simulator.validation import run_validation_suite
from web_demo.controller import WebSimulatorController
from web_demo.console_logic import build_demo_waveform
from web_demo.live_monitor import MONITOR_IFRAME_HEIGHT_PX, build_live_monitor_html


st.set_page_config(
    page_title="8-Channel EEG Simulator",
    page_icon="EEG",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _inject_console_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b1219;
            --panel: #101a24;
            --panel2: #0d1722;
            --border: #263946;
            --text: #dce6ee;
            --muted: #8fa5b3;
            --cyan: #67d4ff;
            --green: #63d9a4;
            --amber: #ffd166;
            --red: #ff8d8d;
        }
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #0b1219 0%, #0d151d 100%);
            color: var(--text);
        }
        .block-container {
            max-width: 1780px;
            padding-top: 3.6rem;
            padding-left: 1rem; padding-right: 1rem;
            padding-bottom: 0.45rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(16,26,36,0.92);
            border-color: var(--border);
            border-radius: 8px;
        }
        div[data-testid="stHorizontalBlock"] { gap: 0.42rem; }
        div[data-testid="stVerticalBlock"] { gap: 0.42rem; }
        div[data-baseweb="select"] > div { min-height: 34px; height: 34px; }
        div[data-testid="stSelectbox"] { margin-bottom: 0; }
        div[data-testid="stCheckbox"] { margin: 0; padding-top: 0.08rem; }
        .stButton > button { min-height: 34px; padding: 0.22rem 0.48rem; }
        .compact-source {
            margin-top: 5px; padding-top: 6px; border-top: 1px solid #263946;
            color: #8fa5b3; font-size: 0.73rem; line-height: 1.35;
        }
        .compact-source b { color:#dce6ee; font-weight:700; }
        .console-header {
            display:flex; align-items:center; justify-content:flex-start; flex-wrap:wrap;
            gap:12px; margin:0 0 6px 0;
        }
        .console-title {
            font-size:1.72rem; line-height:1.0; font-weight:800; color:#f2f8fb;
            letter-spacing:-0.02em;
        }
        .console-subtitle { display:none; }
        .mode-badge {
            border:1px solid #3d6277; background:#1a2d3a; color:#78dbff;
            border-radius:6px; padding:7px 11px; font-weight:800; font-size:0.78rem;
            white-space:nowrap;
        }
        .section-label {
            color:#a7c2d0; font-size:0.74rem; font-weight:800; letter-spacing:0.09em;
            margin-bottom:6px;
        }
        .device-line {
            display:flex; justify-content:space-between; align-items:center;
            margin-bottom:5px;
        }
        .device-name { font-weight:800; color:#eef6fa; }
        .state-pill {
            display:inline-block; padding:3px 8px; border-radius:999px;
            font-size:0.72rem; font-weight:800; border:1px solid #415766;
            background:#15222d; color:#b7c8d2;
        }
        .state-running { color:#7de3b8; border-color:#347b65; background:#12352f; }
        .state-paused { color:#ffd27d; border-color:#7c6236; background:#362d1f; }
        .state-disconnected { color:#ffaaaa; border-color:#7e4249; background:#382126; }
        .source-grid {
            display:grid; grid-template-columns:1fr 1fr; gap:5px 10px;
            font-size:0.82rem;
        }
        .source-grid span { color:var(--muted); }
        .source-grid b { color:#dce6ee; font-weight:700; }
        .monitor-toolbar {
            display:flex; justify-content:space-between; align-items:center;
            gap:12px; margin:0 0 6px 0;
        }
        .monitor-title {
            font-size:0.82rem; font-weight:800; letter-spacing:0.08em; color:#b9cfda;
        }
        .event-banner {
            border:1px solid #6f5934; background:#32291c; color:#ffd27d;
            border-radius:5px; padding:5px 8px; font-size:0.78rem; font-weight:700;
            margin-bottom:5px;
        }
        .condition-banner {
            border:1px solid #665b78; background:#282633; color:#d8c9f0;
            border-radius:5px; padding:5px 8px; font-size:0.76rem; font-weight:700;
            margin-bottom:5px;
        }
        .position-strip {
            display:flex; align-items:center; gap:9px; margin-top:-4px;
            color:#b7d8e8; font-size:0.78rem; font-weight:700;
        }
        .position-track {
            height:5px; flex:1; background:#1b2b36; border-radius:99px; overflow:hidden;
        }
        .position-fill { height:100%; background:#4fa7c9; }
        .dac-strip {
            display:grid; grid-template-columns:1fr;
            gap:5px; margin-top:4px;
        }
        .dac-chip {
            display:grid; grid-template-columns:22px minmax(0, 1fr) 42px; align-items:center;
            gap:2px 5px; padding:6px 7px; border:1px solid #2a3e4b;
            background:#0d1722; border-radius:5px; min-width:0;
            font-size:0.70rem; font-variant-numeric:tabular-nums;
        }
        .dac-chip b { color:#7ad7ff; }
        .dac-chip span { color:#c9d7de; text-align:right; white-space:nowrap; }
        .dac-chip code { color:#91a9b7; font-family:Consolas,monospace; text-align:right; }
        .dac-chip.inactive { opacity:0.38; }
        .log-box {
            max-height:190px; overflow:auto; background:#091119; border:1px solid #233541;
            border-radius:5px; padding:8px; color:#b9d3df; font-size:0.72rem;
            line-height:1.45; font-family:Consolas, "SFMono-Regular", monospace;
            white-space:pre-wrap;
        }
        .empty-log { color:#607887; }
        .boundary-note {
            color:#8399a6; font-size:0.72rem; padding-top:4px;
        }
        .chain {
            display:grid; grid-template-columns:1fr 1fr;
            gap:7px; align-items:stretch; margin-top:8px;
        }
        .chain-card {
            padding:9px 8px; border-radius:7px; border:1px solid #314653;
            background:#101c26; min-height:88px;
        }
        .chain-card b { display:block; font-size:0.76rem; color:#e4eef3; }
        .chain-card small { color:#8fa5b3; font-size:0.68rem; }
        .implemented { border-color:#3f796b; background:#112c27; }
        .partial { border-color:#80643a; background:#302719; }
        .future { border-color:#5c5368; background:#24212b; }
        .external { border-color:#41677d; background:#172731; }
        .validation-pass {
            border:1px solid #347b65; background:#12352f; color:#7de3b8;
            border-radius:6px; padding:8px 10px; font-weight:800; margin-bottom:8px;
        }
        .validation-card {
            border:1px solid #2f4552; background:#101b25; border-radius:6px;
            padding:8px 10px; font-size:0.78rem; min-height:96px;
        }
        .validation-card b { color:#e8f2f6; }
        .validation-card span { color:#9fb4bf; }
        .stButton > button {
            border-radius:5px; border:1px solid #38505e; background:#182732;
            color:#dce6ee; font-weight:700;
        }
        .stButton > button:hover { border-color:#5e859b; color:#ffffff; }
        div[data-testid="stSelectbox"] label,
        div[data-testid="stCheckbox"] label,
        div[data-testid="stSelectSlider"] label {
            font-size:0.78rem;
        }
        [data-testid="stCaptionContainer"] { color:#7f96a3; }
        [data-testid="stMarkdownContainer"]:has(.section-label, .device-line, .compact-source,
            .condition-banner, .console-header, .dac-strip, .validation-pass,
            .validation-card, .chain, .log-box) { margin-bottom:0; }
        /* Compact controls retain native keyboard and screen-reader support. */
        .st-key-controls [data-testid="stVerticalBlock"] { gap:2px; }
        .st-key-controls [data-testid="stLayoutWrapper"] { min-height:0; }
        .st-key-controls [data-testid="stSelectbox"] [role="group"] {
            min-height:26px; height:26px; font-size:13px;
        }
        .st-key-controls [role="combobox"] { padding:2px 8px; font-size:13px; }
        .st-key-controls [data-testid="stCheckbox"] label {
            min-height:26px; padding:0;
        }
        .st-key-controls div[data-testid="stCheckbox"] { padding:0; margin:0 !important; }
        .st-key-controls [data-testid="stVerticalBlockBorderWrapper"] > div,
        .st-key-controls [data-testid="stVerticalBlock"][data-test-scroll-behavior] {
            padding:8px;
        }
        .st-key-eeg [data-testid="stElementContainer"]:has(iframe) { flex-basis:auto; }
        .st-key-controls button { min-height:28px; padding:2px 5px; }
        .st-key-controls button p { font-size:13px; }
        .st-key-controls .section-label { margin-bottom:0; }
        .st-key-controls [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-controls [data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"] {
            border-color:var(--border);
        }
        .condition-banner { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .st-key-eeg iframe { height:calc(100dvh - 204px); min-height:400px; }
        .st-key-eeg [data-testid="stElementContainer"]:has(iframe) { height:calc(100dvh - 204px); min-height:400px; }
        .st-key-eeg:has(.condition-banner) iframe,
        .st-key-eeg:has(.condition-banner) [data-testid="stElementContainer"]:has(iframe) {
            height:calc(100dvh - 248px); min-height:364px;
        }
        @media (max-width: 1050px) {
            .chain { grid-template-columns:1fr 1fr; }
            .console-title { font-size:1.5rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_console_css()

if "controller" not in st.session_state:
    st.session_state.controller = WebSimulatorController()
if "control_error" not in st.session_state:
    st.session_state.control_error = ""
if "validation_result" not in st.session_state:
    st.session_state.validation_result = None
for channel in ACTIVE_CHANNELS:
    st.session_state.setdefault(f"active_{channel}", True)
    st.session_state.setdefault(f"condition_{channel}", ContactState.NORMAL)
st.session_state.setdefault("amplitude_scale", 1.0)
st.session_state.setdefault("loop_playback", False)
st.session_state.setdefault("interference_mode", InterferenceMode.NONE)

controller: WebSimulatorController = st.session_state.controller


def _safe_action(action: Callable[[], None]) -> None:
    try:
        action()
        st.session_state.control_error = ""
    except Exception as exc:
        st.session_state.control_error = str(exc)


def _active_channels() -> tuple[str, ...]:
    return tuple(
        channel
        for channel in ACTIVE_CHANNELS
        if st.session_state.get(f"active_{channel}", False)
    )


def _contact_states() -> dict[str, ContactState]:
    return {
        channel: st.session_state.get(
            f"condition_{channel}", ContactState.NORMAL
        )
        for channel in ACTIVE_CHANNELS
    }


def _configuration_signature() -> tuple[object, ...]:
    return (
        st.session_state.amplitude_scale,
        _active_channels(),
        st.session_state.loop_playback,
        tuple(
            st.session_state[f"condition_{channel}"].value
            for channel in ACTIVE_CHANNELS
        ),
        st.session_state.interference_mode.value,
    )


def _ensure_active_channel(changed_channel: str) -> None:
    if not _active_channels():
        st.session_state[f"active_{changed_channel}"] = True
        st.session_state.control_error = "At least one channel must remain active."


def _reset_conditions() -> None:
    for channel in ACTIVE_CHANNELS:
        st.session_state[f"condition_{channel}"] = ContactState.NORMAL
    st.session_state.interference_mode = InterferenceMode.NONE
    controller.append_app_log("Electrode and signal conditions reset to normal")


def _start_playback() -> None:
    demo_now = build_demo_waveform(
        amplitude_scale=st.session_state.amplitude_scale,
        contact_states=_contact_states(),
        interference_mode=st.session_state.interference_mode,
    )
    active = _active_channels()
    indices = [demo_now.channel_names.index(channel) for channel in active]
    controller.start(
        samples_uv=demo_now.samples_uv[indices],
        sample_rate_hz=demo_now.sample_rate_hz,
        amplitude_scale=st.session_state.amplitude_scale,
        channels=active,
        loop=st.session_state.loop_playback,
        signature=_configuration_signature(),
    )


def _run_validation() -> None:
    st.session_state.validation_result = run_validation_suite(
        generate_synthetic_eeg()
    )
    controller.append_app_log("Three-level software-reference validation completed")


st.markdown(
    """
    <div class="console-header">
      <div>
        <div class="console-title">8-Channel EEG Simulator</div>
        <div class="console-subtitle"></div>
      </div>
      <div class="mode-badge">MOCK · NOT CLINICAL</div>
    </div>
    """,
    unsafe_allow_html=True,
)

status = controller.status
running = status.state is DeviceState.RUNNING
editing_disabled = running

left, right = st.columns([0.23, 0.77], gap="medium")

with left, st.container(key="controls"):
    with st.container(border=True):
        if not status.connected:
            state_class = "state-disconnected"
        elif status.state is DeviceState.RUNNING:
            state_class = "state-running"
        elif status.state is DeviceState.PAUSED:
            state_class = "state-paused"
        else:
            state_class = ""

        st.markdown(
            f"""
            <div class="device-line">
              <span class="device-name">Mock EEG</span>
              <span class="state-pill {state_class}">{status.state.value}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        c1.button(
            "Connect",
            key="connect_btn",
            width="stretch",
            disabled=status.connected,
            on_click=_safe_action,
            args=(controller.connect,),
        )
        c2.button(
            "Disconnect",
            key="disconnect_btn",
            width="stretch",
            disabled=not status.connected,
            on_click=_safe_action,
            args=(controller.disconnect,),
        )

        cfg1, cfg2 = st.columns([0.58, 0.42])
        cfg1.selectbox(
            "Amplitude",
            options=(0.25, 0.5, 1.0, 2.0),
            key="amplitude_scale",
            disabled=editing_disabled,
            format_func=lambda value: f"{value:g}×",
            label_visibility="collapsed",
            help="Amplitude scale",
        )
        cfg2.checkbox(
            "Loop",
            key="loop_playback",
            disabled=editing_disabled,
        )

        p1, p2, p3, p4 = st.columns(4)
        p1.button(
            "▶",
            key="start_btn",
            help="Start",
            type="primary",
            width="stretch",
            disabled=not status.connected or status.state is DeviceState.RUNNING,
            on_click=_safe_action,
            args=(_start_playback,),
        )
        p2.button(
            "Ⅱ",
            key="pause_btn",
            help="Pause",
            width="stretch",
            disabled=status.state is not DeviceState.RUNNING,
            on_click=_safe_action,
            args=(controller.pause,),
        )
        p3.button(
            "■",
            key="stop_btn",
            help="Stop",
            width="stretch",
            disabled=status.state not in (DeviceState.RUNNING, DeviceState.PAUSED),
            on_click=_safe_action,
            args=(controller.stop,),
        )
        p4.button(
            "↺",
            key="reset_btn",
            help="Reset",
            width="stretch",
            disabled=not status.connected,
            on_click=_safe_action,
            args=(controller.reset,),
        )

        st.markdown(
            """
            <div class="compact-source">
              <b>Synthetic</b> · 256 Hz · 60 s · 8 ch · 15,360/ch
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        title_col, reset_col = st.columns([0.74, 0.26])
        title_col.markdown(
            '<div class="section-label">CHANNELS / CONDITIONS</div>',
            unsafe_allow_html=True,
        )
        reset_col.button(
            "Reset",
            key="reset_conditions_btn",
            width="stretch",
            disabled=editing_disabled,
            on_click=_reset_conditions,
        )

        for channel in ACTIVE_CHANNELS:
            row = st.columns([0.30, 0.70])
            row[0].checkbox(
                channel,
                key=f"active_{channel}",
                disabled=editing_disabled,
                on_change=_ensure_active_channel,
                args=(channel,),
            )
            row[1].selectbox(
                f"{channel} condition",
                list(ContactState),
                key=f"condition_{channel}",
                disabled=editing_disabled,
                format_func=lambda value: {
                    ContactState.NORMAL: "Normal",
                    ContactState.MODERATE: "20 kΩ",
                    ContactState.HIGH: "50 kΩ",
                    ContactState.VERY_HIGH: "100 kΩ",
                    ContactState.LEAD_OFF: "Off",
                }[value],
                label_visibility="collapsed",
            )

        artifact = st.columns([0.30, 0.70])
        artifact[0].markdown(
            '<div style="padding-top:8px;color:#9fb4bf;font-size:0.74rem;font-weight:700">ARTEFACT</div>',
            unsafe_allow_html=True,
        )
        artifact[1].selectbox(
            "Signal artefact",
            list(InterferenceMode),
            key="interference_mode",
            disabled=editing_disabled,
            format_func=lambda value: value.value,
            label_visibility="collapsed",
        )

    if st.session_state.control_error:
        st.toast(st.session_state.control_error, icon="⚠️")

demo = build_demo_waveform(
    amplitude_scale=st.session_state.amplitude_scale,
    contact_states=_contact_states(),
    interference_mode=st.session_state.interference_mode,
)
active_channels = _active_channels()
contact_states = _contact_states()
monitor_snapshot = controller.snapshot()
monitor_running = controller.status.state is DeviceState.RUNNING
telemetry_every = 0.5 if monitor_running else None

with right:
    monitor_col, dac_col = st.columns([0.79, 0.21], gap="small")

    with monitor_col:
        with st.container(border=True, key="eeg"):
            if demo.abnormal_summary:
                st.markdown(
                    '<div class="condition-banner">SIMULATED · '
                    + " · ".join(escape(item) for item in demo.abnormal_summary)
                    + "</div>",
                    unsafe_allow_html=True,
                )
            components.html(
                build_live_monitor_html(
                    demo,
                    active_channels,
                    contact_states,
                    initial_position_seconds=monitor_snapshot.position_seconds,
                    running=monitor_running,
                    loop=st.session_state.loop_playback,
                    state_label=controller.status.state.value,
                ),
                height=MONITOR_IFRAME_HEIGHT_PX,
                scrolling=False,
            )

    with dac_col:
        @st.fragment(run_every=telemetry_every)
        def render_live_dac() -> None:
            snapshot = controller.snapshot()
            if snapshot.completed:
                st.rerun()

            sample_index = min(
                max(snapshot.sample_index, 0),
                demo.samples_uv.shape[1] - 1,
            )
            values = demo.samples_uv[:, sample_index]
            codes = demo.dac_codes[:, sample_index]

            chips = []
            for index, channel in enumerate(demo.channel_names):
                enabled = channel in active_channels
                css_class = "dac-chip" if enabled else "dac-chip inactive"
                value_text = f"{values[index]:+.1f} µV" if enabled else "--"
                code_text = str(int(codes[index])) if enabled else "--"
                chips.append(
                    f'<div class="{css_class}"><b>{channel}</b>'
                    f'<span>{value_text}</span><code>{code_text}</code></div>'
                )

            with st.container(border=True):
                st.markdown(
                    '<div class="section-label">DAC</div>'
                    '<div class="dac-strip">' + "".join(chips) + "</div>",
                    unsafe_allow_html=True,
                )

        render_live_dac()


log_col, validation_col, chain_col = st.columns(3)
with log_col:
    with st.popover("Protocol log", width="stretch"):
        log_tools, _spacer = st.columns([0.18, 0.82])
        log_tools.button(
            "Clear",
            key="clear_log_btn",
            disabled=not controller.logs,
            on_click=controller.clear_logs,
        )
        if controller.logs:
            log_text = "\n".join(controller.logs[-20:])
            st.markdown(
                '<div class="log-box">' + escape(log_text) + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="log-box empty-log">No protocol traffic yet.</div>',
                unsafe_allow_html=True,
            )

with validation_col, st.popover("Validation", width="stretch"):
    st.button("Run validation", key="validation_btn", on_click=_safe_action, args=(_run_validation,))
    result = st.session_state.validation_result
    if result is not None:
        with st.container(border=True):
            status_text = "PASS" if result.overall_passed else "CHECK"
            st.markdown(
                f'<div class="validation-pass">SOFTWARE-REFERENCE VALIDATION · {status_text} · {result.passed_count}/{result.total_count} checks passed</div>',
                unsafe_allow_html=True,
            )
            v1, v2, v3 = st.columns(3)
            with v1:
                st.markdown(
                    f"""
                    <div class="validation-card">
                      <b>LEVEL 1 · Simulator output</b><br>
                      <span>RMSE {result.simulator_output.rmse_uv:.3f} µV<br>
                      r = {result.simulator_output.correlation:.5f}<br>
                      Gain error {result.simulator_output.gain_error_percent:.3f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with v2:
                st.markdown(
                    f"""
                    <div class="validation-card">
                      <b>LEVEL 2 · Acquisition</b><br>
                      <span>r = {result.acquisition.correlation:.5f}<br>
                      Amplitude ratio {result.acquisition.amplitude_ratio:.4f}<br>
                      Timing offset {result.acquisition.timing_offset_ms:.2f} ms</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with v3:
                st.markdown(
                    f"""
                    <div class="validation-card">
                      <b>LEVEL 3 · Event timing</b><br>
                      <span>Expected {result.event_detection.expected_start_seconds:.2f} s<br>
                      Detected {result.event_detection.detected_onset_seconds:.2f} s<br>
                      Latency {result.event_detection.latency_seconds:.2f} s</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.caption("Deterministic software mocks only; these are not physical-device measurements or clinical detection results.")

with chain_col, st.popover("Signal chain", width="stretch"):
    with st.container(border=True):
        st.markdown('<div class="section-label">SIGNAL CHAIN / SYSTEM ARCHITECTURE</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="chain">
              <div class="chain-card implemented"><b>EEG SOURCE</b><small>Recorded or synthetic<br>8 ch · 256 Hz example<br>IMPLEMENTED</small></div>
              <div class="chain-card implemented"><b>DESKTOP / WEB CONTROL</b><small>Playback · conditions<br>device abstraction<br>IMPLEMENTED</small></div>
              <div class="chain-card partial"><b>USB / SERIAL</b><small>Command path exists<br>binary waveform transfer<br>PARTIAL</small></div>
              <div class="chain-card future"><b>MCU / TIMER / DMA</b><small>Ring buffer<br>deterministic updates<br>FUTURE HARDWARE</small></div>
              <div class="chain-card future"><b>8-CH DAC / ANALOGUE</b><small>Scaling · attenuation<br>calibrated µV output<br>FUTURE HARDWARE</small></div>
              <div class="chain-card future"><b>ELECTRODE INTERFACE</b><small>Impedance · lead-off<br>fault switching<br>FUTURE HARDWARE</small></div>
              <div class="chain-card external"><b>EEG ACQUISITION</b><small>AFE · ADC · downstream<br>measurement path<br>EXTERNAL</small></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Known input → controlled signal path → acquisition → automated verification. The browser console intentionally keeps the physical-hardware boundary explicit.")


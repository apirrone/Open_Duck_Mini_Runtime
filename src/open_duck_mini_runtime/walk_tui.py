"""
Textual TUI for Open Duck Mini Runtime.

Entry points:
  walk        — local terminal TUI
  walk-serve  — serve the same TUI in a browser on port 7000
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    ProgressBar,
    Static,
)

HOME_DIR = os.path.expanduser("~")

# ── Shared state helpers ──────────────────────────────────────────────────────

_MOTOR_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]


# ── Widgets ───────────────────────────────────────────────────────────────────


class StatusPanel(Static):
    DEFAULT_CSS = """
    StatusPanel {
        border: round $primary;
        padding: 0 1;
        height: auto;
    }
    StatusPanel .label-key { color: $text-muted; }
    StatusPanel .foot-on  { color: $success; }
    StatusPanel .foot-off { color: $error; }
    """

    def compose(self) -> ComposeResult:
        yield Label("STATUS", classes="label-key")
        yield Label("State:   initializing", id="st-state")
        yield Label("Hz:      --", id="st-hz")
        yield Label("Phase:   --", id="st-phase")
        yield Label("Feet:    L:○  R:○", id="st-feet")
        yield Label("Motors:  ON", id="st-motors")

    def refresh_data(self, t: dict) -> None:
        state = t.get("state", "?")
        color = {"walking": "green", "paused": "yellow", "initializing": "dim"}.get(state, "white")
        self.query_one("#st-state", Label).update(f"State:   [{color}]{state}[/{color}]")
        self.query_one("#st-hz", Label).update(f"Hz:      {t.get('hz', 0.0):.1f}")
        self.query_one("#st-phase", Label).update(
            f"Phase:   {t.get('phase', 0.0):.1f}"
        )
        fc = t.get("feet_contacts", [False, False])
        lf = "[green]●[/green]" if fc[0] else "[red]○[/red]"
        rf = "[green]●[/green]" if fc[1] else "[red]○[/red]"
        self.query_one("#st-feet", Label).update(f"Feet:    L:{lf}  R:{rf}")
        mot = "[green]ON[/green]" if t.get("motors_enabled", True) else "[red]OFF[/red]"
        self.query_one("#st-motors", Label).update(f"Motors:  {mot}")


class IMUPanel(Static):
    DEFAULT_CSS = """
    IMUPanel {
        border: round $primary;
        padding: 0 1;
        height: auto;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("IMU", classes="label-key")
        yield Label("Gyro X:  0.000", id="imu-gx")
        yield Label("Gyro Y:  0.000", id="imu-gy")
        yield Label("Gyro Z:  0.000", id="imu-gz")
        yield Label("Accel X: 0.000", id="imu-ax")
        yield Label("Accel Y: 0.000", id="imu-ay")
        yield Label("Accel Z: 0.000", id="imu-az")

    def refresh_data(self, t: dict) -> None:
        imu = t.get("imu", {})
        g = imu.get("gyro", [0.0, 0.0, 0.0])
        a = imu.get("accel", [0.0, 0.0, 0.0])
        self.query_one("#imu-gx", Label).update(f"Gyro X:  {g[0]:+.3f}")
        self.query_one("#imu-gy", Label).update(f"Gyro Y:  {g[1]:+.3f}")
        self.query_one("#imu-gz", Label).update(f"Gyro Z:  {g[2]:+.3f}")
        self.query_one("#imu-ax", Label).update(f"Accel X: {a[0]:+.3f}")
        self.query_one("#imu-ay", Label).update(f"Accel Y: {a[1]:+.3f}")
        self.query_one("#imu-az", Label).update(f"Accel Z: {a[2]:+.3f}")


class ControllerPanel(Static):
    DEFAULT_CSS = """
    ControllerPanel {
        border: round $primary;
        padding: 0 1;
        height: auto;
    }
    ControllerPanel Label { margin-bottom: 0; }
    ControllerPanel ProgressBar { margin-bottom: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Label("CONTROLLER", classes="label-key")
        yield Label("Forward/Back", id="lbl-fwd")
        yield ProgressBar(total=1.0, show_eta=False, id="pb-fwd")
        yield Label("Turn", id="lbl-turn")
        yield ProgressBar(total=1.0, show_eta=False, id="pb-turn")
        yield Label("Strafe", id="lbl-strafe")
        yield ProgressBar(total=1.0, show_eta=False, id="pb-strafe")

    def refresh_data(self, t: dict) -> None:
        cmds = t.get("commands", [0.0] * 7)
        fwd   = cmds[0] if len(cmds) > 0 else 0.0
        yaw   = cmds[2] if len(cmds) > 2 else 0.0
        strafe = cmds[1] if len(cmds) > 1 else 0.0

        self.query_one("#lbl-fwd",    Label).update(f"Forward/Back  {fwd:+.2f}")
        self.query_one("#lbl-turn",   Label).update(f"Turn          {yaw:+.2f}")
        self.query_one("#lbl-strafe", Label).update(f"Strafe        {strafe:+.2f}")

        self.query_one("#pb-fwd",    ProgressBar).progress = min(abs(fwd),    1.0)
        self.query_one("#pb-turn",   ProgressBar).progress = min(abs(yaw),    1.0)
        self.query_one("#pb-strafe", ProgressBar).progress = min(abs(strafe), 1.0)


# ── Main App ──────────────────────────────────────────────────────────────────


class DuckApp(App):
    """Open Duck Mini Runtime — live telemetry dashboard."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #top-row {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }
    StatusPanel {
        width: 1fr;
        margin-right: 1;
    }
    IMUPanel {
        width: 1fr;
        margin-right: 1;
    }
    ControllerPanel {
        width: 1fr;
    }
    #motor-section {
        border: round $primary;
        padding: 0 1;
        height: 1fr;
    }
    DataTable {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("p", "toggle_pause",  "Pause/Resume"),
        Binding("m", "toggle_motors", "Motors On/Off"),
        Binding("q", "quit",          "Quit"),
    ]

    def __init__(self, rl_walk_kwargs: dict):
        super().__init__()
        self._rl_walk_kwargs = rl_walk_kwargs
        self._rl_walk = None
        self._walk_thread: threading.Thread | None = None

    # ── Layout ────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="top-row"):
            yield StatusPanel()
            yield IMUPanel()
            yield ControllerPanel()
        with Vertical(id="motor-section"):
            yield Label("MOTORS")
            yield DataTable(id="motor-table")
        yield Footer()

    def on_mount(self) -> None:
        # Set up motor table columns
        table = self.query_one("#motor-table", DataTable)
        table.add_columns("Joint", "Target", "Actual", "Velocity", "Error")
        for name in _MOTOR_NAMES:
            table.add_row(name, "--", "--", "--", "--", key=name)

        # Start RLWalk in a background daemon thread
        self._walk_thread = threading.Thread(
            target=self._run_walk, daemon=True, name="rl-walk"
        )
        self._walk_thread.start()

        # Poll telemetry every 100 ms
        self.set_interval(0.1, self._poll_telemetry)

    # ── Background walk thread ─────────────────────────────────

    def _run_walk(self) -> None:
        try:
            from open_duck_mini_runtime.rl_walk.walk import RLWalk
            self._rl_walk = RLWalk(**self._rl_walk_kwargs)
            self._rl_walk.run()
        except Exception as exc:
            # Surface fatal errors in the title bar
            self.call_from_thread(self.set_title, f"Duck Runtime — ERROR: {exc}")

    # ── Telemetry polling ──────────────────────────────────────

    def _poll_telemetry(self) -> None:
        if self._rl_walk is None:
            self.title = "Duck Runtime — initializing…"
            return

        with self._rl_walk._telem_lock:
            t = dict(self._rl_walk.telemetry)  # shallow copy under lock

        state = t.get("state", "?")
        hz    = t.get("hz", 0.0)
        self.title = f"Duck Runtime  ·  {state.upper()}  ·  {hz:.1f} Hz"

        self.query_one(StatusPanel).refresh_data(t)
        self.query_one(IMUPanel).refresh_data(t)
        self.query_one(ControllerPanel).refresh_data(t)
        self._refresh_motor_table(t)

    def _refresh_motor_table(self, t: dict) -> None:
        table  = self.query_one("#motor-table", DataTable)
        names  = t.get("motor_names", _MOTOR_NAMES)
        tgts   = t.get("motor_targets",   [0.0] * 14)
        pos    = t.get("motor_positions",  [0.0] * 14)
        vels   = t.get("motor_velocities", [0.0] * 14)

        for i, name in enumerate(names):
            if i >= len(tgts):
                break
            tgt = tgts[i]
            act = pos[i]
            vel = vels[i]
            err = tgt - act
            table.update_cell(name, "Target",   f"{tgt:+.3f}")
            table.update_cell(name, "Actual",   f"{act:+.3f}")
            table.update_cell(name, "Velocity", f"{vel:+.3f}")
            table.update_cell(name, "Error",    f"{err:+.3f}")

    # ── Key actions ────────────────────────────────────────────

    def action_toggle_pause(self) -> None:
        if self._rl_walk is not None:
            self._rl_walk.paused = not self._rl_walk.paused

    def action_toggle_motors(self) -> None:
        if self._rl_walk is None:
            return
        if self._rl_walk.motors_enabled:
            self._rl_walk.hwi.turn_off()
            self._rl_walk.motors_enabled = False
            self._rl_walk.paused = True
        else:
            self._rl_walk.start()
            self._rl_walk.motors_enabled = True
            self._rl_walk.paused = False


# ── Entry points ──────────────────────────────────────────────────────────────


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Open Duck Mini Runtime TUI")
    parser.add_argument(
        "--onnx_model_path", type=str,
        default=f"{HOME_DIR}/BEST_WALK_ONNX_2.onnx",
    )
    parser.add_argument(
        "--duck_config_path", type=str,
        default=f"{HOME_DIR}/duck_config.json",
    )
    parser.add_argument("-a", "--action_scale", type=float, default=0.25)
    parser.add_argument("-p", type=int, default=30)
    parser.add_argument("-i", type=int, default=0)
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-c", "--control_freq", type=int, default=50)
    parser.add_argument("--pitch_bias", type=float, default=0)
    parser.add_argument("--commands", action="store_true", default=True)
    parser.add_argument("--save_obs", default=False)
    parser.add_argument("--replay_obs", default=None)
    parser.add_argument("--cutoff_frequency", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    """Local terminal TUI."""
    args = _parse_args()
    kwargs = dict(
        onnx_model_path=args.onnx_model_path,
        duck_config_path=args.duck_config_path,
        action_scale=args.action_scale,
        pid=[args.p, args.i, args.d],
        control_freq=args.control_freq,
        commands=args.commands,
        pitch_bias=args.pitch_bias,
        save_obs=args.save_obs,
        replay_obs=args.replay_obs,
        cutoff_frequency=args.cutoff_frequency,
    )
    DuckApp(kwargs).run()


def serve_main() -> None:
    """Serve the TUI in a browser on port 7000."""
    from textual_serve.server import Server

    args = _parse_args()
    kwargs = dict(
        onnx_model_path=args.onnx_model_path,
        duck_config_path=args.duck_config_path,
        action_scale=args.action_scale,
        pid=[args.p, args.i, args.d],
        control_freq=args.control_freq,
        commands=args.commands,
        pitch_bias=args.pitch_bias,
        save_obs=args.save_obs,
        replay_obs=args.replay_obs,
        cutoff_frequency=args.cutoff_frequency,
    )

    def make_app():
        return DuckApp(kwargs)

    server = Server(make_app, port=7000, title="Duck Runtime")
    server.serve()


if __name__ == "__main__":
    main()

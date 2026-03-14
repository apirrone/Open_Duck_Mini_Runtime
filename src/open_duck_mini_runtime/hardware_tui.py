"""
Interactive hardware test TUI for Open Duck Mini Runtime.

Navigate between hardware components with ↑↓ arrow keys.
Each panel lets you test and interact with that subsystem live.

Usage:
    uv run test-hw
"""

from __future__ import annotations

import glob
import importlib
import math
import os
import threading
import time
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll, Vertical
from textual.widgets import (
    Button,
    ContentSwitcher,
    DataTable,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    Static,
)

HOME_DIR = Path.home()
SERIAL_PORT = os.environ.get("DUCK_SERIAL_PORT", "/dev/ttyACM0")

# 14 DOFs in hwi.py order
MOTOR_NAMES = [
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
MOTOR_IDS = [20, 21, 22, 23, 24, 30, 31, 32, 33, 10, 11, 12, 13, 14]

# Sidebar navigation: (display name, panel widget id)
COMPONENTS = [
    ("System Check",  "panel-system"),
    ("IMU",           "panel-imu"),
    ("Servos",        "panel-servos"),
    ("Foot Contacts", "panel-contacts"),
    ("Audio",         "panel-audio"),
    ("LEDs",          "panel-leds"),
    ("Antennas",      "panel-antennas"),
]


# ── Markup helpers ────────────────────────────────────────────────────────────

def _ok(text: str) -> str:
    return f"[green]✓[/green] {text}"


def _fail(text: str) -> str:
    return f"[red]✗[/red] {text}"


# ── Panel base CSS ────────────────────────────────────────────────────────────

_PANEL_CSS = """
{cls} {{
    padding: 1 2;
}}
{cls} .panel-title {{
    text-style: bold;
    color: $primary;
    margin-bottom: 1;
}}
{cls} .btn-row {{
    height: auto;
    margin-top: 1;
}}
{cls} .btn-row Button {{
    margin-right: 1;
}}
"""


# ═════════════════════════════════════════════════════════════════════════════
# Panel: System Check
# ═════════════════════════════════════════════════════════════════════════════

class SystemCheckPanel(VerticalScroll):
    DEFAULT_CSS = _PANEL_CSS.format(cls="SystemCheckPanel") + """
    SystemCheckPanel Button { margin-top: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Label("System Check", classes="panel-title")
        yield Label("[dim]Checks device files and library availability.[/dim]")
        yield Label("")
        yield Label("…", id="sc-i2c")
        yield Label("…", id="sc-serial")
        yield Label("…", id="sc-audio")
        yield Label("…", id="sc-gpio")
        yield Label("…", id="sc-onnx")
        yield Label("…", id="sc-config")
        yield Button("Re-run Checks", id="btn-recheck", variant="primary")

    def on_mount(self) -> None:
        self._run_checks()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-recheck":
            self._run_checks()

    def _set(self, label_id: str, ok: bool, name: str, detail: str = "") -> None:
        txt = f"{name}: {detail}" if detail else name
        self.query_one(f"#{label_id}", Label).update(_ok(txt) if ok else _fail(txt))

    def _run_checks(self) -> None:
        # I2C
        i2c = glob.glob("/dev/i2c-*")
        self._set(
            "sc-i2c", bool(i2c), "I2C bus (/dev/i2c-*)",
            ", ".join(i2c) if i2c
            else "not found — rebuild NixOS image with i2c_arm=on; see module.nix",
        )

        # Serial
        has_ser = Path(SERIAL_PORT).exists()
        self._set(
            "sc-serial", has_ser, f"Serial ({SERIAL_PORT})",
            "present" if has_ser else "not found — motor controller not plugged in?",
        )

        # Audio
        snd = glob.glob("/dev/snd/*")
        self._set(
            "sc-audio", bool(snd), "Audio (/dev/snd/*)",
            f"{len(snd)} device(s)" if snd
            else "none — I2S amp not configured? Check module.nix dt-overlays",
        )

        # GPIO
        gpio_ok, gpio_lib = False, "not found"
        for mod in ("RPi.GPIO", "lgpio"):
            try:
                importlib.import_module(mod)
                gpio_ok, gpio_lib = True, mod
                break
            except Exception as exc:
                gpio_lib = str(exc)
        self._set("sc-gpio", gpio_ok, "GPIO library", gpio_lib)

        # ONNX model
        onnx = HOME_DIR / "BEST_WALK_ONNX_2.onnx"
        self._set(
            "sc-onnx", onnx.exists(), "ONNX model",
            str(onnx) if onnx.exists() else f"not found at {onnx}",
        )

        # Duck config
        cfg = HOME_DIR / "duck_config.json"
        self._set(
            "sc-config", cfg.exists(), "Duck config",
            str(cfg) if cfg.exists() else f"not found at {cfg}",
        )


# ═════════════════════════════════════════════════════════════════════════════
# Panel: IMU (BNO055)
# ═════════════════════════════════════════════════════════════════════════════

class IMUTestPanel(VerticalScroll):
    DEFAULT_CSS = _PANEL_CSS.format(cls="IMUTestPanel")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._imu = None
        self._state: dict = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def compose(self) -> ComposeResult:
        yield Label("IMU (BNO055)", classes="panel-title")
        yield Label("[dim]Auto-connects on startup. Reads gyro + accel at 50 Hz.[/dim]")
        yield Label("[yellow]Connecting…[/yellow]", id="imu-status")
        yield Label("")
        yield Label("Gyro X:   ---  rad/s", id="imu-gx")
        yield Label("Gyro Y:   ---  rad/s", id="imu-gy")
        yield Label("Gyro Z:   ---  rad/s", id="imu-gz")
        yield Label("Accel X:  ---  m/s²",  id="imu-ax")
        yield Label("Accel Y:  ---  m/s²",  id="imu-ay")
        yield Label("Accel Z:  ---  m/s²",  id="imu-az")
        yield Label("Gravity:  checking…",   id="imu-grav")
        with Horizontal(classes="btn-row"):
            yield Button("Reconnect", id="btn-imu-connect", variant="primary")
            yield Button("Tare X",    id="btn-imu-tare",    variant="default")

    def on_mount(self) -> None:
        self.set_interval(0.1, self._poll)
        threading.Thread(target=self._init_imu, daemon=True, name="imu-init").start()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-imu-connect":
            self._stop.set()
            self._imu = None
            self._stop = threading.Event()
            with self._lock:
                self._state["status"] = "[yellow]Reconnecting…[/yellow]"
            threading.Thread(target=self._init_imu, daemon=True).start()
        elif event.button.id == "btn-imu-tare":
            if self._imu is not None:
                threading.Thread(target=self._imu.tare_x, daemon=True).start()

    def _init_imu(self) -> None:
        try:
            from open_duck_mini_runtime import raw_imu
            imu = raw_imu.Imu(sampling_freq=50, upside_down=False)
            time.sleep(0.5)
            self._imu = imu
            with self._lock:
                self._state["status"] = _ok("Connected — live data below")
            threading.Thread(
                target=self._read_loop, daemon=True, name="imu-read"
            ).start()
        except Exception as exc:
            with self._lock:
                self._state["status"] = _fail(f"Connection failed: {exc}")

    def _read_loop(self) -> None:
        import numpy as np

        while not self._stop.is_set() and self._imu is not None:
            try:
                d = self._imu.get_data()
                g = list(d.get("gyro", [0, 0, 0]))
                a = list(d.get("accelero", [0, 0, 0]))
                with self._lock:
                    self._state.update(
                        gyro=g,
                        accel=a,
                        grav_ok=float(np.linalg.norm(a)) > 5.0,
                    )
            except Exception:
                pass
            time.sleep(0.02)

    def _poll(self) -> None:
        with self._lock:
            s = dict(self._state)
        if "status" in s:
            self.query_one("#imu-status", Label).update(s["status"])
            del s["status"]
            with self._lock:
                self._state.pop("status", None)
        if "gyro" not in s:
            return
        g, a = s["gyro"], s["accel"]
        self.query_one("#imu-gx", Label).update(f"Gyro X:   {g[0]:+.3f} rad/s")
        self.query_one("#imu-gy", Label).update(f"Gyro Y:   {g[1]:+.3f} rad/s")
        self.query_one("#imu-gz", Label).update(f"Gyro Z:   {g[2]:+.3f} rad/s")
        self.query_one("#imu-ax", Label).update(f"Accel X:  {a[0]:+.3f} m/s²")
        self.query_one("#imu-ay", Label).update(f"Accel Y:  {a[1]:+.3f} m/s²")
        self.query_one("#imu-az", Label).update(f"Accel Z:  {a[2]:+.3f} m/s²")
        grav = s.get("grav_ok", False)
        self.query_one("#imu-grav", Label).update(
            _ok("Gravity OK (~9.8 m/s²)") if grav
            else _fail("No gravity signal — IMU not connected or I2C down?"),
        )


# ═════════════════════════════════════════════════════════════════════════════
# Panel: Feetech Servos
# ═════════════════════════════════════════════════════════════════════════════

class ServoPanel(VerticalScroll):
    DEFAULT_CSS = _PANEL_CSS.format(cls="ServoPanel") + """
    ServoPanel DataTable { height: 18; }
    ServoPanel #servo-jog-row { height: auto; margin-top: 1; }
    ServoPanel #servo-jog-row Button { margin-right: 1; }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._io = None
        self._responding: list[int] = []
        self._torque_on = False
        self._positions: dict[int, float] = {}
        self._lock = threading.Lock()
        self._poll_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._selected_idx: int = 0
        self._status_msg: Optional[str] = None
        self._row_updates: dict[str, str] = {}  # motor_name -> status markup

    def compose(self) -> ComposeResult:
        yield Label("Feetech Servos", classes="panel-title")
        yield Label(
            f"[dim]Port: {SERIAL_PORT}  ·  Right leg IDs 10–14, Left leg 20–24, Head 30–33[/dim]"
        )
        yield Label("", id="servo-status")
        yield DataTable(id="servo-table")
        with Horizontal(classes="btn-row"):
            yield Button("Detect All",     id="btn-detect",      variant="primary")
            yield Button("Enable Torque",  id="btn-torque-on",   variant="success")
            yield Button("Disable Torque", id="btn-torque-off",  variant="error")
        yield Label("")
        yield Label("Selected: (highlight a row)", id="servo-selected-label")
        with Horizontal(id="servo-jog-row"):
            yield Button("+0.1 rad", id="btn-jog-plus",  variant="default")
            yield Button("-0.1 rad", id="btn-jog-minus", variant="default")
            yield Button("Home",     id="btn-jog-home",  variant="default")
        yield Label(
            "[dim]Torque must be enabled before jogging.[/dim]", id="servo-torque-hint"
        )

    def on_mount(self) -> None:
        table = self.query_one("#servo-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Joint", "ID", "Position", "Status")
        for name, sid in zip(MOTOR_NAMES, MOTOR_IDS):
            table.add_row(name, str(sid), "---", "[dim]not scanned[/dim]", key=name)
        self.set_interval(0.2, self._poll)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-detect":
            threading.Thread(target=self._detect, daemon=True).start()
        elif bid == "btn-torque-on":
            threading.Thread(target=self._set_torque, args=(True,), daemon=True).start()
        elif bid == "btn-torque-off":
            threading.Thread(target=self._set_torque, args=(False,), daemon=True).start()
        elif bid == "btn-jog-plus":
            self._jog(+0.1)
        elif bid == "btn-jog-minus":
            self._jog(-0.1)
        elif bid == "btn-jog-home":
            self._jog_absolute(0.0)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        try:
            key = str(event.row_key.value)
            if key in MOTOR_NAMES:
                self._selected_idx = MOTOR_NAMES.index(key)
                sid = MOTOR_IDS[self._selected_idx]
                self.query_one("#servo-selected-label", Label).update(
                    f"Selected: [bold]{key}[/bold] (ID {sid})"
                )
        except Exception:
            pass

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detect(self) -> None:
        self._set_status("[yellow]Detecting servos…[/yellow]")
        try:
            import rustypot
            if self._io is None:
                self._io = rustypot.feetech(SERIAL_PORT, 1_000_000)
        except Exception as exc:
            self._set_status(_fail(f"Cannot open port: {exc}"))
            return

        responding = []
        for sid, name in zip(MOTOR_IDS, MOTOR_NAMES):
            try:
                pos = self._io.read_present_position([sid])[0]
                responding.append(sid)
                with self._lock:
                    self._positions[sid] = pos
                    self._row_updates[name] = _ok("ok")
            except Exception:
                with self._lock:
                    self._row_updates[name] = _fail("no response")

        with self._lock:
            self._responding = responding

        n, total = len(responding), len(MOTOR_IDS)
        msg = f"{n}/{total} servos detected"
        self._set_status(_ok(msg) if n == total else _fail(msg))

        if responding and (
            self._poll_thread is None or not self._poll_thread.is_alive()
        ):
            self._stop.clear()
            self._poll_thread = threading.Thread(
                target=self._pos_loop, daemon=True, name="servo-poll"
            )
            self._poll_thread.start()

    def _pos_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                ids = list(self._responding)
            for sid in ids:
                try:
                    pos = self._io.read_present_position([sid])[0]
                    with self._lock:
                        self._positions[sid] = pos
                except Exception:
                    pass
            time.sleep(0.2)

    # ── Torque ─────────────────────────────────────────────────────────────────

    def _set_torque(self, on: bool) -> None:
        with self._lock:
            ids = list(self._responding)
        if not ids:
            self._set_status(_fail("No servos detected — run Detect All first"))
            return
        try:
            if on:
                self._io.set_kps(ids, [30] * len(ids))
                self._torque_on = True
                self._set_status(_ok(f"Torque ENABLED on {len(ids)} servos"))
            else:
                self._io.disable_torque(ids)
                self._torque_on = False
                self._set_status("[yellow]Torque DISABLED[/yellow]")
        except Exception as exc:
            self._set_status(_fail(str(exc)))

    # ── Jogging ───────────────────────────────────────────────────────────────

    def _jog(self, delta: float) -> None:
        if not self._torque_on:
            self._set_status(_fail("Enable torque first before jogging"))
            return
        sid = MOTOR_IDS[self._selected_idx]
        with self._lock:
            if sid not in self._responding:
                self._set_status(_fail(f"ID {sid} not responding"))
                return
            current = self._positions.get(sid, 0.0)
        threading.Thread(
            target=self._write_goal, args=(sid, current + delta), daemon=True
        ).start()

    def _jog_absolute(self, target: float) -> None:
        if not self._torque_on:
            self._set_status(_fail("Enable torque first before jogging"))
            return
        sid = MOTOR_IDS[self._selected_idx]
        with self._lock:
            if sid not in self._responding:
                self._set_status(_fail(f"ID {sid} not responding"))
                return
        threading.Thread(
            target=self._write_goal, args=(sid, target), daemon=True
        ).start()

    def _write_goal(self, sid: int, target: float) -> None:
        try:
            self._io.write_goal_position([sid], [target])
        except Exception as exc:
            self._set_status(_fail(str(exc)))

    # ── Polling ───────────────────────────────────────────────────────────────

    def _set_status(self, msg: str) -> None:
        with self._lock:
            self._status_msg = msg

    def _poll(self) -> None:
        with self._lock:
            msg = self._status_msg
            self._status_msg = None
            updates = dict(self._row_updates)
            self._row_updates.clear()
            positions = dict(self._positions)

        if msg is not None:
            self.query_one("#servo-status", Label).update(msg)

        table = self.query_one("#servo-table", DataTable)
        for name, status in updates.items():
            table.update_cell(name, "Status", status)
        for name, sid in zip(MOTOR_NAMES, MOTOR_IDS):
            if sid in positions:
                table.update_cell(name, "Position", f"{positions[sid]:+.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# Panel: Foot Contacts
# ═════════════════════════════════════════════════════════════════════════════

class FootContactsPanel(VerticalScroll):
    DEFAULT_CSS = _PANEL_CSS.format(cls="FootContactsPanel") + """
    FootContactsPanel .contact-box {
        width: 1fr;
        height: 5;
        border: round $primary;
        content-align: center middle;
        text-align: center;
    }
    FootContactsPanel #fc-boxes { height: 7; }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._contacts = None
        self._state: dict = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def compose(self) -> ComposeResult:
        yield Label("Foot Contacts", classes="panel-title")
        yield Label("[dim]GPIO22 = left, GPIO27 = right.  Auto-connects on startup.[/dim]")
        yield Label("[yellow]Connecting…[/yellow]", id="fc-status")
        yield Label("")
        with Horizontal(id="fc-boxes"):
            yield Static("[dim]○ LEFT\nno data[/dim]",  id="fc-left",  classes="contact-box")
            yield Static("[dim]○ RIGHT\nno data[/dim]", id="fc-right", classes="contact-box")
        yield Label("")
        yield Label("GPIO pin states:", id="fc-pins")

    def on_mount(self) -> None:
        self.set_interval(0.1, self._poll)
        threading.Thread(target=self._init, daemon=True, name="fc-init").start()

    def _init(self) -> None:
        try:
            from open_duck_mini_runtime.feet_contacts import FeetContacts
            fc = FeetContacts()
            self._contacts = fc
            with self._lock:
                self._state["status"] = _ok("Connected — GPIO22 (L) / GPIO27 (R)")
            threading.Thread(
                target=self._read_loop, daemon=True, name="fc-read"
            ).start()
        except Exception as exc:
            with self._lock:
                self._state["status"] = _fail(f"Failed: {exc}")

    def _read_loop(self) -> None:
        while not self._stop.is_set() and self._contacts is not None:
            try:
                vals = self._contacts.get()
                with self._lock:
                    self._state["left"] = vals[0]
                    self._state["right"] = vals[1]
            except Exception:
                pass
            time.sleep(0.05)

    def _poll(self) -> None:
        with self._lock:
            s = dict(self._state)
        if "status" in s:
            self.query_one("#fc-status", Label).update(s["status"])
            with self._lock:
                self._state.pop("status", None)
        if "left" not in s:
            return
        left, right = s["left"], s["right"]
        self.query_one("#fc-left", Static).update(
            "[green]● LEFT\nIN CONTACT[/green]" if left
            else "[dim]○ LEFT\nno contact[/dim]"
        )
        self.query_one("#fc-right", Static).update(
            "[green]● RIGHT\nIN CONTACT[/green]" if right
            else "[dim]○ RIGHT\nno contact[/dim]"
        )
        self.query_one("#fc-pins", Label).update(
            f"GPIO22 (left): {'HIGH' if left else 'LOW'}  "
            f"GPIO27 (right): {'HIGH' if right else 'LOW'}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Panel: Audio (MAX98357A / I2S)
# ═════════════════════════════════════════════════════════════════════════════

class AudioPanel(VerticalScroll):
    DEFAULT_CSS = _PANEL_CSS.format(cls="AudioPanel") + """
    AudioPanel DataTable { height: 10; }
    AudioPanel #audio-vol-row { height: auto; margin-top: 1; }
    AudioPanel #audio-vol-row Button { margin-right: 1; }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sounds = None
        self._volume = 1.0
        self._selected_sound: int = 0

    def compose(self) -> ComposeResult:
        yield Label("Audio (MAX98357A / I2S)", classes="panel-title")
        yield Label("[dim]Requires I2S configured in module.nix (dtoverlay=max98357a).[/dim]")
        yield Label("", id="audio-snd-devs")
        yield Label("", id="audio-status")
        yield Label("")
        yield Label("[bold]Sound files:[/bold]")
        yield DataTable(id="audio-table")
        with Horizontal(classes="btn-row"):
            yield Button("Play Selected", id="btn-play",   variant="primary")
            yield Button("Stop",          id="btn-stop",   variant="error")
        with Horizontal(id="audio-vol-row"):
            yield Button("Vol +10%", id="btn-vol-up", variant="default")
            yield Button("Vol -10%", id="btn-vol-dn", variant="default")
            yield Label("Volume: 100%", id="audio-vol")

    def on_mount(self) -> None:
        # Check audio devices
        snd = glob.glob("/dev/snd/*")
        if snd:
            self.query_one("#audio-snd-devs", Label).update(
                _ok(f"Audio devices: {', '.join(snd)}")
            )
        else:
            self.query_one("#audio-snd-devs", Label).update(
                _fail("No /dev/snd/* found — I2S amp not configured (see module.nix dt-overlays)")
            )

        # Init pygame mixer + sound list
        table = self.query_one("#audio-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Index", "Filename")
        try:
            from open_duck_mini_runtime.sounds import Sounds
            self._sounds = Sounds(volume=self._volume)
            for i, wav in enumerate(self._sounds.wav_files):
                table.add_row(str(i), wav.name, key=str(i))
            self.query_one("#audio-status", Label).update(
                _ok(f"Pygame mixer ready — {len(self._sounds.wav_files)} sounds found")
            )
        except Exception as exc:
            self.query_one("#audio-status", Label).update(_fail(str(exc)))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        try:
            self._selected_sound = int(str(event.row_key.value))
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-play":
            self._play()
        elif bid == "btn-stop":
            self._stop_audio()
        elif bid == "btn-vol-up":
            self._set_volume(min(1.0, self._volume + 0.1))
        elif bid == "btn-vol-dn":
            self._set_volume(max(0.0, self._volume - 0.1))

    def _play(self) -> None:
        if self._sounds is None:
            return
        idx = self._selected_sound
        threading.Thread(
            target=self._sounds.play_sound, args=(idx,), daemon=True
        ).start()

    def _stop_audio(self) -> None:
        try:
            import pygame
            pygame.mixer.stop()
        except Exception:
            pass

    def _set_volume(self, vol: float) -> None:
        self._volume = vol
        if self._sounds is not None:
            self._sounds.volume = vol
        self.query_one("#audio-vol", Label).update(f"Volume: {int(vol * 100)}%")


# ═════════════════════════════════════════════════════════════════════════════
# Panel: NeoPixel LEDs
# ═════════════════════════════════════════════════════════════════════════════

class LEDPanel(VerticalScroll):
    DEFAULT_CSS = _PANEL_CSS.format(cls="LEDPanel") + """
    LEDPanel .color-row { height: auto; margin-bottom: 1; }
    LEDPanel .color-row Button { margin-right: 1; }
    LEDPanel .section-label { margin-top: 1; text-style: bold; }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._leds = None

    def compose(self) -> ComposeResult:
        yield Label("NeoPixel LEDs", classes="panel-title")
        yield Label(
            "[dim]Initialize first. LED index: 0=projector, 1=right eye, 2=left eye.[/dim]"
        )
        yield Label("", id="led-status")
        yield Button("Initialize LEDs", id="btn-led-init", variant="primary")

        yield Label("Both Eyes", classes="section-label")
        with Horizontal(classes="color-row"):
            yield Button("Red",   id="btn-eyes-red",   variant="error")
            yield Button("Green", id="btn-eyes-green", variant="success")
            yield Button("Blue",  id="btn-eyes-blue",  variant="primary")
            yield Button("White", id="btn-eyes-white", variant="default")
            yield Button("Off",   id="btn-eyes-off",   variant="warning")

        yield Label("Left Eye", classes="section-label")
        with Horizontal(classes="color-row"):
            yield Button("Red",   id="btn-leye-red",   variant="error")
            yield Button("Green", id="btn-leye-green", variant="success")
            yield Button("Blue",  id="btn-leye-blue",  variant="primary")
            yield Button("White", id="btn-leye-white", variant="default")

        yield Label("Right Eye", classes="section-label")
        with Horizontal(classes="color-row"):
            yield Button("Red",   id="btn-reye-red",   variant="error")
            yield Button("Green", id="btn-reye-green", variant="success")
            yield Button("Blue",  id="btn-reye-blue",  variant="primary")
            yield Button("White", id="btn-reye-white", variant="default")

        yield Label("Projector", classes="section-label")
        with Horizontal(classes="color-row"):
            yield Button("Red",    id="btn-proj-red",    variant="error")
            yield Button("Green",  id="btn-proj-green",  variant="success")
            yield Button("Blue",   id="btn-proj-blue",   variant="primary")
            yield Button("White",  id="btn-proj-white",  variant="default")
            yield Button("On/Off", id="btn-proj-toggle", variant="warning")

        yield Label("")
        yield Button("All Off", id="btn-all-off", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-led-init":
            self._init_leds()
            return

        if self._leds is None:
            self.query_one("#led-status", Label).update(
                _fail("Initialize LEDs first")
            )
            return

        actions: dict[str, object] = {
            "btn-eyes-red":    lambda: self._leds.set_eyes_color("red"),
            "btn-eyes-green":  lambda: self._leds.set_eyes_color("green"),
            "btn-eyes-blue":   lambda: self._leds.set_eyes_color("blue"),
            "btn-eyes-white":  lambda: self._leds.set_eyes_color("white"),
            "btn-eyes-off":    lambda: self._leds.set_eyes(False),
            "btn-leye-red":    lambda: self._leds.set_left_eye_color("red"),
            "btn-leye-green":  lambda: self._leds.set_left_eye_color("green"),
            "btn-leye-blue":   lambda: self._leds.set_left_eye_color("blue"),
            "btn-leye-white":  lambda: self._leds.set_left_eye_color("white"),
            "btn-reye-red":    lambda: self._leds.set_right_eye_color("red"),
            "btn-reye-green":  lambda: self._leds.set_right_eye_color("green"),
            "btn-reye-blue":   lambda: self._leds.set_right_eye_color("blue"),
            "btn-reye-white":  lambda: self._leds.set_right_eye_color("white"),
            "btn-proj-red":    lambda: self._leds.set_projector_color("red"),
            "btn-proj-green":  lambda: self._leds.set_projector_color("green"),
            "btn-proj-blue":   lambda: self._leds.set_projector_color("blue"),
            "btn-proj-white":  lambda: self._leds.set_projector_color("white"),
            "btn-proj-toggle": lambda: self._leds.set_projector(not self._leds.projector_on),
            "btn-all-off":     lambda: self._leds.all_off(),
        }
        fn = actions.get(bid)
        if fn:
            try:
                fn()
            except Exception as exc:
                self.query_one("#led-status", Label).update(_fail(str(exc)))

    def _init_leds(self) -> None:
        try:
            from open_duck_mini_runtime.led_controller import LedController
            if self._leds is not None:
                try:
                    self._leds.deinit()
                except Exception:
                    pass
            self._leds = LedController()
            self.query_one("#led-status", Label).update(_ok("LedController ready"))
        except Exception as exc:
            self.query_one("#led-status", Label).update(_fail(str(exc)))


# ═════════════════════════════════════════════════════════════════════════════
# Panel: MG90s Antennas
# ═════════════════════════════════════════════════════════════════════════════

class AntennaPanel(VerticalScroll):
    DEFAULT_CSS = _PANEL_CSS.format(cls="AntennaPanel") + """
    AntennaPanel .pos-row { height: auto; }
    AntennaPanel .pos-row Button { margin-right: 1; }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._antennas = None
        self._sweep_stop = threading.Event()
        self._left_pos: float = 0.0
        self._right_pos: float = 0.0

    def compose(self) -> ComposeResult:
        yield Label("MG90s Antennas", classes="panel-title")
        yield Label(
            "[dim]PWM via GPIO13 (left) and GPIO12 (right). Initialize first.[/dim]"
        )
        yield Label("", id="ant-status")
        with Horizontal(classes="btn-row"):
            yield Button("Initialize",      id="btn-ant-init", variant="primary")
            yield Button("Sweep Test (3s)", id="btn-sweep",    variant="default")
            yield Button("Stop Sweep",      id="btn-ant-stop", variant="error")

        yield Label("")
        yield Label("[bold]Left Antenna[/bold]")
        with Horizontal(classes="pos-row"):
            yield Button("+0.1", id="btn-l-plus",  variant="default")
            yield Button("Home", id="btn-l-home",  variant="default")
            yield Button("-0.1", id="btn-l-minus", variant="default")
        yield Label("Position: 0.00", id="ant-left-pos")

        yield Label("")
        yield Label("[bold]Right Antenna[/bold]")
        with Horizontal(classes="pos-row"):
            yield Button("+0.1", id="btn-r-plus",  variant="default")
            yield Button("Home", id="btn-r-home",  variant="default")
            yield Button("-0.1", id="btn-r-minus", variant="default")
        yield Label("Position: 0.00", id="ant-right-pos")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-ant-init":
            self._init_antennas()
        elif bid == "btn-sweep":
            self._sweep_stop.clear()
            threading.Thread(target=self._sweep, daemon=True, name="antenna-sweep").start()
        elif bid == "btn-ant-stop":
            self._sweep_stop.set()
            if self._antennas is not None:
                try:
                    self._antennas.set_position_left(0)
                    self._antennas.set_position_right(0)
                except Exception:
                    pass
        elif bid == "btn-l-plus":
            self._move_left(min(1.0, self._left_pos + 0.1))
        elif bid == "btn-l-minus":
            self._move_left(max(-1.0, self._left_pos - 0.1))
        elif bid == "btn-l-home":
            self._move_left(0.0)
        elif bid == "btn-r-plus":
            self._move_right(min(1.0, self._right_pos + 0.1))
        elif bid == "btn-r-minus":
            self._move_right(max(-1.0, self._right_pos - 0.1))
        elif bid == "btn-r-home":
            self._move_right(0.0)

    def _init_antennas(self) -> None:
        try:
            from open_duck_mini_runtime.antennas import Antennas
            if self._antennas is not None:
                try:
                    self._antennas.stop()
                except Exception:
                    pass
            self._antennas = Antennas()
            self.query_one("#ant-status", Label).update(_ok("Antennas ready"))
        except Exception as exc:
            self.query_one("#ant-status", Label).update(_fail(str(exc)))

    def _move_left(self, pos: float) -> None:
        if self._antennas is None:
            self.query_one("#ant-status", Label).update(_fail("Initialize first"))
            return
        self._left_pos = pos
        self._antennas.set_position_left(pos)
        self.query_one("#ant-left-pos", Label).update(f"Position: {pos:.2f}")

    def _move_right(self, pos: float) -> None:
        if self._antennas is None:
            self.query_one("#ant-status", Label).update(_fail("Initialize first"))
            return
        self._right_pos = pos
        self._antennas.set_position_right(pos)
        self.query_one("#ant-right-pos", Label).update(f"Position: {pos:.2f}")

    def _sweep(self) -> None:
        if self._antennas is None:
            return
        self.query_one("#ant-status", Label).update("[yellow]Sweeping…[/yellow]")
        start = time.monotonic()
        while not self._sweep_stop.is_set() and time.monotonic() - start < 3.0:
            val = math.sin(2 * math.pi * 1 * time.monotonic())
            self._antennas.set_position_left(val)
            self._antennas.set_position_right(val)
            time.sleep(1 / 50)
        if not self._sweep_stop.is_set():
            self._antennas.set_position_left(0)
            self._antennas.set_position_right(0)
        self.query_one("#ant-status", Label).update(_ok("Sweep complete"))


# ═════════════════════════════════════════════════════════════════════════════
# Main App
# ═════════════════════════════════════════════════════════════════════════════

class HardwareTUI(App):
    """Interactive hardware test TUI for Open Duck Mini Runtime."""

    TITLE = "Duck Hardware Tester"

    CSS = """
    Screen { layout: horizontal; }

    #sidebar {
        width: 22;
        border-right: solid $primary;
        padding: 0;
        height: 100%;
    }
    #sidebar-title {
        text-align: center;
        text-style: bold;
        background: $primary;
        color: $background;
        padding: 0 1;
        height: 1;
    }
    #sidebar ListView {
        height: 1fr;
        border: none;
    }

    ContentSwitcher {
        width: 1fr;
        height: 1fr;
    }
    ContentSwitcher > * {
        height: 100%;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Label("COMPONENTS", id="sidebar-title")
                yield ListView(
                    *[
                        ListItem(Label(name), id=f"item-{panel_id.replace('panel-', '')}")
                        for name, panel_id in COMPONENTS
                    ]
                )
            with ContentSwitcher(initial="panel-system"):
                yield SystemCheckPanel(id="panel-system")
                yield IMUTestPanel(id="panel-imu")
                yield ServoPanel(id="panel-servos")
                yield FootContactsPanel(id="panel-contacts")
                yield AudioPanel(id="panel-audio")
                yield LEDPanel(id="panel-leds")
                yield AntennaPanel(id="panel-antennas")
        yield Footer()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item and event.item.id:
            panel_id = "panel-" + event.item.id.replace("item-", "")
            try:
                self.query_one(ContentSwitcher).current = panel_id
            except Exception:
                pass


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    HardwareTUI().run()


if __name__ == "__main__":
    main()

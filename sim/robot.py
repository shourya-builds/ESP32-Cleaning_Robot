"""
ESP32 Hardware-in-the-Loop (HIL) Robot Simulation
Pygame differential drive simulator communicating with ESP32 via Serial UART.

Usage:
    python sim/robot.py                    # Auto-detect COM port or fallback to mock
    python sim/robot.py --port COM3        # Connect to specific serial port
    python sim/robot.py --mock             # Run in mock/software-in-the-loop mode
    python sim/robot.py --mock --headless  # Run headless for automated testing
"""

import sys
import os
import math
import time
import argparse
import threading
from typing import Optional, Tuple, List

import pygame

# Optional pyserial import with helpful fallback
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


# Simulation Constants & Cabin Room Definitions
WINDOW_WIDTH  = 900
WINDOW_HEIGHT = 700
FPS   = 60
SIM_DT = 1.0 / FPS

# Physics coordinate space: 100 px = 1 m
SCALE_PX_PER_M = 100.0

# Cabin: 5.0 m × 5.0 m  →  500 × 500 px in logic space
ROOM_SIZE = 500.0

# Centre the room in the window (HUD = 60 px at top)
HUD_H   = 60
USABLE_W = WINDOW_WIDTH
USABLE_H = WINDOW_HEIGHT - HUD_H
ROOM_X0 = (USABLE_W - ROOM_SIZE) / 2.0          # 200 px
ROOM_Y0 = HUD_H + (USABLE_H - ROOM_SIZE) / 2.0  # 130 px

# Wheel & robot geometry
WHEEL_BASE   = 20.0
ROBOT_RADIUS = 10.0

# Laser/sensor max range for DISPLAY only (8 cm = 8 px).
# Navigation uses detect_obstacles() clearances, not this.
SENSOR_DISPLAY_RANGE = 8.0
SENSOR_MAX_RANGE     = ROOM_SIZE   # used only by cast_distance_sensor (raycasts full room)

MODE_MANUAL      = 1
MODE_AUTOMATIC   = 2
MODE_DESTINATION = 3

# ---------------------------------------------------------------------------
# Logic-space <-> screen-space coordinate transform.
# Physics/PID all run in "logic" px (100 px = 1 m, room anchored at
# ROOM_X0/ROOM_Y0). The window renders that room zoomed to fit
# (VISUAL_SCALE) and re-centered (WIN_CX/WIN_CY vs ROOM_CX/ROOM_CY) — see
# draw(). Anything that turns a screen click into a world-space target MUST
# invert that same transform, or the target ends up nowhere near the
# cursor (this is what draw()'s s_x()/s_y() do in the forward direction).
# ---------------------------------------------------------------------------
_HUD_HEIGHT_T = 60
_USABLE_W_T   = WINDOW_WIDTH
_USABLE_H_T   = WINDOW_HEIGHT - _HUD_HEIGHT_T
VISUAL_SCALE  = min(_USABLE_W_T, _USABLE_H_T) / ROOM_SIZE
ROOM_CX = ROOM_X0 + ROOM_SIZE / 2.0
ROOM_CY = ROOM_Y0 + ROOM_SIZE / 2.0
WIN_CX  = WINDOW_WIDTH / 2.0
WIN_CY  = _HUD_HEIGHT_T + _USABLE_H_T / 2.0


def screen_to_logic(sx: float, sy: float) -> Tuple[float, float]:
    """Inverse of draw()'s s_x()/s_y() — screen pixel -> logic-space coord."""
    lx = ROOM_CX + (sx - WIN_CX) / VISUAL_SCALE
    ly = ROOM_CY + (sy - WIN_CY) / VISUAL_SCALE
    return lx, ly


class MockESP32Controller:
    """Software-in-the-loop controller with robust obstacle avoidance."""

    def __init__(self):
        self.prev_error = 0.0
        self.wheel_base = WHEEL_BASE
        self.max_speed = 120.0
        self.kp = 80.0
        self.kd = 10.0
        self.waypoint_tolerance = 12.0

        self.is_vacuum_on = True
        self.battery_pct = 95
        self.mode = MODE_MANUAL

        # Auto navigation state
        self.auto_state = "FORWARD"
        self.turn_dir = "LEFT"

        # Number of simulation frames remaining in current action
        self.state_timer = 0

        # Used to detect repeated left/right oscillation
        self.turn_history = []

        # Prevent immediately choosing the opposite direction
        self.last_escape_dir = None

    # ------------------------------------------------------------------
    # Direction scoring
    # ------------------------------------------------------------------

    def _score_left(self, dist_left, dist_fl, dist_back):
        """
        Score space on the left side.

        Forward-left and left are most important.
        A little rear clearance is useful when escaping.
        """
        return (
            dist_left * 1.4 +
            dist_fl * 1.8 +
            dist_back * 0.25
        )

    def _score_right(self, dist_right, dist_fr, dist_back):
        """Score space on the right side."""
        return (
            dist_right * 1.4 +
            dist_fr * 1.8 +
            dist_back * 0.25
        )

    def _choose_turn_direction(
        self,
        dist_left,
        dist_fl,
        dist_right,
        dist_fr,
        dist_back
    ):
        """
        Choose the side with more free space.

        We deliberately use the robot-relative distances instead
        of room X/Y coordinates. This works regardless of heading.
        """

        left_score = self._score_left(
            dist_left, dist_fl, dist_back
        )

        right_score = self._score_right(
            dist_right, dist_fr, dist_back
        )

        # If one side is clearly better, use it.
        if left_score > right_score * 1.15:
            direction = "LEFT"

        elif right_score > left_score * 1.15:
            direction = "RIGHT"

        else:
            # Similar clearance.
            # Prefer the previous escape direction to prevent
            # left-right-left-right oscillation.
            if self.last_escape_dir is not None:
                direction = self.last_escape_dir
            else:
                direction = "LEFT" if left_score >= right_score else "RIGHT"

        self.last_escape_dir = direction
        return direction

    # ------------------------------------------------------------------
    # Main controller
    # ------------------------------------------------------------------

    def compute(
        self,
        x: float,
        y: float,
        theta: float,
        tgt_x: float,
        tgt_y: float,
        dist_front: float,
        mode: int = 1,
        man_l: float = 0.0,
        man_r: float = 0.0,
        dist_left: float = 200.0,
        dist_right: float = 200.0,
        obstacle_detected: bool = False,
        dist_back: float = 200.0,
        dist_fl: float = 200.0,
        dist_fr: float = 200.0,
        wall_left: float = 500.0,
        wall_right: float = 500.0,
        wall_top: float = 500.0,
        wall_bottom: float = 500.0,
        is_stuck: bool = False
    ) -> Tuple[float, float]:

        self.mode = mode

        # ==============================================================
        # MANUAL
        # ==============================================================

        if mode == MODE_MANUAL:
            return man_l, man_r

        # ==============================================================
        # AUTOMATIC
        # ==============================================================

        if mode == MODE_AUTOMATIC:

            # ----------------------------------------------------------
            # Thresholds
            # ----------------------------------------------------------

            FRONT_DANGER = 20.0
            FRONT_SAFE = 38.0

            SIDE_DANGER = 15.0
            SIDE_SAFE = 30.0

            BACK_DANGER = 18.0

            # ----------------------------------------------------------
            # Timer helper
            # ----------------------------------------------------------

            if self.state_timer > 0:
                self.state_timer -= 1

            # ----------------------------------------------------------
            # Determine obstacle situation
            # ----------------------------------------------------------

            front_blocked = (
                obstacle_detected
                or dist_front < FRONT_DANGER
                or dist_fl < FRONT_DANGER
                or dist_fr < FRONT_DANGER
            )

            left_blocked = (
                dist_left < SIDE_DANGER
            )

            right_blocked = (
                dist_right < SIDE_DANGER
            )

            back_blocked = (
                dist_back < BACK_DANGER
            )

            # ==========================================================
            # CORNER DETECTION
            # ==========================================================

            # A corner/deadlock is when the robot has an obstacle
            # ahead AND one/both sides are extremely restricted.

            left_corner = (
                front_blocked
                and left_blocked
                and dist_fl < SIDE_DANGER
            )

            right_corner = (
                front_blocked
                and right_blocked
                and dist_fr < SIDE_DANGER
            )

            trapped = (
                front_blocked
                and left_blocked
                and right_blocked
            )

            # ==========================================================
            # STUCK -> CORNER ESCAPE
            # ==========================================================

            if is_stuck:

                self.auto_state = "CORNER_ESCAPE"

                self.state_timer = 30

                # IMPORTANT:
                # Do not always choose the same direction.
                #
                # If left is more open, escape left.
                # Otherwise escape right.

                if dist_left > dist_right:

                    self.turn_dir = "LEFT"

                else:

                    self.turn_dir = "RIGHT"

                return 0.0, 0.0

            # ==========================================================
            # FORWARD
            # ==========================================================

            if self.auto_state == "FORWARD":

                # ------------------------------------------------------
                # No obstacle
                # ------------------------------------------------------

                if not front_blocked:

                    # Keep some clearance from nearby obstacles.

                    if dist_left < SIDE_SAFE:

                        return 82.0, 96.0

                    if dist_right < SIDE_SAFE:

                        return 96.0, 82.0

                    return 90.0, 90.0

                # ------------------------------------------------------
                # We hit an obstacle.
                #
                # STOP completely before deciding.
                # ------------------------------------------------------

                self.auto_state = "DECIDE"

                self.state_timer = 4

                return 0.0, 0.0

            # ==========================================================
            # DECIDE
            # ==========================================================

            if self.auto_state == "DECIDE":

                # ------------------------------------------------------
                # If this is a corner, DO NOT simply pivot.
                # ------------------------------------------------------

                if trapped or left_corner or right_corner:

                    self.auto_state = "CORNER_ESCAPE"

                    self.state_timer = 30

                    # Choose the side with more room.

                    if dist_left > dist_right:

                        self.turn_dir = "LEFT"

                    else:

                        self.turn_dir = "RIGHT"

                    return 0.0, 0.0

                # ------------------------------------------------------
                # Normal obstacle.
                # ------------------------------------------------------

                left_score = (
                    dist_left
                    + 1.5 * dist_fl
                )

                right_score = (
                    dist_right
                    + 1.5 * dist_fr
                )

                # ------------------------------------------------------
                # Prefer the side with more clearance.
                # ------------------------------------------------------

                if (
                    left_score > right_score
                    and dist_left > SIDE_DANGER
                ):

                    self.turn_dir = "LEFT"

                    self.auto_state = "TURNING"

                    self.state_timer = 18

                    return 0.0, 0.0

                elif dist_right > SIDE_DANGER:

                    self.turn_dir = "RIGHT"

                    self.auto_state = "TURNING"

                    self.state_timer = 18

                    return 0.0, 0.0

                # ------------------------------------------------------
                # Both sides are blocked.
                # ------------------------------------------------------

                elif not back_blocked:

                    self.auto_state = "BACKING_UP"

                    self.state_timer = 22

                    return -55.0, -55.0

                # ------------------------------------------------------
                # Completely trapped.
                # ------------------------------------------------------

                else:

                    self.auto_state = "CORNER_ESCAPE"

                    self.state_timer = 35

                    if dist_left >= dist_right:

                        self.turn_dir = "LEFT"

                    else:

                        self.turn_dir = "RIGHT"

                    return 0.0, 0.0

            # ==========================================================
            # CORNER ESCAPE
            # ==========================================================

            if self.auto_state == "CORNER_ESCAPE":

                # ------------------------------------------------------
                # PHASE 1:
                # Back away while turning slightly.
                #
                # This is the key difference from the old behavior.
                #
                # Straight reverse can keep the robot trapped between
                # two obstacles. A curved reverse moves it diagonally
                # out of the corner.
                # ------------------------------------------------------

                if self.state_timer > 15:

                    if not back_blocked:

                        if self.turn_dir == "LEFT":

                            # Reverse while curving left.
                            return -72.0, -42.0

                        else:

                            # Reverse while curving right.
                            return -42.0, -72.0

                    # If directly behind is blocked, don't reverse.
                    self.state_timer = 15

                # ------------------------------------------------------
                # PHASE 2:
                # Strong committed turn.
                # ------------------------------------------------------

                if self.state_timer > 0:

                    if self.turn_dir == "LEFT":

                        return -75.0, 80.0

                    else:

                        return 80.0, -75.0

                # ------------------------------------------------------
                # PHASE 3:
                # Check whether we escaped.
                # ------------------------------------------------------

                if (
                    dist_front > FRONT_SAFE
                    and (
                        dist_left > SIDE_DANGER
                        or dist_right > SIDE_DANGER
                    )
                ):

                    self.auto_state = "FORWARD"

                    self.state_timer = 12

                    return 70.0, 70.0

                # ------------------------------------------------------
                # Still trapped.
                #
                # Reverse again, but switch direction.
                # This prevents left-right deadlock.
                # ------------------------------------------------------

                if not back_blocked:

                    if self.turn_dir == "LEFT":

                        self.turn_dir = "RIGHT"

                    else:

                        self.turn_dir = "LEFT"

                    self.state_timer = 22

                    self.auto_state = "CORNER_ESCAPE"

                    return -50.0, -50.0

                # No rear clearance either.
                # Perform a strong pivot.
                self.state_timer = 25

                return (
                    (-80.0, 85.0)
                    if self.turn_dir == "LEFT"
                    else
                    (85.0, -80.0)
                )

            # ==========================================================
            # BACKING UP
            # ==========================================================

            if self.auto_state == "BACKING_UP":

                # ------------------------------------------------------
                # Never blindly reverse into another leg.
                # ------------------------------------------------------

                if back_blocked:

                    self.auto_state = "CORNER_ESCAPE"

                    self.state_timer = 25

                    if dist_left >= dist_right:

                        self.turn_dir = "LEFT"

                    else:

                        self.turn_dir = "RIGHT"

                    return 0.0, 0.0

                if self.state_timer > 0:

                    self.state_timer -= 1

                    return -55.0, -55.0

                # Finished backing up.
                self.auto_state = "DECIDE"

                self.state_timer = 3

                return 0.0, 0.0

            # ==========================================================
            # TURNING
            # ==========================================================

            if self.auto_state == "TURNING":

                # ------------------------------------------------------
                # Commit to selected direction.
                # ------------------------------------------------------

                if self.turn_dir == "LEFT":

                    turn_l = -65.0
                    turn_r = 75.0

                    selected_clearance = min(
                        dist_left,
                        dist_fl
                    )

                else:

                    turn_l = 75.0
                    turn_r = -65.0

                    selected_clearance = min(
                        dist_right,
                        dist_fr
                    )

                # ------------------------------------------------------
                # Keep turning for the committed duration.
                # ------------------------------------------------------

                if self.state_timer > 0:

                    self.state_timer -= 1

                    return turn_l, turn_r

                # ------------------------------------------------------
                # We finished the turn.
                # ------------------------------------------------------

                if (
                    dist_front > FRONT_SAFE
                    and dist_fl > SIDE_DANGER
                    and dist_fr > SIDE_DANGER
                ):

                    self.auto_state = "FORWARD"

                    self.state_timer = 10

                    return 70.0, 70.0

                # ------------------------------------------------------
                # Direction is still blocked.
                # Don't keep spinning.
                #
                # Try a corner escape.
                # ------------------------------------------------------

                self.auto_state = "CORNER_ESCAPE"

                self.state_timer = 25

                return 0.0, 0.0

            # ==========================================================
            # SAFETY FALLBACK
            # ==========================================================

            self.auto_state = "CORNER_ESCAPE"

            self.state_timer = 25

            if dist_left >= dist_right:

                self.turn_dir = "LEFT"

            else:

                self.turn_dir = "RIGHT"

            return 0.0, 0.0
        elif mode == MODE_DESTINATION:

            dx = tgt_x - x
            dy = tgt_y - y

            dist = math.hypot(dx, dy)

            desired_heading = math.atan2(dy, dx)

            error = desired_heading - theta

            while error > math.pi:
                error -= 2.0 * math.pi

            while error < -math.pi:
                error += 2.0 * math.pi

            d_error = error - self.prev_error
            self.prev_error = error

            if dist <= self.waypoint_tolerance:
                return 0.0, 0.0

            blocking_obstacle = (
                (obstacle_detected or dist_front < 35.0)
                and dist_front <
                max(0.0, dist - ROBOT_RADIUS - 4.0)
            )

            if blocking_obstacle:

                if dist_left >= dist_right:
                    return -30.0, 60.0
                else:
                    return 60.0, -30.0

            angular_speed = (
                self.kp * error
                + self.kd * d_error
            )

            alignment = math.cos(error)

            linear_speed = (
                self.max_speed *
                max(0.0, alignment)
            )

            if dist < 60.0:
                linear_speed *= dist / 60.0

            left_cmd = max(
                -self.max_speed,
                min(
                    self.max_speed,
                    linear_speed -
                    angular_speed * self.wheel_base / 2.0
                )
            )

            right_cmd = max(
                -self.max_speed,
                min(
                    self.max_speed,
                    linear_speed +
                    angular_speed * self.wheel_base / 2.0
                )
            )

            return left_cmd, right_cmd

        return 0.0, 0.0


class SerialHILClient:
    """Manages Serial communication with the physical ESP32 or mock fallback."""
    def __init__(self, port: Optional[str] = None, baud: int = 115200, mock: bool = False):
        self.port = port
        self.baud = baud
        self.is_mock = mock
        self.ser = None
        self.running = False
        self.left_cmd = 0.0
        self.right_cmd = 0.0
        self.last_response_time = time.time()
        self.packet_count = 0
        self.status_msg = "Initializing..."

        self.mock_controller = MockESP32Controller() if mock else None
        self.is_vacuum_on = True
        self.battery_pct = 95
        self.thread: Optional[threading.Thread] = None

        self._connect()

    def _auto_detect_port(self) -> Optional[str]:
        if not SERIAL_AVAILABLE:
            return None
        ports = serial.tools.list_ports.comports()
        for p in ports:
            # Common ESP32 USB-to-UART identifiers (CP210x, CH340, FTDI, ESP32)
            desc = (p.description + " " + (p.manufacturer or "")).lower()
            if any(k in desc for k in ["cp210", "ch340", "ch341", "ftdi", "uart", "esp32", "usb to"]):
                return p.device
        if ports:
            return ports[0].device
        return None

    def _connect(self):
        if self.is_mock or not SERIAL_AVAILABLE:
            self.is_mock = True
            self.status_msg = "MOCK MODE (Software-in-the-Loop)"
            return

        target_port = self.port or self._auto_detect_port()
        if not target_port:
            print("[INFO] No active serial port detected. Defaulting to Mock mode.")
            self.is_mock = True
            self.mock_controller = MockESP32Controller()
            self.status_msg = "MOCK MODE (No COM port found)"
            return

        try:
            if "://" in str(target_port):
                self.ser = serial.serial_for_url(target_port, baudrate=self.baud, timeout=0.05)
            else:
                self.ser = serial.Serial(target_port, self.baud, timeout=0.05)
            self.port = target_port
            self.is_mock = False
            self.running = True
            self.status_msg = f"CONNECTED ({self.port})"
            print(f"[INFO] Connected to ESP32 on {self.port}")
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
        except Exception as e:
            print(f"[WARN] Failed to open {target_port}: {e}. Falling back to Mock mode.")
            self.is_mock = True
            self.mock_controller = MockESP32Controller()
            self.status_msg = f"MOCK MODE (Failed {target_port})"

    def _read_loop(self):
        """Asynchronous reader for incoming serial command packets."""
        while self.running and self.ser and self.ser.is_open:
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if line.startswith("CMD,"):
                    parts = line.split(",")
                    if len(parts) >= 3:
                        self.left_cmd = float(parts[1])
                        self.right_cmd = float(parts[2])
                        self.last_response_time = time.time()
                        self.packet_count += 1
                elif line.startswith("STATUS,"):
                    sub_msg = line.split(',', 1)[1]
                    self.status_msg = f"ESP32: {sub_msg}"
                    if "CLIFF" in sub_msg or "PAUSED" in sub_msg or "LOW_BATT" in sub_msg:
                        self.is_vacuum_on = False
                    elif "CLEANING" in sub_msg or "INITIALIZED" in sub_msg:
                        self.is_vacuum_on = True
            except Exception:
                pass

    def send_telemetry(self, x: float, y: float, theta: float,
                       tgt_x: float, tgt_y: float, dist_front: float,
                       mode: int = 1, man_l: float = 0.0, man_r: float = 0.0,
                       dist_left: float = 200.0, dist_right: float = 200.0,
                       obstacle_detected: bool = False,
                       dist_back: float = 200.0,
                       dist_fl: float = 200.0, dist_fr: float = 200.0,
                       wall_left: float = 500.0, wall_right: float = 500.0,
                       wall_top: float = 500.0, wall_bottom: float = 500.0,
                       is_stuck: bool = False):
        """Send telemetry packet to ESP32 or process in mock."""
        if self.is_mock:
            self.left_cmd, self.right_cmd = self.mock_controller.compute(
                x, y, theta, tgt_x, tgt_y, dist_front, mode, man_l, man_r,
                dist_left, dist_right, obstacle_detected,
                dist_back, dist_fl, dist_fr,
                wall_left, wall_right, wall_top, wall_bottom,
                is_stuck
            )
            self.last_response_time = time.time()
            self.packet_count += 1
            return

        if self.ser and self.ser.is_open:
            # Packet: TELEM,x,y,theta,tgt_x,tgt_y,dist_front,mode,man_l,man_r\n
            msg = f"TELEM,{x:.1f},{y:.1f},{theta:.3f},{tgt_x:.1f},{tgt_y:.1f},{dist_front:.1f},{mode},{man_l:.1f},{man_r:.1f}\n"
            try:
                self.ser.write(msg.encode("utf-8"))
            except Exception as e:
                self.status_msg = f"TX Error: {e}"

    def set_mode(self, mode: int):
        """Send explicit mode switch command to ESP32."""
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(f"MODE,{mode}\n".encode("utf-8"))
            except Exception:
                pass

    def close(self):
        self.running = False
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(b"CMD,0.0,0.0\n")
                self.ser.close()
            except Exception:
                pass

class RobotSimulation:
    def __init__(self, client: SerialHILClient, headless: bool = False):
        self.client = client
        self.headless = headless

        # -------------------------------------------------------------
        # Robot state
        # -------------------------------------------------------------
        self.mode = MODE_MANUAL

        self.x = ROOM_X0 + 50.0
        self.y = ROOM_Y0 + 50.0
        self.theta = 0.0

        self.v_l = 0.0
        self.v_r = 0.0

        # -------------------------------------------------------------
        # Destination / waypoint
        # -------------------------------------------------------------
        self.target_x = ROOM_X0 + 400.0
        self.target_y = ROOM_Y0 + 400.0

        # -------------------------------------------------------------
        # Furniture geometry
        #
        # IMPORTANT:
        # The table/chair rectangles are VISUAL ONLY.
        # Only the individual legs in self.obstacles are physical
        # collision obstacles.
        #
        # Therefore the robot CAN go underneath the table.
        # -------------------------------------------------------------

        cx = ROOM_X0 + 250.0
        cy = ROOM_Y0 + 250.0

        # Table: 80 cm x 50 cm
        self.table_rect = pygame.Rect(
            int(cx - 40),
            int(cy - 25),
            80,
            50
        )

        # Chair 1
        self.chair1_rect = pygame.Rect(
            int(cx - 13),
            int(cy - 60),
            26,
            26
        )

        # Chair 2
        self.chair2_rect = pygame.Rect(
            int(cx - 38),
            int(cy + 34),
            26,
            26
        )

        # Chair 3
        self.chair3_rect = pygame.Rect(
            int(cx + 12),
            int(cy + 34),
            26,
            26
        )

        # -------------------------------------------------------------
        # Physical obstacles
        #
        # ONLY these legs are collision obstacles.
        # -------------------------------------------------------------

        self.obstacles: List[Tuple[float, float, float]] = [

            # -------------------------
            # Table legs
            # -------------------------
            (cx - 35.0, cy - 20.0, 4.0),
            (cx + 35.0, cy - 20.0, 4.0),
            (cx - 35.0, cy + 20.0, 4.0),
            (cx + 35.0, cy + 20.0, 4.0),

            # -------------------------
            # Chair 1 legs
            # -------------------------
            (cx - 10.0, cy - 58.0, 3.0),
            (cx + 10.0, cy - 58.0, 3.0),
            (cx - 10.0, cy - 36.0, 3.0),
            (cx + 10.0, cy - 36.0, 3.0),

            # -------------------------
            # Chair 2 legs
            # -------------------------
            (cx - 36.0, cy + 36.0, 3.0),
            (cx - 14.0, cy + 36.0, 3.0),
            (cx - 36.0, cy + 58.0, 3.0),
            (cx - 14.0, cy + 58.0, 3.0),

            # -------------------------
            # Chair 3 legs
            # -------------------------
            (cx + 14.0, cy + 36.0, 3.0),
            (cx + 36.0, cy + 36.0, 3.0),
            (cx + 14.0, cy + 58.0, 3.0),
            (cx + 36.0, cy + 58.0, 3.0),
        ]

        # -------------------------------------------------------------
        # Sensor / navigation state
        # -------------------------------------------------------------

        self.trail: List[Tuple[float, float]] = []
        self.max_trail = 400

        self.front_dist = SENSOR_MAX_RANGE

        self.detected_legs: List[
            Tuple[float, float, float]
        ] = []

        self.obstacle_detected = False

        self.dist_left = 200.0
        self.dist_right = 200.0
        self.dist_back = 200.0
        self.dist_fl = 200.0
        self.dist_fr = 200.0

        self.wall_left = ROOM_SIZE
        self.wall_right = ROOM_SIZE
        self.wall_top = ROOM_SIZE
        self.wall_bottom = ROOM_SIZE

        # -------------------------------------------------------------
        # Stuck detection
        # -------------------------------------------------------------

        self.pos_history: List[
            Tuple[float, float]
        ] = []

        self.is_stuck = False

        # -------------------------------------------------------------
        # Pygame
        # -------------------------------------------------------------

        if not self.headless:

            pygame.init()

            pygame.display.set_caption(
                "ESP32 HIL Simulation - 2m x 2m Cabin Environment"
            )

            self.screen = pygame.display.set_mode(
                (int(WINDOW_WIDTH), int(WINDOW_HEIGHT))
            )

            self.clock = pygame.time.Clock()

            self.font_small = pygame.font.Font(
                None,
                18
            )

            self.font_bold = pygame.font.Font(
                None,
                20
            )

            self.font_title = pygame.font.Font(
                None,
                24
            )

    # =================================================================
    # OBSTACLE DETECTION
    # =================================================================

    def detect_obstacles(self):
        """
        Multi-direction virtual obstacle scanner.

        The scanner checks:
        - Cabin walls
        - 4 table legs
        - 12 chair legs

        The tabletop and chair rectangles are NOT physical obstacles.

        This allows the robot to travel underneath the table.
        """

        DETECT_RANGE = 45.0

        # Wider forward detection field
        FRONT_FOV = math.radians(55.0)

        # -------------------------------------------------------------
        # Wall clearances
        # -------------------------------------------------------------

        wall_left = (
            self.x - ROBOT_RADIUS
        ) - ROOM_X0

        wall_right = (
            ROOM_X0 + ROOM_SIZE
        ) - (
            self.x + ROBOT_RADIUS
        )

        wall_top = (
            self.y - ROBOT_RADIUS
        ) - ROOM_Y0

        wall_bottom = (
            ROOM_Y0 + ROOM_SIZE
        ) - (
            self.y + ROBOT_RADIUS
        )

        def wall_clear_in_direction(angle):

            ux = math.cos(angle)
            uy = math.sin(angle)

            d = 2000.0

            if ux > 0.001:
                d = min(
                    d,
                    wall_right / ux
                )

            elif ux < -0.001:
                d = min(
                    d,
                    wall_left / (-ux)
                )

            if uy > 0.001:
                d = min(
                    d,
                    wall_bottom / uy
                )

            elif uy < -0.001:
                d = min(
                    d,
                    wall_top / (-uy)
                )

            return max(0.0, d)

        # -------------------------------------------------------------
        # Initial wall distances
        # -------------------------------------------------------------

        dist_front = wall_clear_in_direction(
            self.theta
        )

        dist_fl = wall_clear_in_direction(
            self.theta - math.radians(45.0)
        )

        dist_fr = wall_clear_in_direction(
            self.theta + math.radians(45.0)
        )

        dist_left = wall_clear_in_direction(
            self.theta - math.radians(90.0)
        )

        dist_right = wall_clear_in_direction(
            self.theta + math.radians(90.0)
        )

        dist_back = wall_clear_in_direction(
            self.theta + math.pi
        )

        detected_legs = []

        obstacle_detected = False

        # -------------------------------------------------------------
        # Wall detection
        # -------------------------------------------------------------

        if (
            dist_front < 28.0
            or dist_fl < 22.0
            or dist_fr < 22.0
        ):
            obstacle_detected = True

        # -------------------------------------------------------------
        # Scan every furniture leg
        # -------------------------------------------------------------

        for ox, oy, r in self.obstacles:

            dx = ox - self.x
            dy = oy - self.y

            center_distance = math.hypot(
                dx,
                dy
            )

            # Distance between robot surface and leg surface
            surface_distance = (
                center_distance
                - ROBOT_RADIUS
                - r
            )

            surface_distance = max(
                0.0,
                surface_distance
            )

            # Angle from robot to obstacle
            angle = math.atan2(
                dy,
                dx
            )

            relative_angle = (
                angle - self.theta
            )

            # Normalize angle
            while relative_angle > math.pi:
                relative_angle -= 2 * math.pi

            while relative_angle < -math.pi:
                relative_angle += 2 * math.pi

            degrees = math.degrees(
                relative_angle
            )

            # ---------------------------------------------------------
            # Directional distances
            # ---------------------------------------------------------

            # Directly ahead
            if abs(degrees) <= 25.0:

                dist_front = min(
                    dist_front,
                    surface_distance
                )

            # Front-left
            if -70.0 <= degrees < 0.0:

                dist_fl = min(
                    dist_fl,
                    surface_distance
                )

            # Front-right
            if 0.0 < degrees <= 70.0:

                dist_fr = min(
                    dist_fr,
                    surface_distance
                )

            # Left
            if -115.0 <= degrees < -25.0:

                dist_left = min(
                    dist_left,
                    surface_distance
                )

            # Right
            if 25.0 < degrees <= 115.0:

                dist_right = min(
                    dist_right,
                    surface_distance
                )

            # Behind
            if abs(degrees) >= 135.0:

                dist_back = min(
                    dist_back,
                    surface_distance
                )

            # ---------------------------------------------------------
            # Forward detection cone
            # ---------------------------------------------------------

            if (
                surface_distance <= DETECT_RANGE
                and abs(relative_angle) <= FRONT_FOV
            ):

                detected_legs.append(
                    (ox, oy, r)
                )

                obstacle_detected = True

        return (
            detected_legs,
            obstacle_detected,
            dist_left,
            dist_right,
            dist_back,
            dist_fl,
            dist_fr,
            wall_left,
            wall_right,
            wall_top,
            wall_bottom
        )

    # =================================================================
    # FRONT DISTANCE SENSOR
    # =================================================================

    def cast_distance_sensor(self) -> float:
        """
        Raycast forward to cabin walls or furniture legs.
        """

        sensor_dist = SENSOR_MAX_RANGE

        cos_t = math.cos(
            self.theta
        )

        sin_t = math.sin(
            self.theta
        )

        # -------------------------------------------------------------
        # Cabin walls
        # -------------------------------------------------------------

        if cos_t > 0:

            sensor_dist = min(
                sensor_dist,
                (
                    (ROOM_X0 + ROOM_SIZE) - self.x
                ) / cos_t
            )

        elif cos_t < 0:

            sensor_dist = min(
                sensor_dist,
                (
                    ROOM_X0 - self.x
                ) / cos_t
            )

        if sin_t > 0:

            sensor_dist = min(
                sensor_dist,
                (
                    (ROOM_Y0 + ROOM_SIZE) - self.y
                ) / sin_t
            )

        elif sin_t < 0:

            sensor_dist = min(
                sensor_dist,
                (
                    ROOM_Y0 - self.y
                ) / sin_t
            )

        # -------------------------------------------------------------
        # Furniture legs
        # -------------------------------------------------------------

        for ox, oy, r in self.obstacles:

            dx = ox - self.x
            dy = oy - self.y

            proj = (
                dx * cos_t
                + dy * sin_t
            )

            if proj > 0:

                perp_sq = (
                    dx * dx
                    + dy * dy
                    - proj * proj
                )

                if perp_sq < r * r:

                    chord_dist = math.sqrt(
                        max(
                            0.0,
                            r * r - perp_sq
                        )
                    )

                    hit_dist = (
                        proj - chord_dist
                    )

                    if (
                        0 < hit_dist
                        < sensor_dist
                    ):

                        sensor_dist = hit_dist

        return max(
            0.0,
            sensor_dist
        )

    # =================================================================
    # PHYSICS
    # =================================================================

    def step_physics(self, dt: float):

        # -------------------------------------------------------------
        # Manual WASD input
        # -------------------------------------------------------------

        man_l = 0.0
        man_r = 0.0

        if (
            self.mode == MODE_MANUAL
            and not self.headless
        ):

            keys = pygame.key.get_pressed()

            speed = 100.0
            turn_spd = 70.0

            fw = keys[pygame.K_w]
            bw = keys[pygame.K_s]
            lt = keys[pygame.K_a]
            rt = keys[pygame.K_d]

            if fw:

                if lt:

                    man_l = 30.0
                    man_r = 100.0

                elif rt:

                    man_l = 100.0
                    man_r = 30.0

                else:

                    man_l = speed
                    man_r = speed

            elif bw:

                if lt:

                    man_l = -30.0
                    man_r = -100.0

                elif rt:

                    man_l = -100.0
                    man_r = -30.0

                else:

                    man_l = -speed
                    man_r = -speed

            elif lt:

                man_l = -turn_spd
                man_r = turn_spd

            elif rt:

                man_l = turn_spd
                man_r = -turn_spd

        # -------------------------------------------------------------
        # Update sensors
        # -------------------------------------------------------------

        result = self.detect_obstacles()

        (
            self.detected_legs,
            self.obstacle_detected,
            self.dist_left,
            self.dist_right,
            self.dist_back,
            self.dist_fl,
            self.dist_fr,
            self.wall_left,
            self.wall_right,
            self.wall_top,
            self.wall_bottom
        ) = result

        self.front_dist = (
            self.cast_distance_sensor()
        )

        # -------------------------------------------------------------
        # Stuck detection
        #
        # Turning and backing up are NOT considered stuck.
        # -------------------------------------------------------------

        self.pos_history.append(
            (self.x, self.y)
        )

        if len(self.pos_history) > 45:

            self.pos_history.pop(0)

        self.is_stuck = False

        if (
            len(self.pos_history) == 45
            and self.mode == MODE_AUTOMATIC
        ):

            controller = (
                self.client.mock_controller
            )

            auto_state = (
                controller.auto_state
                if controller is not None
                else "FORWARD"
            )

            if auto_state == "FORWARD":

                old_x, old_y = (
                    self.pos_history[0]
                )

                displacement = math.hypot(
                    self.x - old_x,
                    self.y - old_y
                )

                if displacement < 8.0:

                    self.is_stuck = True

                    self.pos_history.clear()

        # -------------------------------------------------------------
        # Send telemetry
        # -------------------------------------------------------------

        self.client.send_telemetry(
            self.x,
            self.y,
            self.theta,
            self.target_x,
            self.target_y,
            self.front_dist,
            self.mode,
            man_l,
            man_r,
            self.dist_left,
            self.dist_right,
            self.obstacle_detected,
            self.dist_back,
            self.dist_fl,
            self.dist_fr,
            self.wall_left,
            self.wall_right,
            self.wall_top,
            self.wall_bottom,
            self.is_stuck
        )

        # -------------------------------------------------------------
        # Receive motor commands
        # -------------------------------------------------------------

        self.v_l = self.client.left_cmd
        self.v_r = self.client.right_cmd

        # -------------------------------------------------------------
        # Differential drive
        # -------------------------------------------------------------

        v = (
            self.v_l + self.v_r
        ) / 2.0

        w = (
            self.v_r - self.v_l
        ) / WHEEL_BASE

        self.x += (
            v
            * math.cos(self.theta)
            * dt
        )

        self.y += (
            v
            * math.sin(self.theta)
            * dt
        )

        self.theta += (
            w * dt
        )

        # -------------------------------------------------------------
        # Normalize angle
        # -------------------------------------------------------------

        while self.theta > math.pi:

            self.theta -= (
                2.0 * math.pi
            )

        while self.theta < -math.pi:

            self.theta += (
                2.0 * math.pi
            )

        # -------------------------------------------------------------
        # Cabin wall collision
        # -------------------------------------------------------------

        self.x = max(
            ROOM_X0 + ROBOT_RADIUS,
            min(
                ROOM_X0
                + ROOM_SIZE
                - ROBOT_RADIUS,
                self.x
            )
        )

        self.y = max(
            ROOM_Y0 + ROBOT_RADIUS,
            min(
                ROOM_Y0
                + ROOM_SIZE
                - ROBOT_RADIUS,
                self.y
            )
        )

        # -------------------------------------------------------------
        # Furniture LEG collision
        #
        # IMPORTANT:
        # No table_rect or chair_rect collision here.
        #
        # This means the robot CAN go underneath the table.
        # -------------------------------------------------------------

        for ox, oy, r in self.obstacles:

            dx = self.x - ox
            dy = self.y - oy

            distance = math.hypot(
                dx,
                dy
            )

            min_distance = (
                r + ROBOT_RADIUS
            )

            if (
                0 < distance
                < min_distance
            ):

                overlap = (
                    min_distance
                    - distance
                )

                self.x += (
                    dx / distance
                ) * overlap

                self.y += (
                    dy / distance
                ) * overlap

        # -------------------------------------------------------------
        # Trail
        # -------------------------------------------------------------

        if (
            not self.trail
            or math.hypot(
                self.x - self.trail[-1][0],
                self.y - self.trail[-1][1]
            ) > 4.0
        ):

            self.trail.append(
                (self.x, self.y)
            )

            if len(self.trail) > self.max_trail:

                self.trail.pop(0)

    # =================================================================
    # DRAW
    # =================================================================

    def draw(self):

        HUD_HEIGHT = 60

        USABLE_W = WINDOW_WIDTH
        USABLE_H = WINDOW_HEIGHT - HUD_HEIGHT

        VISUAL_SCALE = min(
            USABLE_W,
            USABLE_H
        ) / ROOM_SIZE

        ROOM_CX = (
            ROOM_X0
            + ROOM_SIZE / 2.0
        )

        ROOM_CY = (
            ROOM_Y0
            + ROOM_SIZE / 2.0
        )

        WIN_CX = (
            WINDOW_WIDTH / 2.0
        )

        WIN_CY = (
            HUD_HEIGHT
            + USABLE_H / 2.0
        )

        # -------------------------------------------------------------
        # Coordinate helpers
        # -------------------------------------------------------------

        def s_x(lx):

            return (
                WIN_CX
                + (lx - ROOM_CX)
                * VISUAL_SCALE
            )

        def s_y(ly):

            return (
                WIN_CY
                + (ly - ROOM_CY)
                * VISUAL_SCALE
            )

        def s_r(lr):

            return max(
                1,
                int(lr * VISUAL_SCALE)
            )

        def s_rect(rect):

            return pygame.Rect(
                int(s_x(rect.left)),
                int(s_y(rect.top)),
                int(s_r(rect.width)),
                int(s_r(rect.height))
            )

        def sc(
            surface,
            color,
            lx,
            ly,
            lr,
            width=0
        ):

            pygame.draw.circle(
                surface,
                color,
                (
                    int(s_x(lx)),
                    int(s_y(ly))
                ),
                s_r(lr),
                width
            )

        def sl(
            surface,
            color,
            lp1,
            lp2,
            width=1
        ):

            pygame.draw.line(
                surface,
                color,
                (
                    int(s_x(lp1[0])),
                    int(s_y(lp1[1]))
                ),
                (
                    int(s_x(lp2[0])),
                    int(s_y(lp2[1]))
                ),
                width
            )

        # -------------------------------------------------------------
        # Background
        # -------------------------------------------------------------

        self.screen.fill(
            (18, 22, 28)
        )

        # -------------------------------------------------------------
        # Cabin floor
        # -------------------------------------------------------------

        room_vis = s_rect(
            pygame.Rect(
                ROOM_X0,
                ROOM_Y0,
                ROOM_SIZE,
                ROOM_SIZE
            )
        )

        pygame.draw.rect(
            self.screen,
            (52, 42, 34),
            room_vis
        )

        # Horizontal floor lines
        for gy in range(
            int(ROOM_Y0),
            int(ROOM_Y0 + ROOM_SIZE) + 1,
            25
        ):

            sl(
                self.screen,
                (66, 54, 44),
                (ROOM_X0, gy),
                (
                    ROOM_X0 + ROOM_SIZE,
                    gy
                ),
                1
            )

        # Vertical floor lines
        for gx in range(
            int(ROOM_X0),
            int(ROOM_X0 + ROOM_SIZE) + 1,
            50
        ):

            sl(
                self.screen,
                (60, 49, 40),
                (gx, ROOM_Y0),
                (
                    gx,
                    ROOM_Y0 + ROOM_SIZE
                ),
                1
            )

        # -------------------------------------------------------------
        # Cabin walls
        # -------------------------------------------------------------

        pygame.draw.rect(
            self.screen,
            (160, 110, 60),
            room_vis,
            10
        )

        pygame.draw.rect(
            self.screen,
            (210, 160, 100),
            room_vis,
            2
        )

        pygame.draw.rect(
            self.screen,
            (100, 70, 35),
            room_vis,
            1
        )

        # -------------------------------------------------------------
        # Room label
        # -------------------------------------------------------------

        lbl = self.font_small.render(
            "2.0 m",
            True,
            (160, 130, 80)
        )

        self.screen.blit(
            lbl,
            (
                room_vis.left + 4,
                room_vis.top + 4
            )
        )

        # -------------------------------------------------------------
        # Robot trail
        # -------------------------------------------------------------

        if len(self.trail) > 1:

            trail_color = (
                (60, 200, 180)
                if self.mode == MODE_AUTOMATIC
                else (60, 130, 200)
            )

            scaled_trail = [
                (
                    int(s_x(tx)),
                    int(s_y(ty))
                )
                for tx, ty in self.trail
            ]

            pygame.draw.lines(
                self.screen,
                trail_color,
                False,
                scaled_trail,
                2
            )

        # -------------------------------------------------------------
        # Chair 1
        # -------------------------------------------------------------

        v_c1 = s_rect(
            self.chair1_rect
        )

        pygame.draw.rect(
            self.screen,
            (45, 100, 160),
            v_c1,
            border_radius=4
        )

        pygame.draw.rect(
            self.screen,
            (80, 150, 210),
            v_c1,
            2,
            border_radius=4
        )

        pygame.draw.line(
            self.screen,
            (130, 200, 255),
            (
                v_c1.left + 3,
                v_c1.top + 3
            ),
            (
                v_c1.right - 3,
                v_c1.top + 3
            ),
            4
        )

        # -------------------------------------------------------------
        # Chair 2
        # -------------------------------------------------------------

        v_c2 = s_rect(
            self.chair2_rect
        )

        pygame.draw.rect(
            self.screen,
            (45, 100, 160),
            v_c2,
            border_radius=4
        )

        pygame.draw.rect(
            self.screen,
            (80, 150, 210),
            v_c2,
            2,
            border_radius=4
        )

        pygame.draw.line(
            self.screen,
            (130, 200, 255),
            (
                v_c2.left + 3,
                v_c2.bottom - 3
            ),
            (
                v_c2.right - 3,
                v_c2.bottom - 3
            ),
            4
        )

        # -------------------------------------------------------------
        # Chair 3
        # -------------------------------------------------------------

        v_c3 = s_rect(
            self.chair3_rect
        )

        pygame.draw.rect(
            self.screen,
            (45, 100, 160),
            v_c3,
            border_radius=4
        )

        pygame.draw.rect(
            self.screen,
            (80, 150, 210),
            v_c3,
            2,
            border_radius=4
        )

        pygame.draw.line(
            self.screen,
            (130, 200, 255),
            (
                v_c3.left + 3,
                v_c3.bottom - 3
            ),
            (
                v_c3.right - 3,
                v_c3.bottom - 3
            ),
            4
        )

        # -------------------------------------------------------------
        # Tabletop
        #
        # VISUAL ONLY.
        # It is NOT used for collision.
        # -------------------------------------------------------------

        v_tbl = s_rect(
            self.table_rect
        )

        pygame.draw.rect(
            self.screen,
            (140, 80, 35),
            v_tbl,
            border_radius=5
        )

        pygame.draw.rect(
            self.screen,
            (200, 130, 60),
            v_tbl,
            2,
            border_radius=5
        )

        for gi in range(3):

            gy_tbl = (
                v_tbl.top
                + v_tbl.height
                * (gi + 1)
                // 4
            )

            pygame.draw.line(
                self.screen,
                (155, 95, 45),
                (
                    v_tbl.left + 4,
                    gy_tbl
                ),
                (
                    v_tbl.right - 4,
                    gy_tbl
                ),
                1
            )

        # -------------------------------------------------------------
        # Physical legs
        # -------------------------------------------------------------

        for ox, oy, r in self.obstacles:

            sc(
                self.screen,
                (200, 55, 55),
                ox,
                oy,
                r + 1
            )

            sc(
                self.screen,
                (240, 110, 110),
                ox,
                oy,
                r + 1,
                1
            )

        # -------------------------------------------------------------
        # Detected legs
        # -------------------------------------------------------------

        for ox, oy, r in self.detected_legs:

            sl(
                self.screen,
                (255, 200, 40),
                (self.x, self.y),
                (ox, oy),
                2
            )

            sc(
                self.screen,
                (255, 220, 50),
                ox,
                oy,
                r + 4,
                2
            )

        # -------------------------------------------------------------
        # Forward detection cone
        # -------------------------------------------------------------

        cone_surf = pygame.Surface(
            (
                WINDOW_WIDTH,
                WINDOW_HEIGHT
            ),
            pygame.SRCALPHA
        )

        cone_range = (
            ROBOT_RADIUS + 45.0
        )

        fov_half = math.radians(
            55.0
        )

        num_pts = 16

        arc_points = [
            (
                s_x(self.x),
                s_y(self.y)
            )
        ]

        for i in range(
            num_pts + 1
        ):

            ang = (
                self.theta
                - fov_half
                + (
                    2.0
                    * fov_half
                    * i
                    / num_pts
                )
            )

            px = (
                self.x
                + cone_range
                * math.cos(ang)
            )

            py = (
                self.y
                + cone_range
                * math.sin(ang)
            )

            arc_points.append(
                (
                    s_x(px),
                    s_y(py)
                )
            )

        if self.obstacle_detected:

            cone_fill = (
                255,
                60,
                60,
                60
            )

            cone_outline = (
                255,
                80,
                80
            )

        else:

            cone_fill = (
                40,
                220,
                100,
                35
            )

            cone_outline = (
                80,
                240,
                140
            )

        if len(arc_points) > 2:

            pygame.draw.polygon(
                cone_surf,
                cone_fill,
                arc_points
            )

            pygame.draw.polygon(
                cone_surf,
                cone_outline,
                arc_points,
                1
            )

        self.screen.blit(
            cone_surf,
            (0, 0)
        )

        # -------------------------------------------------------------
        # Destination target
        # -------------------------------------------------------------

        if self.mode == MODE_DESTINATION:

            pulse = (
                4
                * math.sin(
                    time.time() * 6.0
                )
            )

            sc(
                self.screen,
                (255, 180, 50),
                self.target_x,
                self.target_y,
                10 + pulse,
                2
            )

            sc(
                self.screen,
                (255, 210, 80),
                self.target_x,
                self.target_y,
                3
            )

        # -------------------------------------------------------------
        # Forward sensor ray
        # -------------------------------------------------------------

        sensor_end_x = (
            self.x
            + self.front_dist
            * math.cos(self.theta)
        )

        sensor_end_y = (
            self.y
            + self.front_dist
            * math.sin(self.theta)
        )

        sl(
            self.screen,
            (200, 55, 55),
            (self.x, self.y),
            (
                sensor_end_x,
                sensor_end_y
            ),
            1
        )

        sc(
            self.screen,
            (255, 60, 60),
            sensor_end_x,
            sensor_end_y,
            2
        )

        # -------------------------------------------------------------
        # Robot body
        # -------------------------------------------------------------

        sc(
            self.screen,
            (25, 35, 50),
            self.x,
            self.y,
            ROBOT_RADIUS
        )

        sc(
            self.screen,
            (35, 150, 215),
            self.x,
            self.y,
            ROBOT_RADIUS - 1
        )

        sc(
            self.screen,
            (190, 235, 255),
            self.x,
            self.y,
            ROBOT_RADIUS,
            2
        )

        # -------------------------------------------------------------
        # Front bumper
        # -------------------------------------------------------------

        b_ang = 0.75

        p_left = (
            self.x
            + ROBOT_RADIUS
            * math.cos(
                self.theta - b_ang
            ),
            self.y
            + ROBOT_RADIUS
            * math.sin(
                self.theta - b_ang
            )
        )

        p_front = (
            self.x
            + (ROBOT_RADIUS + 1.5)
            * math.cos(self.theta),
            self.y
            + (ROBOT_RADIUS + 1.5)
            * math.sin(self.theta)
        )

        p_right = (
            self.x
            + ROBOT_RADIUS
            * math.cos(
                self.theta + b_ang
            ),
            self.y
            + ROBOT_RADIUS
            * math.sin(
                self.theta + b_ang
            )
        )

        scaled_bumper = [
            (
                int(s_x(px)),
                int(s_y(py))
            )
            for px, py in [
                p_left,
                p_front,
                p_right
            ]
        ]

        pygame.draw.lines(
            self.screen,
            (230, 60, 60),
            False,
            scaled_bumper,
            2
        )

        # -------------------------------------------------------------
        # Vacuum
        # -------------------------------------------------------------

        vac_color = (
            (0, 230, 255)
            if self.client.is_vacuum_on
            else (80, 90, 100)
        )

        sc(
            self.screen,
            vac_color,
            self.x,
            self.y,
            4
        )

        if self.client.is_vacuum_on:

            sc(
                self.screen,
                (255, 255, 255),
                self.x,
                self.y,
                1
            )

        # -------------------------------------------------------------
        # Side sweeper brush
        # -------------------------------------------------------------

        brush_base_ang = (
            self.theta + 0.65
        )

        bx_l = (
            self.x
            + (ROBOT_RADIUS - 2)
            * math.cos(brush_base_ang)
        )

        by_l = (
            self.y
            + (ROBOT_RADIUS - 2)
            * math.sin(brush_base_ang)
        )

        spin = (
            (time.time() * 14.0)
            % (2.0 * math.pi)
            if self.client.is_vacuum_on
            else 0.0
        )

        for b_sub in [
            spin,
            spin + 2.094,
            spin + 4.188
        ]:

            br_end_x = (
                bx_l
                + 5
                * math.cos(b_sub)
            )

            br_end_y = (
                by_l
                + 5
                * math.sin(b_sub)
            )

            sl(
                self.screen,
                (255, 215, 60),
                (bx_l, by_l),
                (
                    br_end_x,
                    br_end_y
                ),
                2
            )

        # -------------------------------------------------------------
        # Differential-drive wheels
        # -------------------------------------------------------------

        perp = (
            self.theta
            + math.pi / 2.0
        )

        w_offset = (
            WHEEL_BASE / 2.0
        )

        for side in [-1, 1]:

            wx = (
                self.x
                + side
                * w_offset
                * math.cos(perp)
            )

            wy = (
                self.y
                + side
                * w_offset
                * math.sin(perp)
            )

            w_len = 8
            w_th = 3

            p1 = (
                wx
                - (w_len / 2)
                * math.cos(self.theta)
                - (w_th / 2)
                * math.sin(self.theta),

                wy
                - (w_len / 2)
                * math.sin(self.theta)
                + (w_th / 2)
                * math.cos(self.theta)
            )

            p2 = (
                wx
                + (w_len / 2)
                * math.cos(self.theta)
                - (w_th / 2)
                * math.sin(self.theta),

                wy
                + (w_len / 2)
                * math.sin(self.theta)
                + (w_th / 2)
                * math.cos(self.theta)
            )

            p3 = (
                wx
                + (w_len / 2)
                * math.cos(self.theta)
                + (w_th / 2)
                * math.sin(self.theta),

                wy
                + (w_len / 2)
                * math.sin(self.theta)
                - (w_th / 2)
                * math.cos(self.theta)
            )

            p4 = (
                wx
                - (w_len / 2)
                * math.cos(self.theta)
                + (w_th / 2)
                * math.sin(self.theta),

                wy
                - (w_len / 2)
                * math.sin(self.theta)
                - (w_th / 2)
                * math.cos(self.theta)
            )

            scaled_wheel = [
                (
                    int(s_x(px)),
                    int(s_y(py))
                )
                for px, py in [
                    p1,
                    p2,
                    p3,
                    p4
                ]
            ]

            pygame.draw.polygon(
                self.screen,
                (18, 18, 18),
                scaled_wheel
            )

            pygame.draw.polygon(
                self.screen,
                (90, 90, 90),
                scaled_wheel,
                1
            )

        # -------------------------------------------------------------
        # Heading indicator
        # -------------------------------------------------------------

        head_x = (
            self.x
            + (ROBOT_RADIUS + 4)
            * math.cos(self.theta)
        )

        head_y = (
            self.y
            + (ROBOT_RADIUS + 4)
            * math.sin(self.theta)
        )

        sl(
            self.screen,
            (255, 255, 80),
            (self.x, self.y),
            (head_x, head_y),
            2
        )

        # -------------------------------------------------------------
        # HUD
        # -------------------------------------------------------------

        self._render_hud()

        pygame.display.flip()

    # =================================================================
    # HUD
    # =================================================================

    def _render_hud(self):

        HUD_H = 58

        hud_surf = pygame.Surface(
            (
                WINDOW_WIDTH,
                HUD_H
            ),
            pygame.SRCALPHA
        )

        hud_surf.fill(
            (10, 15, 22, 245)
        )

        self.screen.blit(
            hud_surf,
            (0, 0)
        )

        pygame.draw.line(
            self.screen,
            (55, 75, 105),
            (
                0,
                HUD_H - 1
            ),
            (
                WINDOW_WIDTH,
                HUD_H - 1
            ),
            1
        )

        # -------------------------------------------------------------
        # Connection status
        # -------------------------------------------------------------

        badge_color = (
            (80, 220, 100)
            if not self.client.is_mock
            else (240, 160, 40)
        )

        auto_state_str = ""

        if (
            self.client.mock_controller
            and hasattr(
                self.client.mock_controller,
                "auto_state"
            )
        ):

            auto_state_str = (
                self.client.mock_controller.auto_state
            )

        mode_labels = {

            MODE_MANUAL:
                "MANUAL (WASD)",

            MODE_AUTOMATIC:
                f"AUTO ({auto_state_str or 'FORWARD'})",

            MODE_DESTINATION:
                "DESTINATION (Waypoint PID)"
        }

        mode_colors = {

            MODE_MANUAL:
                (0, 230, 255),

            MODE_AUTOMATIC:
                (100, 255, 120),

            MODE_DESTINATION:
                (255, 180, 50)
        }

        # -------------------------------------------------------------
        # Position in metres
        # -------------------------------------------------------------

        rel_x_m = (
            self.x - ROOM_X0
        ) / SCALE_PX_PER_M

        rel_y_m = (
            self.y - ROOM_Y0
        ) / SCALE_PX_PER_M

        # -------------------------------------------------------------
        # Row 1
        # -------------------------------------------------------------

        title_s = self.font_title.render(
            "2m×2m CABIN ROBOT SIM",
            True,
            (255, 255, 255)
        )

        mode_s = self.font_bold.render(
            f"[{mode_labels[self.mode]}]",
            True,
            mode_colors[self.mode]
        )

        status_s = self.font_bold.render(
            self.client.status_msg,
            True,
            badge_color
        )

        self.screen.blit(
            title_s,
            (10, 5)
        )

        curr_x = (
            10
            + title_s.get_width()
            + 14
        )

        self.screen.blit(
            mode_s,
            (curr_x, 5)
        )

        curr_x += (
            mode_s.get_width()
            + 10
        )

        # Obstacle warning
        if self.obstacle_detected:

            obs_bg = pygame.Rect(
                curr_x,
                3,
                155,
                20
            )

            pygame.draw.rect(
                self.screen,
                (220, 40, 40),
                obs_bg,
                border_radius=3
            )

            obs_s = self.font_bold.render(
                "OBSTACLE DETECTED",
                True,
                (255, 255, 255)
            )

            self.screen.blit(
                obs_s,
                (
                    curr_x + 6,
                    5
                )
            )

        self.screen.blit(
            status_s,
            (
                WINDOW_WIDTH
                - status_s.get_width()
                - 10,
                5
            )
        )

        # -------------------------------------------------------------
        # Row 2
        # -------------------------------------------------------------

        pose_s = self.font_small.render(
            (
                f"Pose  "
                f"X={rel_x_m:.2f}m  "
                f"Y={rel_y_m:.2f}m  "
                f"θ={math.degrees(self.theta):.1f}°"
            ),
            True,
            (200, 220, 240)
        )

        sens_s = self.font_small.render(
            (
                f"Sensor "
                f"{self.front_dist / SCALE_PX_PER_M:.2f} m"
            ),
            True,
            (250, 140, 140)
        )

        mot_s = self.font_small.render(
            (
                f"Motors  "
                f"L={self.v_l:.0f}  "
                f"R={self.v_r:.0f}"
            ),
            True,
            (140, 240, 160)
        )

        hint_s = self.font_small.render(
            (
                "[1] Manual  "
                "[2] Auto  "
                "[3] Dest  |  "
                "WASD drive  |  R reset"
            ),
            True,
            (200, 200, 130)
        )

        row2_y = 33

        for surf, col_x in zip(
            [
                pose_s,
                sens_s,
                mot_s,
                hint_s
            ],
            [
                10,
                285,
                445,
                592
            ]
        ):

            self.screen.blit(
                surf,
                (
                    col_x,
                    row2_y
                )
            )

    # =================================================================
    # MAIN SIMULATION LOOP
    # =================================================================

    def run(
        self,
        max_seconds: Optional[float] = None
    ):

        start_time = time.time()

        running = True

        try:

            while running:

                # -----------------------------------------------------
                # Optional duration
                # -----------------------------------------------------

                if (
                    max_seconds
                    and (
                        time.time()
                        - start_time
                    ) > max_seconds
                ):

                    break

                # -----------------------------------------------------
                # Pygame events
                # -----------------------------------------------------

                if not self.headless:

                    for event in pygame.event.get():

                        # Window close
                        if event.type == pygame.QUIT:

                            running = False

                        # Mouse click = destination
                        elif (
                            event.type
                            == pygame.MOUSEBUTTONDOWN
                        ):

                            mx, my = (
                                pygame.mouse.get_pos()
                            )

                            if event.button == 1:

                                lx, ly = (
                                    screen_to_logic(
                                        float(mx),
                                        float(my)
                                    )
                                )

                                self.target_x = lx
                                self.target_y = ly

                        # Keyboard
                        elif (
                            event.type
                            == pygame.KEYDOWN
                        ):

                            # -------------------------------------------------
                            # Mode 1
                            # -------------------------------------------------

                            if event.key in (
                                pygame.K_1,
                                pygame.K_KP1
                            ):

                                self.mode = MODE_MANUAL

                                self.client.set_mode(
                                    MODE_MANUAL
                                )

                                print(
                                    "[MODE SWITCH] "
                                    "Switched to Mode 1: "
                                    "Manual Mode (WASD)"
                                )

                            # -------------------------------------------------
                            # Mode 2
                            # -------------------------------------------------

                            elif event.key in (
                                pygame.K_2,
                                pygame.K_KP2
                            ):

                                self.mode = MODE_AUTOMATIC

                                self.client.set_mode(
                                    MODE_AUTOMATIC
                                )

                                # Reset Auto controller state
                                if self.client.mock_controller:

                                    controller = (
                                        self.client.mock_controller
                                    )

                                    controller.auto_state = (
                                        "FORWARD"
                                    )

                                    controller.state_timer = 0

                                    controller.turn_history.clear()

                                    controller.last_escape_dir = None

                                self.pos_history.clear()

                                print(
                                    "[MODE SWITCH] "
                                    "Switched to Mode 2: "
                                    "Automatic Mode"
                                )

                            # -------------------------------------------------
                            # Mode 3
                            # -------------------------------------------------

                            elif event.key in (
                                pygame.K_3,
                                pygame.K_KP3
                            ):

                                self.mode = MODE_DESTINATION

                                self.client.set_mode(
                                    MODE_DESTINATION
                                )

                                print(
                                    "[MODE SWITCH] "
                                    "Switched to Mode 3: "
                                    "Destination Mode"
                                )

                            # -------------------------------------------------
                            # Reset
                            # -------------------------------------------------

                            elif event.key == pygame.K_r:

                                self.x = (
                                    ROOM_X0 + 35.0
                                )

                                self.y = (
                                    ROOM_Y0 + 35.0
                                )

                                self.theta = 0.0

                                self.v_l = 0.0
                                self.v_r = 0.0

                                self.trail.clear()

                                self.pos_history.clear()

                                self.is_stuck = False

                                # Reset controller
                                if self.client.mock_controller:

                                    controller = (
                                        self.client.mock_controller
                                    )

                                    controller.auto_state = (
                                        "FORWARD"
                                    )

                                    controller.state_timer = 0

                                    controller.turn_history.clear()

                                    controller.last_escape_dir = None

                # -----------------------------------------------------
                # Physics
                # -----------------------------------------------------

                self.step_physics(
                    SIM_DT
                )

                # -----------------------------------------------------
                # Render
                # -----------------------------------------------------

                if not self.headless:

                    self.draw()

                    self.clock.tick(
                        FPS
                    )

                else:

                    time.sleep(
                        SIM_DT
                    )

        finally:

            self.client.close()

            if not self.headless:

                pygame.quit()

def main():
    parser = argparse.ArgumentParser(description="ESP32 HIL Robot Simulation")
    parser.add_argument("--port", type=str, default=None, help="Serial COM port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate (default: 115200)")
    parser.add_argument("--wokwi", action="store_true", help="Connect to Wokwi simulation RFC2217 server (localhost:4000)")
    parser.add_argument("--mock", action="store_true", help="Force software mock mode (no ESP32 hardware needed)")
    parser.add_argument("--headless", action="store_true", help="Run without graphical window (for testing)")
    parser.add_argument("--duration", type=float, default=None, help="Run simulation for fixed duration in seconds")
    args = parser.parse_args()

    port = "rfc2217://localhost:4000" if args.wokwi else args.port
    client = SerialHILClient(port=port, baud=args.baud, mock=args.mock)
    sim = RobotSimulation(client, headless=args.headless)
    sim.run(max_seconds=args.duration)


if __name__ == "__main__":
    main()
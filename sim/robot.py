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
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700
FPS = 60
SIM_DT = 1.0 / FPS

# Cabin Room Dimensions (2.0 m x 2.0 m @ 100 px/m)
SCALE_PX_PER_M = 100.0

# Real cabin size
ROOM_SIZE = 200.0       # 2.0 m × 2.0 m in simulation coordinates

# Display scale: enlarge the cabin visually
DISPLAY_SCALE = 2.0

# 400 × 400 pixels on screen
DISPLAY_ROOM_SIZE = ROOM_SIZE * DISPLAY_SCALE

# Center the enlarged cabin on the 900 × 700 window
DISPLAY_ROOM_X0 = (WINDOW_WIDTH - DISPLAY_ROOM_SIZE) / 2
DISPLAY_ROOM_Y0 = (WINDOW_HEIGHT - DISPLAY_ROOM_SIZE) / 2

# Logical room coordinates remain 350–550 / 250–450
ROOM_X0 = (WINDOW_WIDTH - ROOM_SIZE) / 2.0
ROOM_Y0 = (WINDOW_HEIGHT - ROOM_SIZE) / 2.0

WHEEL_BASE = 20.0
ROBOT_RADIUS = 10.0
SENSOR_MAX_RANGE = 200.0


MODE_MANUAL = 1
MODE_AUTOMATIC = 2
MODE_DESTINATION = 3


class MockESP32Controller:
    """Software-in-the-loop fallback controller simulating the ESP32 firmware."""
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
        self.auto_turn_timer = 0

    def compute(self, x: float, y: float, theta: float, tgt_x: float, tgt_y: float, dist_front: float, mode: int = 1, man_l: float = 0.0, man_r: float = 0.0) -> Tuple[float, float]:
        self.mode = mode
        if mode == MODE_MANUAL:
            return man_l, man_r

        elif mode == MODE_AUTOMATIC:
            # Room Discovery / Autonomous Exploration Mode inside 2m x 2m cabin
            # Turn away when close to walls or obstacles, otherwise sweep forward
            if 0.0 < dist_front < 40.0 or x < ROOM_X0 + 20 or x > ROOM_X0 + ROOM_SIZE - 20 or y < ROOM_Y0 + 20 or y > ROOM_Y0 + ROOM_SIZE - 20:
                return -30.0, 60.0  # Pivot turn to discover unexplored room direction
            else:
                return 80.0, 80.0   # Drive forward mapping/discovering room

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

            if 0.0 < dist_front < 40.0:
                # Obstacle avoidance reflex
                return -20.0, 60.0

            angular_speed = (self.kp * error) + (self.kd * d_error)
            alignment = math.cos(error)

            if alignment > 0.0:
                linear_speed = self.max_speed * alignment
                if dist < 60.0:
                    linear_speed *= (dist / 60.0)
            else:
                linear_speed = 0.0

            left_cmd = linear_speed - (angular_speed * self.wheel_base / 2.0)
            right_cmd = linear_speed + (angular_speed * self.wheel_base / 2.0)

            left_cmd = max(-self.max_speed, min(self.max_speed, left_cmd))
            right_cmd = max(-self.max_speed, min(self.max_speed, right_cmd))
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

    def send_telemetry(self, x: float, y: float, theta: float, tgt_x: float, tgt_y: float, dist_front: float, mode: int = 1, man_l: float = 0.0, man_r: float = 0.0):
        """Send telemetry packet to ESP32 or process in mock."""
        if self.is_mock:
            self.left_cmd, self.right_cmd = self.mock_controller.compute(
                x, y, theta, tgt_x, tgt_y, dist_front, mode, man_l, man_r
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

        # Robot State & Mode (Starting in top-left open area of 2m x 2m cabin)
        self.mode = MODE_MANUAL  # Default to Mode 1 (Manual Mode)
        self.x = ROOM_X0 + 35.0  # 385.0 px (0.35 m from left wall)
        self.y = ROOM_Y0 + 35.0  # 285.0 px (0.35 m from top wall)
        self.theta = 0.0
        self.v_l = 0.0
        self.v_r = 0.0

        # Waypoint & Obstacles inside Cabin
        self.target_x = ROOM_X0 + 160.0 # 510.0 px (bottom-right open area)
        self.target_y = ROOM_Y0 + 160.0 # 410.0 px

        # -------------------------------------------------------------
        # Furniture Geometry (2.0 m x 2.0 m Cabin Layout @ 100 px/m)
        # Room Center = (ROOM_X0 + 100, ROOM_Y0 + 100) = (450, 350)
        # -------------------------------------------------------------
        cx = ROOM_X0 + 100.0
        cy = ROOM_Y0 + 100.0

        # Table: 50 cm x 30 cm centered at (cx, cy)
        self.table_rect = pygame.Rect(cx - 25, cy - 15, 50, 30)

        # 3 Chairs: 1 Chair on Top, 2 Chairs on Bottom (Opposite side)
        # Chair 1 (Top side)
        self.chair1_rect = pygame.Rect(cx - 8, cy - 35, 16, 16)
        # Chair 2 (Bottom side, left)
        self.chair2_rect = pygame.Rect(cx - 23, cy + 19, 16, 16)
        # Chair 3 (Bottom side, right)
        self.chair3_rect = pygame.Rect(cx + 7, cy + 19, 16, 16)

        # Leg Obstacles (Table and Chair Legs represented as circular physical obstacles)
        self.obstacles: List[Tuple[float, float, float]] = [
            # 4 Table Legs (Radius 2.5 px)
            (cx - 20.0, cy - 10.0, 2.5),
            (cx + 20.0, cy - 10.0, 2.5),
            (cx - 20.0, cy + 10.0, 2.5),
            (cx + 20.0, cy + 10.0, 2.5),

            # Chair 1 Legs (Top chair, 4 legs)
            (cx - 6.0, cy - 33.0, 2.0),
            (cx + 6.0, cy - 33.0, 2.0),
            (cx - 6.0, cy - 21.0, 2.0),
            (cx + 6.0, cy - 21.0, 2.0),

            # Chair 2 Legs (Bottom left chair, 4 legs)
            (cx - 21.0, cy + 21.0, 2.0),
            (cx - 9.0,  cy + 21.0, 2.0),
            (cx - 21.0, cy + 33.0, 2.0),
            (cx - 9.0,  cy + 33.0, 2.0),

            # Chair 3 Legs (Bottom right chair, 4 legs)
            (cx + 9.0,  cy + 21.0, 2.0),
            (cx + 21.0, cy + 21.0, 2.0),
            (cx + 9.0,  cy + 33.0, 2.0),
            (cx + 21.0, cy + 33.0, 2.0),
        ]

        self.trail: List[Tuple[float, float]] = []
        self.max_trail = 400
        self.front_dist = SENSOR_MAX_RANGE

        # Pygame setup
        if not self.headless:
            pygame.init()
            pygame.display.set_caption("ESP32 HIL Simulation - 2m x 2m Cabin Environment")
            self.screen = pygame.display.set_mode((int(WINDOW_WIDTH), int(WINDOW_HEIGHT)))
            self.clock = pygame.time.Clock()
            self.font_small = pygame.font.Font(None, 18)
            self.font_bold = pygame.font.Font(None, 20)
            self.font_title = pygame.font.Font(None, 24)

    def cast_distance_sensor(self) -> float:
        """Raycast forward along robot heading to measure distance to cabin walls or leg obstacles."""
        sensor_dist = SENSOR_MAX_RANGE
        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)

        # 1. Raycast against 4 Cabin Walls
        if cos_t > 0:
            sensor_dist = min(sensor_dist, ((ROOM_X0 + ROOM_SIZE) - self.x) / cos_t)
        elif cos_t < 0:
            sensor_dist = min(sensor_dist, (ROOM_X0 - self.x) / cos_t)

        if sin_t > 0:
            sensor_dist = min(sensor_dist, ((ROOM_Y0 + ROOM_SIZE) - self.y) / sin_t)
        elif sin_t < 0:
            sensor_dist = min(sensor_dist, (ROOM_Y0 - self.y) / sin_t)

        # 2. Raycast against Furniture Leg Obstacles (circle intersection)
        for ox, oy, r in self.obstacles:
            dx = ox - self.x
            dy = oy - self.y
            proj = dx * cos_t + dy * sin_t
            if proj > 0:
                perp_sq = (dx * dx + dy * dy) - (proj * proj)
                if perp_sq < r * r:
                    chord_dist = math.sqrt(max(0.0, r * r - perp_sq))
                    hit_dist = proj - chord_dist
                    if 0 < hit_dist < sensor_dist:
                        sensor_dist = hit_dist

        return max(0.0, sensor_dist)

    def step_physics(self, dt: float):
        # Determine WASD manual speed inputs if in Mode 1 (Manual Mode)
        man_l, man_r = 0.0, 0.0
        if self.mode == MODE_MANUAL and not self.headless:
            keys = pygame.key.get_pressed()
            speed = 100.0
            turn_spd = 70.0
            fw = keys[pygame.K_w]
            bw = keys[pygame.K_s]
            lt = keys[pygame.K_a]
            rt = keys[pygame.K_d]

            if fw:
                if lt:
                    man_l, man_r = 30.0, 100.0
                elif rt:
                    man_l, man_r = 100.0, 30.0
                else:
                    man_l, man_r = speed, speed
            elif bw:
                if lt:
                    man_l, man_r = -30.0, -100.0
                elif rt:
                    man_l, man_r = -100.0, -30.0
                else:
                    man_l, man_r = -speed, -speed
            elif lt:
                man_l, man_r = -turn_spd, turn_spd
            elif rt:
                man_l, man_r = turn_spd, -turn_spd

        # Update distance sensor reading
        self.front_dist = self.cast_distance_sensor()

        # Send telemetry & manual inputs back to ESP32 / Mock
        self.client.send_telemetry(
            self.x, self.y, self.theta, 
            self.target_x, self.target_y, 
            self.front_dist, 
            self.mode, man_l, man_r
        )

        # Update motor commands from HIL client response
        self.v_l = self.client.left_cmd
        self.v_r = self.client.right_cmd

        # Differential drive forward kinematics
        v = (self.v_l + self.v_r) / 2.0
        w = (self.v_r - self.v_l) / WHEEL_BASE

        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += w * dt

        # Wrap angle
        while self.theta > math.pi:
            self.theta -= 2.0 * math.pi
        while self.theta < -math.pi:
            self.theta += 2.0 * math.pi

        # 1. Cabin Wall Position Clamping
        self.x = max(ROOM_X0 + ROBOT_RADIUS, min(ROOM_X0 + ROOM_SIZE - ROBOT_RADIUS, self.x))
        self.y = max(ROOM_Y0 + ROBOT_RADIUS, min(ROOM_Y0 + ROOM_SIZE - ROBOT_RADIUS, self.y))

        # 2. Furniture Leg Obstacle Physical Collision Resolution
        for ox, oy, r in self.obstacles:
            dx = self.x - ox
            dy = self.y - oy
            dist = math.hypot(dx, dy)
            min_dist = r + ROBOT_RADIUS
            if 0 < dist < min_dist:
                overlap = min_dist - dist
                self.x += (dx / dist) * overlap
                self.y += (dy / dist) * overlap

        # Update trail
        if not self.trail or math.hypot(self.x - self.trail[-1][0], self.y - self.trail[-1][1]) > 4.0:
            self.trail.append((self.x, self.y))
            if len(self.trail) > self.max_trail:
                self.trail.pop(0)

    def draw(self):
        # ---------------------------------------------------------------
        # Visual scale mapping
        # Logic:  100 px = 1 m  (physics / collision coordinate space)
        # Visual: 200 px = 1 m  → VISUAL_SCALE = 2.0
        #         2.0 m room   → 400 x 400 px displayed on screen
        # HUD_HEIGHT: 60 px reserved at the very top of the 900x700 window.
        # Remaining usable height: 640 px → cabin visual center at y = 60 + 320 = 380
        # Cabin visual center horizontally at x = 900/2 = 450
        # ---------------------------------------------------------------
        VISUAL_SCALE = 2.0
        HUD_HEIGHT   = 60          # px reserved for the top HUD bar
        ROOM_CX = ROOM_X0 + ROOM_SIZE / 2.0   # logical center of room
        ROOM_CY = ROOM_Y0 + ROOM_SIZE / 2.0
        WIN_CX  = WINDOW_WIDTH  / 2.0                               # 450 px
        WIN_CY  = HUD_HEIGHT + (WINDOW_HEIGHT - HUD_HEIGHT) / 2.0  # 380 px

        # Helper: map a logical coordinate → screen pixel
        def s_x(lx): return WIN_CX + (lx - ROOM_CX) * VISUAL_SCALE
        def s_y(ly): return WIN_CY + (ly - ROOM_CY) * VISUAL_SCALE
        def s_r(lr): return max(1, int(lr * VISUAL_SCALE))
        def s_rect(rect):
            return pygame.Rect(
                int(s_x(rect.left)), int(s_y(rect.top)),
                int(s_r(rect.width)), int(s_r(rect.height)))

        def sc(surface, color, lx, ly, lr, width=0):
            """Scaled circle draw."""
            pygame.draw.circle(surface, color,
                               (int(s_x(lx)), int(s_y(ly))), s_r(lr), width)

        def sl(surface, color, lp1, lp2, width=1):
            """Scaled line draw."""
            pygame.draw.line(surface, color,
                             (int(s_x(lp1[0])), int(s_y(lp1[1]))),
                             (int(s_x(lp2[0])), int(s_y(lp2[1]))), width)

        # 1. Dark background outside cabin
        self.screen.fill((18, 22, 28))

        # 2. Cabin floor (warm wood tone)
        room_vis = s_rect(pygame.Rect(ROOM_X0, ROOM_Y0, ROOM_SIZE, ROOM_SIZE))
        pygame.draw.rect(self.screen, (52, 42, 34), room_vis)

        # Floor planks — horizontal lines every 25 px logic (25 cm)
        for gy in range(int(ROOM_Y0), int(ROOM_Y0 + ROOM_SIZE) + 1, 25):
            sl(self.screen, (66, 54, 44), (ROOM_X0, gy), (ROOM_X0 + ROOM_SIZE, gy), 1)
        # Faint vertical grain every 50 px logic (50 cm)
        for gx in range(int(ROOM_X0), int(ROOM_X0 + ROOM_SIZE) + 1, 50):
            sl(self.screen, (60, 49, 40), (gx, ROOM_Y0), (gx, ROOM_Y0 + ROOM_SIZE), 1)

        # 3. Cabin walls (thick border + highlight)
        pygame.draw.rect(self.screen, (160, 110, 60), room_vis, 10)   # thick wall
        pygame.draw.rect(self.screen, (210, 160, 100), room_vis, 2)   # inner highlight
        pygame.draw.rect(self.screen, (100, 70, 35), room_vis, 1)     # outer shadow

        # Room label (top-left corner inside cabin)
        lbl = self.font_small.render("2.0 m", True, (160, 130, 80))
        self.screen.blit(lbl, (room_vis.left + 4, room_vis.top + 4))

        # 4. Robot trail
        if len(self.trail) > 1:
            trail_color = (60, 200, 180) if self.mode == MODE_AUTOMATIC else (60, 130, 200)
            scaled_trail = [(int(s_x(tx)), int(s_y(ty))) for tx, ty in self.trail]
            pygame.draw.lines(self.screen, trail_color, False, scaled_trail, 2)

        # 5. Furniture
        # --- Chair 1 (top side — backrest strip at top edge)
        v_c1 = s_rect(self.chair1_rect)
        pygame.draw.rect(self.screen, (45, 100, 160), v_c1, border_radius=4)
        pygame.draw.rect(self.screen, (80, 150, 210), v_c1, 2, border_radius=4)
        pygame.draw.line(self.screen, (130, 200, 255),
                         (v_c1.left + 3, v_c1.top + 3),
                         (v_c1.right - 3, v_c1.top + 3), 4)  # backrest bar

        # --- Chair 2 (bottom-left — backrest strip at bottom edge)
        v_c2 = s_rect(self.chair2_rect)
        pygame.draw.rect(self.screen, (45, 100, 160), v_c2, border_radius=4)
        pygame.draw.rect(self.screen, (80, 150, 210), v_c2, 2, border_radius=4)
        pygame.draw.line(self.screen, (130, 200, 255),
                         (v_c2.left + 3, v_c2.bottom - 3),
                         (v_c2.right - 3, v_c2.bottom - 3), 4)

        # --- Chair 3 (bottom-right — backrest strip at bottom edge)
        v_c3 = s_rect(self.chair3_rect)
        pygame.draw.rect(self.screen, (45, 100, 160), v_c3, border_radius=4)
        pygame.draw.rect(self.screen, (80, 150, 210), v_c3, 2, border_radius=4)
        pygame.draw.line(self.screen, (130, 200, 255),
                         (v_c3.left + 3, v_c3.bottom - 3),
                         (v_c3.right - 3, v_c3.bottom - 3), 4)

        # --- Table top (warm brown, centered)
        v_tbl = s_rect(self.table_rect)
        pygame.draw.rect(self.screen, (140, 80, 35), v_tbl, border_radius=5)
        pygame.draw.rect(self.screen, (200, 130, 60), v_tbl, 2, border_radius=5)
        # subtle wood grain on table
        for gi in range(3):
            gy_tbl = v_tbl.top + v_tbl.height * (gi + 1) // 4
            pygame.draw.line(self.screen, (155, 95, 45),
                             (v_tbl.left + 4, gy_tbl),
                             (v_tbl.right - 4, gy_tbl), 1)

        # 6. Leg obstacles
        for ox, oy, r in self.obstacles:
            sc(self.screen, (200, 55, 55), ox, oy, r + 1)        # dark red fill
            sc(self.screen, (240, 110, 110), ox, oy, r + 1, 1)   # bright red ring

        # 7. Target waypoint (Mode 3)
        if self.mode == MODE_DESTINATION:
            pulse = 4 * math.sin(time.time() * 6.0)
            sc(self.screen, (255, 180, 50), self.target_x, self.target_y, 10 + pulse, 2)
            sc(self.screen, (255, 210, 80), self.target_x, self.target_y, 3)

        # 8. Forward sensor ray
        sensor_end_x = self.x + self.front_dist * math.cos(self.theta)
        sensor_end_y = self.y + self.front_dist * math.sin(self.theta)
        sl(self.screen, (200, 55, 55),
           (self.x, self.y), (sensor_end_x, sensor_end_y), 1)
        sc(self.screen, (255, 60, 60), sensor_end_x, sensor_end_y, 2)

        # 9. Robot body
        sc(self.screen, (25, 35, 50), self.x, self.y, ROBOT_RADIUS)          # shadow
        sc(self.screen, (35, 150, 215), self.x, self.y, ROBOT_RADIUS - 1)    # body fill
        sc(self.screen, (190, 235, 255), self.x, self.y, ROBOT_RADIUS, 2)    # rim

        # Front bumper arc
        b_ang = 0.75
        p_left  = (self.x + ROBOT_RADIUS * math.cos(self.theta - b_ang),
                   self.y + ROBOT_RADIUS * math.sin(self.theta - b_ang))
        p_front = (self.x + (ROBOT_RADIUS + 1.5) * math.cos(self.theta),
                   self.y + (ROBOT_RADIUS + 1.5) * math.sin(self.theta))
        p_right = (self.x + ROBOT_RADIUS * math.cos(self.theta + b_ang),
                   self.y + ROBOT_RADIUS * math.sin(self.theta + b_ang))
        scaled_bumper = [(int(s_x(px)), int(s_y(py)))
                         for px, py in [p_left, p_front, p_right]]
        pygame.draw.lines(self.screen, (230, 60, 60), False, scaled_bumper, 2)

        # Vacuum vortex indicator
        vac_color = (0, 230, 255) if self.client.is_vacuum_on else (80, 90, 100)
        sc(self.screen, vac_color, self.x, self.y, 4)
        if self.client.is_vacuum_on:
            sc(self.screen, (255, 255, 255), self.x, self.y, 1)

        # Spinning side sweeper brush
        brush_base_ang = self.theta + 0.65
        bx_l = self.x + (ROBOT_RADIUS - 2) * math.cos(brush_base_ang)
        by_l = self.y + (ROBOT_RADIUS - 2) * math.sin(brush_base_ang)
        spin = (time.time() * 14.0) % (2.0 * math.pi) if self.client.is_vacuum_on else 0.0
        for b_sub in [spin, spin + 2.094, spin + 4.188]:
            br_end_x = bx_l + 5 * math.cos(b_sub)
            br_end_y = by_l + 5 * math.sin(b_sub)
            sl(self.screen, (255, 215, 60), (bx_l, by_l), (br_end_x, br_end_y), 2)

        # Differential drive wheels
        perp     = self.theta + math.pi / 2.0
        w_offset = WHEEL_BASE / 2.0
        for side in [-1, 1]:
            wx = self.x + side * w_offset * math.cos(perp)
            wy = self.y + side * w_offset * math.sin(perp)
            w_len, w_th = 8, 3
            p1 = (wx - (w_len/2)*math.cos(self.theta) - (w_th/2)*math.sin(self.theta),
                  wy - (w_len/2)*math.sin(self.theta) + (w_th/2)*math.cos(self.theta))
            p2 = (wx + (w_len/2)*math.cos(self.theta) - (w_th/2)*math.sin(self.theta),
                  wy + (w_len/2)*math.sin(self.theta) + (w_th/2)*math.cos(self.theta))
            p3 = (wx + (w_len/2)*math.cos(self.theta) + (w_th/2)*math.sin(self.theta),
                  wy + (w_len/2)*math.sin(self.theta) - (w_th/2)*math.cos(self.theta))
            p4 = (wx - (w_len/2)*math.cos(self.theta) + (w_th/2)*math.sin(self.theta),
                  wy - (w_len/2)*math.sin(self.theta) - (w_th/2)*math.cos(self.theta))
            scaled_wheel = [(int(s_x(px)), int(s_y(py)))
                            for px, py in [p1, p2, p3, p4]]
            pygame.draw.polygon(self.screen, (18, 18, 18), scaled_wheel)
            pygame.draw.polygon(self.screen, (90, 90, 90), scaled_wheel, 1)

        # Heading indicator
        head_x = self.x + (ROBOT_RADIUS + 4) * math.cos(self.theta)
        head_y = self.y + (ROBOT_RADIUS + 4) * math.sin(self.theta)
        sl(self.screen, (255, 255, 80), (self.x, self.y), (head_x, head_y), 2)

        # 10. HUD (top bar, never overlaps cabin)
        self._render_hud()

        pygame.display.flip()

    def _render_hud(self):
        # HUD occupies a 58 px horizontal strip at the very top of the window.
        HUD_H = 58
        hud_surf = pygame.Surface((WINDOW_WIDTH, HUD_H), pygame.SRCALPHA)
        hud_surf.fill((10, 15, 22, 245))
        self.screen.blit(hud_surf, (0, 0))
        pygame.draw.line(self.screen, (55, 75, 105),
                         (0, HUD_H - 1), (WINDOW_WIDTH, HUD_H - 1), 1)

        badge_color = (80, 220, 100) if not self.client.is_mock else (240, 160, 40)
        mode_labels = {
            MODE_MANUAL:      "MANUAL (WASD)",
            MODE_AUTOMATIC:   "AUTO (Room Discovery)",
            MODE_DESTINATION: "DESTINATION (Waypoint PID)",
        }
        mode_colors = {
            MODE_MANUAL:      (0, 230, 255),
            MODE_AUTOMATIC:   (100, 255, 120),
            MODE_DESTINATION: (255, 180, 50),
        }

        rel_x_m = (self.x - ROOM_X0) / SCALE_PX_PER_M
        rel_y_m = (self.y - ROOM_Y0) / SCALE_PX_PER_M

        # --- Row 1: Title | Mode | Status (right-aligned)
        title_s  = self.font_title.render("2m×2m CABIN ROBOT SIM",    True, (255, 255, 255))
        mode_s   = self.font_bold.render(
            f"[{mode_labels[self.mode]}]", True, mode_colors[self.mode])
        status_s = self.font_bold.render(self.client.status_msg, True, badge_color)

        self.screen.blit(title_s,  (10,  5))
        self.screen.blit(mode_s,   (10 + title_s.get_width() + 14, 5))
        self.screen.blit(status_s, (WINDOW_WIDTH - status_s.get_width() - 10, 5))

        # --- Row 2: Pose | Sensor | Motors | Controls hint
        pose_s = self.font_small.render(
            f"Pose  X={rel_x_m:.2f}m  Y={rel_y_m:.2f}m  θ={math.degrees(self.theta):.1f}°",
            True, (200, 220, 240))
        sens_s = self.font_small.render(
            f"Sensor {self.front_dist / SCALE_PX_PER_M:.2f} m",
            True, (250, 140, 140))
        mot_s  = self.font_small.render(
            f"Motors  L={self.v_l:.0f}  R={self.v_r:.0f}",
            True, (140, 240, 160))
        hint_s = self.font_small.render(
            "[1] Manual  [2] Auto  [3] Dest  |  WASD drive  |  R reset",
            True, (200, 200, 130))

        row2_y = 33
        for surf, col_x in zip([pose_s, sens_s, mot_s, hint_s], [10, 285, 445, 592]):
            self.screen.blit(surf, (col_x, row2_y))

    def run(self, max_seconds: Optional[float] = None):
        start_time = time.time()
        running = True

        try:
            while running:
                if max_seconds and (time.time() - start_time) > max_seconds:
                    break

                if not self.headless:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        elif event.type == pygame.MOUSEBUTTONDOWN:
                            mx, my = pygame.mouse.get_pos()
                            if event.button == 1:  # Left click: Set target
                                self.target_x = float(mx)
                                self.target_y = float(my)
                        elif event.type == pygame.KEYDOWN:
                            if event.key in (pygame.K_1, pygame.K_KP1):
                                self.mode = MODE_MANUAL
                                self.client.set_mode(MODE_MANUAL)
                                print("[MODE SWITCH] Switched to Mode 1: Manual Mode (WASD)")
                            elif event.key in (pygame.K_2, pygame.K_KP2):
                                self.mode = MODE_AUTOMATIC
                                self.client.set_mode(MODE_AUTOMATIC)
                                print("[MODE SWITCH] Switched to Mode 2: Automatic Mode (Room Discovery)")
                            elif event.key in (pygame.K_3, pygame.K_KP3):
                                self.mode = MODE_DESTINATION
                                self.client.set_mode(MODE_DESTINATION)
                                print("[MODE SWITCH] Switched to Mode 3: Destination Mode")
                            elif event.key == pygame.K_r:  # Reset robot position to open start area
                                self.x = ROOM_X0 + 35.0
                                self.y = ROOM_Y0 + 35.0
                                self.theta = 0.0
                                self.trail.clear()

                self.step_physics(SIM_DT)

                if not self.headless:
                    self.draw()
                    self.clock.tick(FPS)
                else:
                    time.sleep(SIM_DT)
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


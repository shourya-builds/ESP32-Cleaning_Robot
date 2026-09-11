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


# Simulation Constants
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700
FPS = 60
SIM_DT = 1.0 / FPS

WHEEL_BASE = 40.0       # Distance between wheels (pixels)
ROBOT_RADIUS = 25.0     # Visual radius of robot body
SENSOR_MAX_RANGE = 250.0  # Max distance sensor range


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
        self.waypoint_tolerance = 15.0
        self.is_vacuum_on = True
        self.battery_pct = 95
        self.mode = MODE_MANUAL
        self.auto_turn_timer = 0

    def compute(self, x: float, y: float, theta: float, tgt_x: float, tgt_y: float, dist_front: float, mode: int = 1, man_l: float = 0.0, man_r: float = 0.0) -> Tuple[float, float]:
        self.mode = mode
        if mode == MODE_MANUAL:
            return man_l, man_r

        elif mode == MODE_AUTOMATIC:
            # Room Discovery / Autonomous Exploration Mode
            # Turn away when close to walls or obstacles, otherwise sweep forward
            if 0.0 < dist_front < 60.0 or x < 35 or x > WINDOW_WIDTH - 35 or y < 35 or y > WINDOW_HEIGHT - 35:
                return -30.0, 60.0  # Pivot turn to discover unexplored room direction
            else:
                return 85.0, 85.0   # Drive forward mapping/discovering room

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

            if 0.0 < dist_front < 60.0:
                # Obstacle avoidance reflex
                return -20.0, 60.0

            angular_speed = (self.kp * error) + (self.kd * d_error)
            alignment = math.cos(error)

            if alignment > 0.0:
                linear_speed = self.max_speed * alignment
                if dist < 80.0:
                    linear_speed *= (dist / 80.0)
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

        # Robot State & Mode
        self.mode = MODE_MANUAL  # Default to Mode 1 (Manual Mode)
        self.x = 200.0
        self.y = 350.0
        self.theta = 0.0
        self.v_l = 0.0
        self.v_r = 0.0

        # Waypoint & Obstacles
        self.target_x = 700.0
        self.target_y = 350.0
        self.obstacles: List[Tuple[float, float, float]] = [
            (450.0, 350.0, 35.0), # (x, y, radius)
        ]

        self.trail: List[Tuple[float, float]] = []
        self.max_trail = 400
        self.front_dist = SENSOR_MAX_RANGE

        # Pygame setup
        if not self.headless:
            pygame.init()
            pygame.display.set_caption("ESP32 Hardware-in-the-Loop Simulation - 3-Mode Robot Controller")
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            self.clock = pygame.time.Clock()
            self.font_small = pygame.font.Font(None, 18)
            self.font_bold = pygame.font.Font(None, 20)
            self.font_title = pygame.font.Font(None, 24)

    def cast_distance_sensor(self) -> float:
        """Raycast forward along robot heading to measure distance to boundaries or obstacles."""
        sensor_dist = SENSOR_MAX_RANGE
        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)

        # 1. Screen boundaries
        if cos_t > 0:
            sensor_dist = min(sensor_dist, (WINDOW_WIDTH - self.x) / cos_t)
        elif cos_t < 0:
            sensor_dist = min(sensor_dist, -self.x / cos_t)

        if sin_t > 0:
            sensor_dist = min(sensor_dist, (WINDOW_HEIGHT - self.y) / sin_t)
        elif sin_t < 0:
            sensor_dist = min(sensor_dist, -self.y / sin_t)

        # 2. Obstacles (circle intersection)
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

        # Keep within boundaries
        self.x = max(ROBOT_RADIUS, min(WINDOW_WIDTH - ROBOT_RADIUS, self.x))
        self.y = max(ROBOT_RADIUS, min(WINDOW_HEIGHT - ROBOT_RADIUS, self.y))

        # Update trail
        if not self.trail or math.hypot(self.x - self.trail[-1][0], self.y - self.trail[-1][1]) > 4.0:
            self.trail.append((self.x, self.y))
            if len(self.trail) > self.max_trail:
                self.trail.pop(0)

    def draw(self):
        self.screen.fill((20, 24, 30))  # Dark slate background

        # 1. Grid lines
        for gx in range(0, WINDOW_WIDTH, 50):
            pygame.draw.line(self.screen, (30, 36, 46), (gx, 0), (gx, WINDOW_HEIGHT), 1)
        for gy in range(0, WINDOW_HEIGHT, 50):
            pygame.draw.line(self.screen, (30, 36, 46), (0, gy), (WINDOW_WIDTH, gy), 1)

        # 2. Trail (Room Discovery path visualization)
        if len(self.trail) > 1:
            trail_color = (60, 200, 180) if self.mode == MODE_AUTOMATIC else (60, 130, 180)
            pygame.draw.lines(self.screen, trail_color, False, self.trail, 2)

        # 3. Obstacles
        for ox, oy, r in self.obstacles:
            pygame.draw.circle(self.screen, (180, 70, 70), (int(ox), int(oy)), int(r))
            pygame.draw.circle(self.screen, (230, 100, 100), (int(ox), int(oy)), int(r), 2)

        # 4. Target Waypoint (Active in Mode 3)
        if self.mode == MODE_DESTINATION:
            pulse = 4 * math.sin(time.time() * 6.0)
            pygame.draw.circle(self.screen, (255, 180, 50), (int(self.target_x), int(self.target_y)), int(12 + pulse), 2)
            pygame.draw.circle(self.screen, (255, 200, 80), (int(self.target_x), int(self.target_y)), 4)
            pygame.draw.line(self.screen, (255, 180, 50, 100), 
                             (int(self.target_x) - 16, int(self.target_y)), (int(self.target_x) + 16, int(self.target_y)), 1)
            pygame.draw.line(self.screen, (255, 180, 50, 100), 
                             (int(self.target_x), int(self.target_y) - 16), (int(self.target_x), int(self.target_y) + 16), 1)

        # 5. Distance sensor ray
        sensor_end_x = self.x + self.front_dist * math.cos(self.theta)
        sensor_end_y = self.y + self.front_dist * math.sin(self.theta)
        pygame.draw.line(self.screen, (240, 70, 70), (int(self.x), int(self.y)), (int(sensor_end_x), int(sensor_end_y)), 1)
        pygame.draw.circle(self.screen, (255, 50, 50), (int(sensor_end_x), int(sensor_end_y)), 4)

        # 6. Robot Body (Cleaning Robot Chassis)
        rx, ry = int(self.x), int(self.y)
        pygame.draw.circle(self.screen, (30, 40, 55), (rx, ry), int(ROBOT_RADIUS))
        pygame.draw.circle(self.screen, (40, 160, 220), (rx, ry), int(ROBOT_RADIUS - 3))
        pygame.draw.circle(self.screen, (200, 240, 255), (rx, ry), int(ROBOT_RADIUS), 2)

        # Front Bumper Bar (tactile bumper arc)
        b_ang = 0.75
        p_left = (rx + ROBOT_RADIUS * math.cos(self.theta - b_ang),
                  ry + ROBOT_RADIUS * math.sin(self.theta - b_ang))
        p_front = (rx + (ROBOT_RADIUS + 4) * math.cos(self.theta),
                   ry + (ROBOT_RADIUS + 4) * math.sin(self.theta))
        p_right = (rx + ROBOT_RADIUS * math.cos(self.theta + b_ang),
                   ry + ROBOT_RADIUS * math.sin(self.theta + b_ang))
        pygame.draw.lines(self.screen, (240, 70, 70), False, [p_left, p_front, p_right], 3)

        # Center Vacuum Suction Vortex
        vac_color = (0, 240, 255) if self.client.is_vacuum_on else (90, 100, 110)
        pygame.draw.circle(self.screen, vac_color, (rx, ry), 7)
        if self.client.is_vacuum_on:
            pygame.draw.circle(self.screen, (255, 255, 255), (rx, ry), 3)

        # Spinning Side Sweeper Brush
        brush_base_ang = self.theta + 0.65
        bx = rx + (ROBOT_RADIUS - 4) * math.cos(brush_base_ang)
        by = ry + (ROBOT_RADIUS - 4) * math.sin(brush_base_ang)
        spin = (time.time() * 14.0) % (2.0 * math.pi) if self.client.is_vacuum_on else 0.0
        for b_sub in [spin, spin + 2.094, spin + 4.188]:
            br_end_x = bx + 9 * math.cos(b_sub)
            br_end_y = by + 9 * math.sin(b_sub)
            pygame.draw.line(self.screen, (255, 220, 70), (bx, by), (br_end_x, br_end_y), 2)
        pygame.draw.circle(self.screen, (50, 50, 50), (int(bx), int(by)), 3)

        # Wheels
        perp = self.theta + math.pi / 2.0
        w_offset = WHEEL_BASE / 2.0
        for side in [-1, 1]:
            wx = self.x + side * w_offset * math.cos(perp)
            wy = self.y + side * w_offset * math.sin(perp)
            # Wheel rectangle aligned with heading
            w_len, w_th = 16, 6
            p1 = (wx - (w_len/2)*math.cos(self.theta) - (w_th/2)*math.sin(self.theta),
                  wy - (w_len/2)*math.sin(self.theta) + (w_th/2)*math.cos(self.theta))
            p2 = (wx + (w_len/2)*math.cos(self.theta) - (w_th/2)*math.sin(self.theta),
                  wy + (w_len/2)*math.sin(self.theta) + (w_th/2)*math.cos(self.theta))
            p3 = (wx + (w_len/2)*math.cos(self.theta) + (w_th/2)*math.sin(self.theta),
                  wy + (w_len/2)*math.sin(self.theta) - (w_th/2)*math.cos(self.theta))
            p4 = (wx - (w_len/2)*math.cos(self.theta) + (w_th/2)*math.sin(self.theta),
                  wy - (w_len/2)*math.sin(self.theta) - (w_th/2)*math.cos(self.theta))
            pygame.draw.polygon(self.screen, (20, 20, 20), [p1, p2, p3, p4])
            pygame.draw.polygon(self.screen, (100, 100, 100), [p1, p2, p3, p4], 1)

        # Heading indicator arrow
        head_x = self.x + (ROBOT_RADIUS + 8) * math.cos(self.theta)
        head_y = self.y + (ROBOT_RADIUS + 8) * math.sin(self.theta)
        pygame.draw.line(self.screen, (255, 255, 100), (rx, ry), (int(head_x), int(head_y)), 3)

        # 7. Telemetry & Status HUD Overlay
        self._render_hud()

        pygame.display.flip()

    def _render_hud(self):
        panel_rect = pygame.Rect(15, 15, 380, 235)
        panel_surface = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
        panel_surface.fill((10, 15, 22, 235))
        self.screen.blit(panel_surface, panel_rect.topleft)
        pygame.draw.rect(self.screen, (60, 80, 100), panel_rect, 1, border_radius=6)

        # Status badge color
        badge_color = (80, 220, 100) if not self.client.is_mock else (240, 160, 40)
        vac_txt = "ACTIVE (Suction + Brush ON)" if self.client.is_vacuum_on else "OFF (Idle/Docked)"
        vac_color = (0, 240, 255) if self.client.is_vacuum_on else (160, 160, 160)

        mode_titles = {
            MODE_MANUAL: "MODE 1: MANUAL MODE (WASD)",
            MODE_AUTOMATIC: "MODE 2: AUTOMATIC MODE (Room Discovery)",
            MODE_DESTINATION: "MODE 3: DESTINATION MODE (Waypoint PID)"
        }
        mode_colors = {
            MODE_MANUAL: (0, 230, 255),
            MODE_AUTOMATIC: (100, 255, 120),
            MODE_DESTINATION: (255, 180, 50)
        }

        lines = [
            ("CLEANING ROBOT HIL SIMULATOR", (255, 255, 255), self.font_title),
            (mode_titles[self.mode], mode_colors[self.mode], self.font_bold),
            (f"Status: {self.client.status_msg}", badge_color, self.font_bold),
            (f"Cleaning: {vac_txt}", vac_color, self.font_small),
            (f"FPS: {self.clock.get_fps():.1f} | Telemetry Pkts: {self.client.packet_count}", (180, 190, 200), self.font_small),
            (f"Robot Pose : X={self.x:.1f}, Y={self.y:.1f}, Th={math.degrees(self.theta):.1f}°", (200, 220, 240), self.font_small),
            (f"Target Waypoint: X={self.target_x:.1f}, Y={self.target_y:.1f}", (240, 220, 120), self.font_small),
            (f"Front Sensor   : {self.front_dist:.1f} px", (250, 140, 140), self.font_small),
            (f"Motor Cmds (L/R): {self.v_l:.1f} / {self.v_r:.1f}", (140, 240, 160), self.font_bold),
            ("Modes: Press [1] Manual | [2] Auto | [3] Destination", (255, 255, 180), self.font_bold),
            ("Manual Mode: Use W A S D keys to drive robot", (160, 200, 240), self.font_small),
        ]

        y_offset = 20
        for text, color, font in lines:
            rendered = font.render(text, True, color)
            self.screen.blit(rendered, (25, y_offset))
            y_offset += rendered.get_height() + 2

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
                            elif event.button == 3:  # Right click: Add obstacle
                                self.obstacles.append((float(mx), float(my), 25.0))
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
                            elif event.key == pygame.K_r:  # Reset
                                self.x = 200.0
                                self.y = 350.0
                                self.theta = 0.0
                                self.trail.clear()
                            elif event.key == pygame.K_o:  # Add obstacle at mouse
                                mx, my = pygame.mouse.get_pos()
                                self.obstacles.append((float(mx), float(my), 25.0))
                            elif event.key == pygame.K_c:  # Clear obstacles
                                self.obstacles.clear()

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


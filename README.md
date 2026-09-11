# Autonomous Cleaning Robot ESP32 HIL Simulation

A Hardware-in-the-Loop (HIL) robotics simulation testbed connecting a physical or Wokwi-simulated ESP32 microcontroller with a 2D autonomous cleaning robot simulator built in Pygame.

## Architecture Overview

```
+-------------------------------------------------------------+
|              Pygame Cleaning Robot Simulator                |
|                      (sim/robot.py)                         |
|  - Differential drive kinematics & cleaning payload display |
|  - Suction fan vortex & rotating side sweeper animation     |
|  - Forward distance sensor (raycasting)                     |
|  - Interactive waypoint selection & obstacle placement      |
+-------------------------------------------------------------+
                            |   ^
   Sensor Telemetry (Serial)|   | Actuator Commands (Serial)
   "TELEM,x,y,theta,..."    |   | "CMD,v_left,v_right\n"
   (115200 Baud)            v   |
+-------------------------------------------------------------+
|              ESP32 Cleaning Robot Controller                |
|                       (src/main.cpp)                        |
|  - Locomotion: L298N Differential Motor Driver (PWM/DIR)    |
|  - Payload: Vacuum Suction Relay & Side Sweeper Brush       |
|  - Perception: 3x HC-SR04 Ultrasonic (Front, Left, Right)   |
|  - Safety Reflexes: Front Bumper Contact & Cliff Drop-off   |
|  - Navigation: MPU-6050 6-Axis IMU (Yaw tracking)           |
|  - Telemetry HUD: SSD1306 128x64 I2C OLED Dashboard         |
|  - Power: Battery Voltage ADC monitor & Buzzer chimes       |
|  - Failsafe watchdog timer (1000 ms timeout)                |
+-------------------------------------------------------------+
```

### Component & Pin Allocation Table

| Component | Function | ESP32 Pin | Interface / Mode |
| :--- | :--- | :--- | :--- |
| **SSD1306 OLED (128x64)** | Real-time Dashboard Display | GPIO 21 (SDA), GPIO 22 (SCL) | I2C (0x3C) |
| **MPU-6050 IMU** | Gyro Yaw & Accelerometer | GPIO 21 (SDA), GPIO 22 (SCL) | I2C (0x68) |
| **L298N Motor Driver** | Left & Right Wheel Motors | GPIO 25, 26, 27, 14 | Digital Output / PWM |
| **Vacuum Suction Relay** | Dust Impeller Fan On/Off | GPIO 23 | Digital Output |
| **Side Sweeper Brush** | Corner & Edge Brush | GPIO 13 | Servo / PWM Output |
| **Front Collision Bumper** | Tactile Obstacle Switch | GPIO 32 | Digital Input (PULLUP) |
| **Cliff / Drop-off Sensor** | Staircase Edge Detection | GPIO 33 | Digital Input (PULLUP) |
| **Battery Monitor** | Real-time Voltage Monitor | GPIO 34 | Analog ADC (ADC1_CH6) |
| **Clean / Pause Button** | Top-panel User Control | GPIO 15 | Digital Input (PULLUP) |
| **Piezo Buzzer** | Audio Alerts & State Chimes | GPIO 12 | PWM / Tone Output |
| **Status LED** | System Cleaning Health | GPIO 2 | Digital Output |
| **Ultrasonic Front** | Long-range Rangefinder | GPIO 5 (TRIG), GPIO 18 (ECHO) | Digital Pulse |
| **Ultrasonic Left** | Wall-following Rangefinder | GPIO 19 (TRIG), GPIO 4 (ECHO) | Digital Pulse |
| **Ultrasonic Right** | Clearance Rangefinder | GPIO 17 (TRIG), GPIO 16 (ECHO) | Digital Pulse |


## Directory Structure

```
esp32-hil-sim/
├── platformio.ini       # PlatformIO ESP32 configuration
├── requirements.txt     # Python dependencies (pygame, pyserial)
├── README.md            # Documentation & setup guide
├── src/
│   └── main.cpp         # ESP32 C++ firmware (Arduino/PlatformIO)
└── sim/
    └── robot.py         # Pygame robot simulation & serial bridge
```

## Getting Started

### 1. Install Python Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Microcontroller Setup (`src/main.cpp`)

You can flash the microcontroller using either **PlatformIO** or the **Arduino IDE**:

#### Option A: PlatformIO (Recommended)
```powershell
pio run -t upload
```

#### Option B: Arduino IDE
1. Open the Arduino IDE.
2. Select **ESP32 Dev Module** under Tools -> Board.
3. Open `src/main.cpp`.
4. Select your ESP32's COM port and click **Upload**.

### 3. Run the Simulation

#### Hardware Mode (ESP32 Connected via USB)
```powershell
# Auto-detects serial port:
python sim/robot.py

# Or specify your COM port explicitly:
python sim/robot.py --port COM3 --baud 115200
```

#### Wokwi Simulation Mode (Virtual ESP32 via RFC2217)
1. In VS Code, open the Command Palette (`F1` or `Ctrl+Shift+P`) and select:
   **`Wokwi: Start Simulator`**
2. In a terminal, run the Pygame robot simulator connected to Wokwi:
```powershell
python sim/robot.py --wokwi
```

#### Mock Mode (Software-in-the-Loop without hardware)
If your ESP32 is not plugged in, you can test the simulation directly with the built-in mock controller:
```powershell
python sim/robot.py --mock
```

## Simulator Controls

- **Left Mouse Click**: Set a new target waypoint for the robot.
- **Right Mouse Click** or **'O' Key**: Place a circular obstacle at the cursor.
- **'C' Key**: Clear all placed obstacles.
- **'R' Key**: Reset robot pose to origin and clear trajectory trail.
- **Close Window**: Gracefully disconnects and stops motors.

## Communication Protocol

- **Telemetry Packet (PC -> ESP32)**:
  ```text
  TELEM,<x>,<y>,<theta_radians>,<target_x>,<target_y>,<distance_front>\n
  ```
- **Actuator Command Packet (ESP32 -> PC)**:
  ```text
  CMD,<v_left>,<v_right>\n
  ```
- **Failsafe**: If the ESP32 does not receive a `TELEM` packet for 1000 ms, it automatically outputs `CMD,0.00,0.00` to prevent runaway behavior.


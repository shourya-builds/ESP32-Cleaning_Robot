/**
 * ESP32 Hardware-in-the-Loop (HIL) Cleaning Robot Firmware
 * 
 * Target: ESP32 DevKit-C (Arduino framework / PlatformIO)
 * Hardware / Peripherals:
 *   - Locomotion: L298N Dual H-Bridge Differential Drive (GPIO 25, 26, 27, 14)
 *   - Cleaning: Vacuum Suction Fan Relay (GPIO 23), Side Sweeper Brush (GPIO 13)
 *   - Perception: 3x HC-SR04 Ultrasonic (Front, Left, Right)
 *   - Safety: Tactile Collision Bumper (GPIO 32), Cliff / Drop-off Sensor (GPIO 33)
 *   - Power: Battery Voltage ADC Monitor (GPIO 34)
 *   - HMI: SSD1306 128x64 OLED Display (I2C SDA=21, SCL=22)
 *   - Navigation: MPU6050 6-Axis IMU (I2C SDA=21, SCL=22)
 *   - User Alerts: Piezo Buzzer (GPIO 12), Clean Button (GPIO 15), Status LED (GPIO 2)
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <math.h>

// --- PIN DEFINITIONS ---
#define PIN_LED             2   // Onboard & Status LED
#define PIN_BUZZER          12  // Warning / Chime Buzzer
#define PIN_BRUSH_PWM       13  // Side Sweeper Brush PWM/Servo
#define PIN_MOTOR_IN4       14  // L298N Right Motor Backwards
#define PIN_CLEAN_BTN       15  // Start/Pause Cleaning Button
#define PIN_US_R_ECHO       16  // Right Ultrasonic Echo
#define PIN_US_R_TRIG       17  // Right Ultrasonic Trig
#define PIN_US_F_ECHO       18  // Front Ultrasonic Echo
#define PIN_US_F_TRIG       19  // Left Ultrasonic Trig (Wait: F_TRIG is 5, L_TRIG is 19)
#define PIN_US_L_TRIG       19  // Left Ultrasonic Trig
#define PIN_US_L_ECHO       4   // Left Ultrasonic Echo
#define PIN_US_F_TRIG_PIN   5   // Front Ultrasonic Trig
#define PIN_I2C_SDA         21  // Shared I2C SDA
#define PIN_I2C_SCL         22  // Shared I2C SCL
#define PIN_VACUUM_RELAY    23  // Suction Fan High-Power Relay
#define PIN_MOTOR_IN1       25  // L298N Left Motor Forward
#define PIN_MOTOR_IN2       26  // L298N Left Motor Backwards
#define PIN_MOTOR_IN3       27  // L298N Right Motor Forward
#define PIN_BUMPER_SW       32  // Front Collision Bumper Microswitch
#define PIN_CLIFF_SW        33  // Cliff / Drop-off Safety Sensor
#define PIN_BATTERY_ADC     34  // Battery Voltage Potentiometer/ADC

// --- OLED CONFIGURATION ---
#define SCREEN_WIDTH        128
#define SCREEN_HEIGHT       64
#define OLED_RESET          -1
#define SCREEN_ADDRESS      0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
Adafruit_MPU6050 mpu;
bool oledReady = false;
bool mpuReady = false;

// --- ROBOT STATE MACHINE ---
enum RobotState {
    STATE_IDLE,             // Ready, waiting for Clean command
    STATE_CLEANING,         // Active vacuuming & waypoint path following
    STATE_OBSTACLE_AVOID,   // Steering around detected ultrasonic obstacle
    STATE_BUMP_REFLEX,      // Physical bumper collision: reverse & pivot
    STATE_CLIFF_EMERGENCY,  // Cliff / drop-off detected: emergency halt & alarm
    STATE_LOW_BATTERY       // Battery < 20%: conserve power, seek dock
};

enum ControlMode {
    MODE_MANUAL = 1,
    MODE_AUTOMATIC = 2,
    MODE_DESTINATION = 3
};

RobotState currentState = STATE_CLEANING; // Default to active cleaning in simulation
RobotState prevState = STATE_IDLE;
ControlMode currentMode = MODE_MANUAL;    // Default to Mode 1 (Manual WASD Mode)
float manualLeftCmd = 0.0f;
float manualRightCmd = 0.0f;

// --- ROBOT KINEMATICS & CONTROL PARAMETERS ---
const float WHEEL_BASE = 40.0f;
const float MAX_SPEED = 120.0f;
const float WAYPOINT_TOLERANCE = 15.0f;
const float KP_HEADING = 80.0f;
const float KD_HEADING = 10.0f;
const unsigned long TIMEOUT_MS = 1000;

// State Variables
unsigned long lastPacketTime = 0;
unsigned long lastDisplayUpdate = 0;
unsigned long bumpStartTime = 0;
float prevHeadingError = 0.0f;
float robotX = 0.0f, robotY = 0.0f, robotTheta = 0.0f;
float targetX = 0.0f, targetY = 0.0f;
float frontDistance = 999.0f;
int batteryPercent = 100;
bool isVacuumOn = false;
bool isBrushOn = false;
bool lastBtnState = HIGH;
unsigned long lastDebounceTime = 0;
float imuYawRate = 0.0f;

// Normalize angle to range [-PI, PI]
float normalizeAngle(float angle) {
    while (angle > M_PI) angle -= 2.0f * M_PI;
    while (angle < -M_PI) angle += 2.0f * M_PI;
    return angle;
}

// Sound simple acoustic beeps without blocking
void beepAlert(int freq, int durationMs) {
    tone(PIN_BUZZER, freq, durationMs);
}

// Update OLED Status Dashboard
void updateDisplay() {
    if (!oledReady) return;

    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);

    // Title Bar
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.print(F("ROBOT CLEANER"));

    // Battery Indicator
    display.setCursor(92, 0);
    display.print(batteryPercent);
    display.print(F("%"));

    display.drawFastHLine(0, 10, 128, SSD1306_WHITE);

    // Control Mode Display
    display.setCursor(0, 14);
    display.print(F("M:"));
    switch (currentMode) {
        case MODE_MANUAL:      display.print(F("MANUAL (WASD)")); break;
        case MODE_AUTOMATIC:   display.print(F("AUTO DISCOVER")); break;
        case MODE_DESTINATION: display.print(F("DESTINATION  ")); break;
    }

    // Actuator Status
    display.setCursor(0, 26);
    display.print(F("Vac:"));
    display.print(isVacuumOn ? F("ON ") : F("OFF"));
    display.setCursor(64, 26);
    display.print(F("Brush:"));
    display.print(isBrushOn ? F("ON") : F("OFF"));

    // Sensor Readings
    display.setCursor(0, 38);
    display.print(F("Dist:"));
    if (frontDistance < 900.0f) {
        display.print((int)frontDistance);
        display.print(F("px"));
    } else {
        display.print(F("CLEAR"));
    }

    // IMU Yaw / Heading
    display.setCursor(0, 50);
    display.print(F("IMU Wz:"));
    display.print(imuYawRate, 2);
    display.print(F("r/s"));

    display.display();
}

// Actuator control helper
void setCleaningActuators(bool vacuum, bool brush) {
    isVacuumOn = vacuum;
    isBrushOn = brush;
    digitalWrite(PIN_VACUUM_RELAY, vacuum ? HIGH : LOW);
    digitalWrite(PIN_BRUSH_PWM, brush ? HIGH : LOW);
    digitalWrite(PIN_LED, vacuum ? HIGH : LOW);
}

// Drive command generator
void sendMotorCommand(float leftSpeed, float rightSpeed) {
    leftSpeed = constrain(leftSpeed, -MAX_SPEED, MAX_SPEED);
    rightSpeed = constrain(rightSpeed, -MAX_SPEED, MAX_SPEED);

    Serial.print("CMD,");
    Serial.print(leftSpeed, 2);
    Serial.print(",");
    Serial.println(rightSpeed, 2);
}

// Process telemetry and execute State Machine navigation logic
void updateNavigation() {
    // Read safety sensors (Active LOW switches with internal pullups)
    bool isBumperHit = (digitalRead(PIN_BUMPER_SW) == LOW);
    bool isCliffTriggered = (digitalRead(PIN_CLIFF_SW) == LOW);

    // Read battery voltage (ADC 0..4095 -> 0..100%)
    int rawAdc = analogRead(PIN_BATTERY_ADC);
    batteryPercent = map(rawAdc, 0, 4095, 0, 100);

    // 1. Safety Priority Overrides
    if (isCliffTriggered) {
        if (currentState != STATE_CLIFF_EMERGENCY) {
            currentState = STATE_CLIFF_EMERGENCY;
            setCleaningActuators(false, false);
            beepAlert(2400, 300);
            Serial.println("STATUS,CLIFF_DETECTED_EMERGENCY_STOP");
        }
    } else if (isBumperHit && currentState != STATE_CLIFF_EMERGENCY) {
        if (currentState != STATE_BUMP_REFLEX) {
            currentState = STATE_BUMP_REFLEX;
            bumpStartTime = millis();
            beepAlert(1800, 150);
            Serial.println("STATUS,BUMPER_COLLISION_REFLEX");
        }
    } else if (batteryPercent < 20 && currentState == STATE_CLEANING) {
        currentState = STATE_LOW_BATTERY;
        setCleaningActuators(false, false);
        beepAlert(800, 400);
        Serial.println("STATUS,LOW_BATTERY_RETURNING");
    }

    // 2. State Machine Execution
    switch (currentState) {
        case STATE_IDLE: {
            setCleaningActuators(false, false);
            sendMotorCommand(0.0f, 0.0f);
            break;
        }

        case STATE_CLIFF_EMERGENCY: {
            // Absolute emergency stop: do not drive forward into a drop-off!
            setCleaningActuators(false, false);
            sendMotorCommand(0.0f, 0.0f);
            // If cliff switch is released, return to cleaning
            if (!isCliffTriggered) {
                currentState = STATE_CLEANING;
            }
            break;
        }

        case STATE_BUMP_REFLEX: {
            // Reverse for 450 ms, then spin 90 degrees to disengage obstacle
            unsigned long bumpElapsed = millis() - bumpStartTime;
            if (bumpElapsed < 450) {
                sendMotorCommand(-40.0f, -40.0f); // Back away
            } else if (bumpElapsed < 900) {
                sendMotorCommand(-50.0f, 50.0f);  // Turn away
            } else {
                currentState = STATE_CLEANING;
            }
            break;
        }

        case STATE_LOW_BATTERY: {
            // Drive slowly to home base without suction
            setCleaningActuators(false, false);
            sendMotorCommand(0.0f, 0.0f);
            break;
        }

        case STATE_CLEANING:
        case STATE_OBSTACLE_AVOID: {
            // Turn on cleaning subsystems
            setCleaningActuators(true, true);

            if (currentMode == MODE_MANUAL) {
                // Mode 1: Manual Control (WASD)
                currentState = STATE_CLEANING;
                sendMotorCommand(manualLeftCmd, manualRightCmd);
            } else if (currentMode == MODE_AUTOMATIC) {
                // Mode 2: Automatic Mode (Self-Discover Room)
                if (frontDistance > 0.0f && frontDistance < 60.0f) {
                    currentState = STATE_OBSTACLE_AVOID;
                    sendMotorCommand(-30.0f, 60.0f); // Pivot turn away from wall/obstacle
                } else {
                    currentState = STATE_CLEANING;
                    sendMotorCommand(85.0f, 85.0f);  // Forward sweep exploration
                }
            } else if (currentMode == MODE_DESTINATION) {
                // Mode 3: Destination Mode (Waypoint PID Navigation)
                float dx = targetX - robotX;
                float dy = targetY - robotY;
                float distToTarget = sqrtf(dx * dx + dy * dy);
                float desiredHeading = atan2f(dy, dx);

                float headingError = normalizeAngle(desiredHeading - robotTheta);
                float headingDerivative = headingError - prevHeadingError;
                prevHeadingError = headingError;

                float linearSpeed = 0.0f;
                float angularSpeed = 0.0f;

                if (distToTarget > WAYPOINT_TOLERANCE) {
                    // Reactive obstacle avoidance if front ultrasonic distance is close
                    if (frontDistance > 0.0f && frontDistance < 60.0f) {
                        currentState = STATE_OBSTACLE_AVOID;
                        linearSpeed = -20.0f;
                        angularSpeed = 60.0f;
                    } else {
                        currentState = STATE_CLEANING;
                        angularSpeed = (KP_HEADING * headingError) + (KD_HEADING * headingDerivative);
                        float alignmentFactor = cosf(headingError);

                        if (alignmentFactor > 0.0f) {
                            linearSpeed = MAX_SPEED * alignmentFactor;
                            if (distToTarget < 80.0f) {
                                linearSpeed *= (distToTarget / 80.0f);
                            }
                        } else {
                            linearSpeed = 0.0f;
                        }
                    }
                } else {
                    linearSpeed = 0.0f;
                    angularSpeed = 0.0f;
                }

                // Differential drive kinematics
                float leftCmd = linearSpeed - (angularSpeed * WHEEL_BASE / 2.0f);
                float rightCmd = linearSpeed + (angularSpeed * WHEEL_BASE / 2.0f);
                sendMotorCommand(leftCmd, rightCmd);
            }
            break;
        }
    }
}

void setup() {
    // 1. Initialize GPIOs
    pinMode(PIN_LED, OUTPUT);
    pinMode(PIN_BUZZER, OUTPUT);
    pinMode(PIN_VACUUM_RELAY, OUTPUT);
    pinMode(PIN_BRUSH_PWM, OUTPUT);

    pinMode(PIN_CLEAN_BTN, INPUT_PULLUP);
    pinMode(PIN_BUMPER_SW, INPUT_PULLUP);
    pinMode(PIN_CLIFF_SW, INPUT_PULLUP);

    digitalWrite(PIN_LED, LOW);
    digitalWrite(PIN_BUZZER, LOW);
    digitalWrite(PIN_VACUUM_RELAY, LOW);
    digitalWrite(PIN_BRUSH_PWM, LOW);

    // 2. Initialize Serial UART
    Serial.begin(115200);
    while (!Serial && millis() < 1500) {}

    // 3. Initialize I2C Bus (SDA=21, SCL=22)
    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);

    // 4. Initialize SSD1306 OLED
    if (display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
        oledReady = true;
        display.clearDisplay();
        display.setTextSize(1);
        display.setTextColor(SSD1306_WHITE);
        display.setCursor(10, 20);
        display.println(F("CLEANING ROBOT"));
        display.setCursor(10, 35);
        display.println(F("SYSTEM INITIALIZING"));
        display.display();
    }

    // 5. Initialize MPU6050 IMU
    if (mpu.begin(0x68, &Wire)) {
        mpuReady = true;
        mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
        mpu.setGyroRange(MPU6050_RANGE_500_DEG);
        mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    }

    // 6. Startup Sound & Telemetry
    beepAlert(1200, 150);
    delay(100);
    beepAlert(1600, 200);

    Serial.println(F("STATUS,CLEANING_ROBOT_INITIALIZED"));
    lastPacketTime = millis();
}

void loop() {
    // 1. Check Clean Start/Pause Pushbutton
    bool reading = digitalRead(PIN_CLEAN_BTN);
    if (reading != lastBtnState && (millis() - lastDebounceTime) > 50) {
        lastDebounceTime = millis();
        if (reading == LOW) { // Button pressed
            if (currentState == STATE_IDLE) {
                currentState = STATE_CLEANING;
                beepAlert(1500, 100);
                Serial.println("STATUS,CLEANING_STARTED_BY_BUTTON");
            } else {
                currentState = STATE_IDLE;
                beepAlert(800, 150);
                Serial.println("STATUS,CLEANING_PAUSED_BY_BUTTON");
            }
        }
    }
    lastBtnState = reading;

    // 2. Read IMU Gyro Data
    if (mpuReady) {
        sensors_event_t a, g, temp;
        mpu.getEvent(&a, &g, &temp);
        imuYawRate = g.gyro.z; // Z-axis angular velocity in rad/s
    }

    // 3. Process Telemetry & Commands from HIL Simulator / Pygame
    if (Serial.available() > 0) {
        String line = Serial.readStringUntil('\n');
        line.trim();

        if (line.startsWith("TELEM,")) {
            // Format: TELEM,x,y,theta,targetX,targetY,distFront,mode,manL,manR
            int commas[8];
            int searchPos = 6;
            int found = 0;
            for (int i = 0; i < 8; i++) {
                int c = line.indexOf(',', searchPos);
                if (c == -1) break;
                commas[i] = c;
                searchPos = c + 1;
                found++;
            }

            if (found >= 4) {
                robotX = line.substring(6, commas[0]).toFloat();
                robotY = line.substring(commas[0] + 1, commas[1]).toFloat();
                robotTheta = line.substring(commas[1] + 1, commas[2]).toFloat();
                targetX = line.substring(commas[2] + 1, commas[3]).toFloat();

                if (found >= 5) targetY = line.substring(commas[3] + 1, commas[4]).toFloat();
                else targetY = line.substring(commas[3] + 1).toFloat();

                if (found >= 6) frontDistance = line.substring(commas[4] + 1, commas[5]).toFloat();
                else if (found == 5) frontDistance = line.substring(commas[4] + 1).toFloat();

                if (found >= 7) {
                    int m = line.substring(commas[5] + 1, commas[6]).toInt();
                    if (m >= 1 && m <= 3) currentMode = (ControlMode)m;
                }
                if (found >= 8) {
                    manualLeftCmd = line.substring(commas[6] + 1, commas[7]).toFloat();
                    manualRightCmd = line.substring(commas[7] + 1).toFloat();
                }

                lastPacketTime = millis();
                updateNavigation();
            }
        } else if (line.startsWith("MODE,")) {
            int m = line.substring(5).toInt();
            if (m >= 1 && m <= 3) {
                currentMode = (ControlMode)m;
                beepAlert(1600, 80);
            }
        } else if (line.equals("PING")) {
            Serial.println("PONG");
            lastPacketTime = millis();
        } else if (line.equals("CMD,START")) {
            currentState = STATE_CLEANING;
            beepAlert(1500, 100);
        } else if (line.equals("CMD,STOP")) {
            currentState = STATE_IDLE;
            beepAlert(800, 150);
        }
    }

    // 4. Update OLED Display periodically (5 Hz)
    if (millis() - lastDisplayUpdate > 200) {
        lastDisplayUpdate = millis();
        updateDisplay();
    }

    // 5. Watchdog Timeout Failsafe
    if (millis() - lastPacketTime > TIMEOUT_MS) {
        sendMotorCommand(0.0f, 0.0f);
        setCleaningActuators(false, false);
        delay(50);
    }
}



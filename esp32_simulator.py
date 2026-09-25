"""
ACMGS v2.0 ESP32 Calibrated Edge Simulator
Module: esp32_simulator.py

Simulates ESP32 Node 1 (DHT11 on GPIO 4 + ACS712 on GPIO 34) streaming telemetry
to the FastAPI Edge Server at 500ms intervals.

Supports Disturbance & Failure Modes:
  - Normal: Standard 500ms telemetry (35-42°C, 12-14A RMS)
  - Thermal Runaway: Chamber heat exceeds 60°C -> Triggers Proportional MOSFET PWM cooling
  - Transient Noise Pulse: Single-window current spike -> Filtered out by Dual-Window Guardrail
  - Irreversible Defect: Sustained tool wear + thermal surge -> Triggers Sunk-Energy Abort (<20ms)
"""

import sys
import time
import json
import random
import math
import argparse
import requests
from datetime import datetime

DEFAULT_SERVER_URL = "http://localhost:8000/api/sensor-data"


class CalibratedSensorSimulator:
    def __init__(self, mode: str = "normal"):
        self.mode = mode
        self.cycle = 0
        self.base_temp = 38.0
        self.base_current = 12.5
        self.base_humidity = 48.0
        self.spindle_speed = 1850.0
        self.clamp_pressure = 5.2
        self.feed_rate = 0.85
        self.grid_carbon = 250.0

    def generate_packet(self) -> dict:
        self.cycle += 1
        t = self.cycle * 0.1

        # Baseline sinusoidal fluctuation
        current = self.base_current + 0.8 * math.sin(t) + random.gauss(0, 0.15)
        temp = self.base_temp + 1.2 * math.sin(t * 0.5) + random.gauss(0, 0.1)
        humidity = self.base_humidity + 1.0 * math.cos(t * 0.3) + random.gauss(0, 0.2)

        # Mode injection
        if self.mode == "thermal_spike":
            # Ramp temperature to 62°C to test closed-loop MOSFET PWM cooling
            temp += min(28.0, self.cycle * 1.5)
            current += 2.0
        elif self.mode == "transient_spike":
            # Single transient pulse on cycle 5
            if self.cycle == 5:
                current += 18.0  # 30A spike
        elif self.mode == "irreversible_defect":
            # Sustained defect across multiple windows (simulates tool fracture at minute 18)
            if self.cycle >= 3:
                current += 16.0  # Excessive motor current
                temp += 25.0     # Severe overheating
                self.feed_rate = 1.6  # Severe tool chatter

        return {
            "temperature": round(max(20.0, temp), 2),
            "humidity": round(max(10.0, min(95.0, humidity)), 2),
            "current": round(max(0.0, current), 2),
            "pressure": round(self.clamp_pressure, 2),
            "speed": round(self.spindle_speed, 0),
            "feed_rate": round(self.feed_rate, 2),
            "carbon_intensity": round(self.grid_carbon, 1),
            "timestamp": int(time.time() * 1000),
            "rssi": random.randint(-65, -45),
            "ip": "192.168.1.147"
        }

    def stream_loop(self, url: str, interval: float = 0.5, max_packets: Optional[int] = None):
        print("\n" + "═" * 70)
        print("  ACMGS v2.0 — ESP32 Calibrated Edge Telemetry Simulator")
        print(f"  Target Server: {url}")
        print(f"  Mode         : {self.mode.upper()}")
        print(f"  Rate         : {interval*1000:.0f} ms per telemetry packet")
        print("═" * 70 + "\n")

        count = 0
        try:
            while max_packets is None or count < max_packets:
                payload = self.generate_packet()
                count += 1
                try:
                    res = requests.post(url, json=payload, timeout=2.0)
                    if res.status_code == 200:
                        data = res.json()
                        act = data.get("actuation", {})
                        health = data.get("health", {})
                        pred = data.get("predictions", {})
                        pwm = act.get("pwm_duty", 0)
                        status = act.get("status", "NOMINAL")
                        h_idx = health.get("health_index", 100)
                        
                        feed_hold_tag = " [EMERGENCY ABORT!]" if act.get("feed_hold") else ""
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] "
                            f"Temp: {payload['temperature']:5.1f}°C | "
                            f"I_rms: {payload['current']:4.1f}A | "
                            f"PWM: {pwm:3d}/255 ({act.get('fan_speed_pct',0):.0f}%) | "
                            f"Health: {h_idx:4.1f}% | "
                            f"Status: {status}{feed_hold_tag}"
                        )
                    else:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Server HTTP {res.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Server connection failed (is esp32_server running?): {e}")

                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n✓ Simulator stopped cleanly.")


def main():
    parser = argparse.ArgumentParser(description="ACMGS v2.0 ESP32 Telemetry Simulator")
    parser.add_argument("--url", default=DEFAULT_SERVER_URL, help="FastAPI sensor endpoint")
    parser.add_argument("--mode", default="normal", choices=["normal", "thermal_spike", "transient_spike", "irreversible_defect"])
    parser.add_argument("--interval", type=float, default=0.5, help="Publish interval in seconds")
    parser.add_argument("--count", type=int, default=None, help="Total packets to send")
    args = parser.parse_args()

    sim = CalibratedSensorSimulator(mode=args.mode)
    sim.stream_loop(url=args.url, interval=args.interval, max_packets=args.count)


if __name__ == "__main__":
    main()

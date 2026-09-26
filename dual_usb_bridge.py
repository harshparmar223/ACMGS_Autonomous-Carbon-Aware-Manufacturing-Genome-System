"""
ACMGS v2.0 - Complete Dual-USB Hardware Pipeline
================================================
Node 1 (ESP32 - Sensors)  -> Read from COM8 -> Ingest into Edge Server (Port 8001)
Node 2 (ESP-12E - Actuator) -> Read decision from Edge Server -> Write to COM13

Zero Wi-Fi required. Works completely over USB cables!
"""

import sys
import time
import json
import requests
import serial
import serial.tools.list_ports as lp

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

SENSOR_PORT = "COM8"
ACTUATOR_PORT = "COM13"
BAUD_RATE = 115200
EDGE_SERVER = "http://localhost:8001"

def main():
    print("=" * 65)
    print("  ACMGS v2.0 - Dual-USB Hardware Bridge (Zero-WiFi Master)")
    print(f"  Node 1 (Sensors):  {SENSOR_PORT} (ESP32)")
    print(f"  Node 2 (Actuator): {ACTUATOR_PORT} (ESP-12E)")
    print("=" * 65)
    print("IMPORTANT: Ensure all Arduino Serial Monitors & Miniterms are CLOSED!\n")

    ser_sensor = None
    ser_actuator = None

    try:
        ser_sensor = serial.Serial(SENSOR_PORT, BAUD_RATE, timeout=1.0)
        print(f"[OK] Connected to Sensor Node on {SENSOR_PORT}")
    except Exception as e:
        print(f"[!] Warning: Could not open {SENSOR_PORT}: {e}")

    try:
        ser_actuator = serial.Serial(ACTUATOR_PORT, BAUD_RATE, timeout=1.0)
        print(f"[OK] Connected to Actuator Node on {ACTUATOR_PORT}")
    except Exception as e:
        print(f"[!] Warning: Could not open {ACTUATOR_PORT}: {e}")

    if not ser_sensor and not ser_actuator:
        print("\n[X] Neither port could be opened. Please close Arduino Serial Monitor and retry.")
        return

    print("\n[+] Streaming loop active. Press Ctrl+C to stop.\n")
    last_act_sync = 0

    while True:
        try:
            # 1. Read Sensor Telemetry from ESP32 (COM8)
            if ser_sensor and ser_sensor.in_waiting > 0:
                line = ser_sensor.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        doc = json.loads(line)
                        t = doc.get("temperature", 0)
                        c = doc.get("current", 0)
                        v = doc.get("voltage", 0)
                        print(f"📥 [COM8 Sensors] Temp: {t}°C | Current: {c}A | Volt: {v}V")
                        
                        # Ingest into local edge server
                        requests.post(f"{EDGE_SERVER}/api/data", json=doc, timeout=0.8)
                    except Exception as err:
                        pass

            # 2. Check for physical button presses on ESP-12E (COM13)
            if ser_actuator and ser_actuator.in_waiting > 0:
                act_line = ser_actuator.readline().decode('utf-8', errors='ignore').strip()
                if act_line.startswith("{") and act_line.endswith("}"):
                    try:
                        btn_doc = json.loads(act_line)
                        if "PHYSICAL" in str(btn_doc.get("source", "")).upper():
                            is_estop = btn_doc.get("estop", False)
                            if is_estop:
                                print("🚨 [COM13 Hardware Event] PHYSICAL E-STOP ENGAGED: Machine Motor STOPPED (0 PWM) + Siren Alarm ON!")
                                requests.post(f"{EDGE_SERVER}/api/actuate", json={"pwm": 0, "feed_hold": True}, timeout=0.8)
                            else:
                                print("🟢 [COM13 Hardware Event] PHYSICAL E-STOP DISARMED: Machine Motor Resumed Running.")
                                requests.post(f"{EDGE_SERVER}/api/override/reset", timeout=0.8)
                                requests.post(f"{EDGE_SERVER}/api/actuate", json={"pwm": 150, "feed_hold": False}, timeout=0.8)
                    except Exception:
                        pass
                elif "PHYSICAL" in act_line.upper() or "ESTOP" in act_line.upper():
                    print(f"ℹ️ [COM13] {act_line}")

            # 3. Sync Actuation Command to ESP-12E (COM13) every 500ms
            if ser_actuator and (time.time() - last_act_sync > 0.5):
                last_act_sync = time.time()
                try:
                    r = requests.get(f"{EDGE_SERVER}/api/latest", timeout=0.5)
                    if r.status_code == 200:
                        payload = r.json()
                        act = payload.get("actuation", {})
                        pwm = act.get("pwm_duty", 0)
                        feed_hold = act.get("feed_hold", False)
                        status_str = act.get("status", "NOMINAL")
                        alert = (status_str != "NOMINAL") or feed_hold

                        # Send to hardware
                        cmd = json.dumps({"pwm": pwm, "alert": alert, "estop": feed_hold}) + "\n"
                        ser_actuator.write(cmd.encode('utf-8'))
                        print(f"⚡ [COM13 Fan/Actuator] PWM: {pwm}/255 | Alert: {alert} | E-Stop: {feed_hold}")
                except Exception as err:
                    pass

            time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n[!] Bridge stopped by user.")
            break
        except Exception as e:
            time.sleep(0.5)

    if ser_sensor: ser_sensor.close()
    if ser_actuator: ser_actuator.close()

if __name__ == "__main__":
    main()

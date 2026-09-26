"""
ACMGS v2.0 - Direct USB Serial Telemetry Bridge
Connects directly to ESP32 over USB COM port and streams data to Dashboard.
Run:
    python serial_bridge.py
"""

import sys
import json
import time
import requests
import serial
import serial.tools.list_ports as lp

SERVER_URL = "http://localhost:8001/api/data"
BAUD_RATE = 115200

def find_esp_port():
    ports = lp.comports()
    for p in ports:
        desc = (p.description or "").lower()
        dev = p.device
        if any(chip in desc for chip in ["cp210", "ch340", "uart", "usb serial"]):
            return dev
    return None

def main():
    port = find_esp_port()
    if not port:
        print("[-] No ESP32 USB COM port auto-detected. Available ports:")
        for p in lp.comports():
            print(f"    - {p.device}: {p.description}")
        if len(sys.argv) > 1:
            port = sys.argv[1]
        else:
            port = input("Enter COM port (e.g. COM8): ").strip()

    print(f"\n[+] Connecting to ESP32 on {port} at {BAUD_RATE} baud...")
    print(f"[+] Forwarding telemetry to {SERVER_URL}...\n")
    print("NOTE: Make sure Arduino IDE Serial Monitor is CLOSED so this script can access the port.")

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1.0)
        time.sleep(1.5)
        print("[✓] Connected! Listening for sensor data packets...\n")
        
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            if line.startswith("{") and line.endswith("}"):
                try:
                    payload = json.loads(line)
                    print(f"📥 USB Packet: Temp={payload.get('temperature')}°C | Hum={payload.get('humidity')}% | Curr={payload.get('current')}A | Volt={payload.get('voltage')}V")
                    r = requests.post(SERVER_URL, json=payload, timeout=1.0)
                    if r.status_code == 200:
                        print(f"   ↳ [OK] Forwarded to Dashboard (HTTP 200)")
                except Exception as e:
                    print(f"   ↳ [!] Forward error: {e}")
            else:
                if any(k in line for k in ["[WIFI]", "[HTTP]", "ACMGS"]):
                    print(f"ℹ️ {line}")
    except serial.SerialException as se:
        print(f"\n[X] Port Error: {se}")
        print("    If port is busy, CLOSE the Arduino IDE Serial Monitor and retry.\n")
    except KeyboardInterrupt:
        print("\n[!] Bridge stopped by user.")

if __name__ == "__main__":
    main()

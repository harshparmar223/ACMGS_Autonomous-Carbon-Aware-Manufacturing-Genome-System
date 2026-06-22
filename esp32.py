"""
ESP32 Sensor Data Simulator
Generates realistic, varying sensor data and posts to ESP32 server
"""

import requests
import time
import json
import random
from datetime import datetime
import math

# Configuration
ESP32_URL = "http://localhost:8001"
ENDPOINT = "/api/sensor-data"
UPDATE_INTERVAL = 5  # seconds between readings

# Sensor ranges and patterns
class SensorSimulator:
    def __init__(self):
        self.cycle = 0
        self.base_current = 50
        self.base_temp = 35
        self.base_humidity = 55
        
    def get_sensor_data(self):
        """Generate realistic, time-varying sensor data"""
        self.cycle += 1
        
        # Create realistic sine wave patterns with noise
        time_factor = self.cycle / 10.0
        
        # Current: 40-100A with a slow cycle
        current = self.base_current + 25 * math.sin(time_factor) + random.uniform(-3, 3)
        current = max(30, min(120, current))  # Clamp between 30-120A
        
        # Temperature: 25-45°C with variation
        temp = self.base_temp + 10 * math.sin(time_factor * 0.7 + 1) + random.uniform(-2, 2)
        temp = max(20, min(50, temp))
        
        # Humidity: 40-70% with inverse current relationship
        humidity = self.base_humidity + 15 * math.sin(time_factor * 0.3 + 2) + random.uniform(-1, 1)
        humidity = max(30, min(80, humidity))
        
        # Occasional spikes (anomalies)
        if random.random() < 0.05:  # 5% chance of anomaly
            current += random.uniform(10, 30)
            temp += random.uniform(5, 15)
        
        return {
            "temperature": round(temp, 1),
            "humidity": round(humidity, 1),
            "current": round(current, 1),
            "timestamp": int(time.time() * 1000),
            "rssi": random.randint(-90, -30),
            "ip": "10.194.49.147"
        }
    
    def send_data(self):
        """Send sensor data to ESP32 server"""
        try:
            data = self.get_sensor_data()
            
            response = requests.post(
                f"{ESP32_URL}{ENDPOINT}",
                json=data,
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                # Parse power from response
                power_w = data['current'] * 230.0
                
                print(f"✓ [{datetime.now().strftime('%H:%M:%S')}] "
                      f"Current: {data['current']:.1f}A | "
                      f"Temp: {data['temperature']:.1f}°C | "
                      f"Humidity: {data['humidity']:.1f}% | "
                      f"Power: {power_w:.0f}W")
            else:
                print(f"✗ Server error: {response.status_code}")
        except Exception as e:
            print(f"✗ Connection error: {e}")

def main():
    """Run the simulator"""
    print("\n" + "="*70)
    print("ESP32 SENSOR SIMULATOR - Starting Dynamic Data Generation")
    print("="*70)
    print(f"Target: {ESP32_URL}")
    print(f"Sending data every {UPDATE_INTERVAL} seconds...")
    print("Press Ctrl+C to stop\n")
    
    simulator = SensorSimulator()
    
    try:
        while True:
            simulator.send_data()
            time.sleep(UPDATE_INTERVAL)
    except KeyboardInterrupt:
        print("\n\n✓ Simulator stopped.")

if __name__ == "__main__":
    main()

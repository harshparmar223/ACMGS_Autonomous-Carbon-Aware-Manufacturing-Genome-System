/*
  =============================================================================
  ACMGS v2.0 - ESP32 Sensor Telemetry Node
  =============================================================================
  WiFi: Harsh / 12345678
  Hardware:
    - DHT11 (Temp/Humidity) -> GPIO 32
    - ACS712 (Current)      -> GPIO 34 (ADC1_CH6)
    - Voltage Sensor (0-25V) -> GPIO 35 (ADC1_CH7)
  =============================================================================
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>
#include <ArduinoJson.h>

// ===== WiFi Configuration =====
const char* ssid = "Harsh";
const char* password = "12345678";

// Backend server URL (PC IP on 'Harsh' WiFi network is 10.206.165.55)
const char* serverUrl = "http://10.206.165.55:8001/api/data";

// ===== Sensor Pins =====
#define DHTPIN          32    // DHT11 Data -> GPIO 32
#define DHTTYPE         DHT11
#define ACS712_PIN      34    // ACS712 Out -> GPIO 34
#define VOLTAGE_PIN     35    // Voltage Sensor Out -> GPIO 35

// ===== Calibration =====
#define ACS712_SENSITIVITY  0.185  // 185 mV/A for 5A module (use 0.066 for 30A)
#define ACS712_ZERO_V       2.50   // 2.5V reference at 0A
#define VOLTAGE_DIVIDER_RATIO 5.0  // Standard 5:1 resistor divider module

DHT dht(DHTPIN, DHTTYPE);

unsigned long lastSampleTime = 0;
const unsigned long SAMPLE_INTERVAL = 1000; // 1 second loop

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n╔════════════════════════════════════════════════════════╗");
  Serial.println("║   ACMGS v2.0 - ESP32 Telemetry Node (Harsh WiFi)       ║");
  Serial.println("║   DHT11: GPIO 32 | ACS712: GPIO 34 | Voltage: GPIO 35  ║");
  Serial.println("╚════════════════════════════════════════════════════════╝");

  pinMode(DHTPIN, INPUT_PULLUP);
  dht.begin();

  // Configure ESP32 ADC (12-bit: 0-4095, 11dB attenuation up to 3.3V)
  analogSetWidth(12);
  analogSetAttenuation(ADC_11db);

  // Auto-Calibrate ACS712 zero-current baseline
  Serial.print("[CALIB] Sampling ACS712 zero-point baseline (keep load OFF)...");
  float calSum = 0;
  for (int i = 0; i < 60; i++) {
    calSum += (analogRead(ACS712_PIN) / 4095.0) * 3.3;
    delay(10);
  }
  zeroCurrentVoltage = calSum / 60.0;
  Serial.printf(" Baseline: %.2fV\n", zeroCurrentVoltage);

  // Connect to WiFi
  WiFi.persistent(false);
  WiFi.disconnect(true);
  delay(200);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false); // Disable power save (vital for mobile hotspots)
  WiFi.setAutoReconnect(true);
  WiFi.begin(ssid, password);
  Serial.print("[WIFI] Connecting to '");
  Serial.print(ssid);
  Serial.println("'...");

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WIFI] Connected successfully!");
    Serial.print("[WIFI] ESP32 IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.printf("\n[WIFI] Initial connect timeout (status: %d). Continuing telemetry in loop...\n", WiFi.status());
  }
}

float zeroCurrentVoltage = 2.50; // Dynamic calibration in setup()

// Read Current (Amperes) with auto-zero baseline
float readCurrent(int samples = 30) {
  float sumVolts = 0.0;
  for (int i = 0; i < samples; i++) {
    int raw = analogRead(ACS712_PIN);
    sumVolts += (raw / 4095.0) * 3.3;
    delayMicroseconds(150);
  }
  float avgVolts = sumVolts / (float)samples;
  float current = (avgVolts - zeroCurrentVoltage) / ACS712_SENSITIVITY;
  return (abs(current) < 0.15) ? 0.0 : abs(current);
}

// Read Voltage (Volts) with floating noise clamp
float readVoltage(int samples = 15) {
  long rawSum = 0;
  for (int i = 0; i < samples; i++) {
    rawSum += analogRead(VOLTAGE_PIN);
    delayMicroseconds(100);
  }
  float rawAvg = rawSum / (float)samples;
  float pinVoltage = (rawAvg / 4095.0) * 3.3;
  float actualVoltage = pinVoltage * VOLTAGE_DIVIDER_RATIO;
  return (actualVoltage < 0.8) ? 0.0 : actualVoltage; // Clamp floating wire noise
}

void loop() {
  if (millis() - lastSampleTime >= SAMPLE_INTERVAL) {
    lastSampleTime = millis();

    float temp = dht.readTemperature();
    float humidity = dht.readHumidity();
    float current = readCurrent();
    float voltage = readVoltage();
    float power = voltage * current;

    if (isnan(temp) || isnan(humidity)) {
      Serial.println("[DHT11] Warning: Failed to read DHT11 on GPIO 32 (check wiring/3.3V power).");
      if (isnan(temp)) temp = 25.0;
      if (isnan(humidity)) humidity = 50.0;
    }

    // Build JSON Payload
    StaticJsonDocument<256> doc;
    doc["node"] = 1;
    doc["temperature"] = round(temp * 10) / 10.0;
    doc["humidity"] = round(humidity * 10) / 10.0;
    doc["current"] = round(current * 100) / 100.0;
    doc["voltage"] = round(voltage * 10) / 10.0;
    doc["power"] = round(power * 10) / 10.0;
    doc["timestamp"] = millis();

    // 1. Output over Serial Monitor
    serializeJson(doc, Serial);
    Serial.println();

    // 2. Transmit to Python Backend via WiFi HTTP POST if connected
    if (WiFi.status() == WL_CONNECTED) {
      HTTPClient http;
      http.begin(serverUrl);
      http.addHeader("Content-Type", "application/json");
      http.setTimeout(2000);
      String jsonStr;
      serializeJson(doc, jsonStr);
      int httpCode = http.POST(jsonStr);
      if (httpCode > 0) {
        Serial.printf("[HTTP] POST Success -> %s (Code: %d)\n", serverUrl, httpCode);
      } else {
        Serial.printf("[HTTP] POST Failed to %s, error: %s\n", serverUrl, http.errorToString(httpCode).c_str());
      }
      http.end();
    } else {
      static unsigned long lastWifiRetry = 0;
      if (millis() - lastWifiRetry > 8000) {
        lastWifiRetry = millis();
        int st = WiFi.status();
        Serial.printf("[WIFI] Status: %d ", st);
        if (st == 1) Serial.println("(SSID 'Harsh' not found - check if hotspot is broadcasting)");
        else if (st == 4) Serial.println("(Auth failed - check password)");
        else if (st == 6) Serial.println("(Connecting/Negotiating...)");
        else Serial.println();
        WiFi.disconnect();
        WiFi.begin(ssid, password);
      }
    }
  }
}

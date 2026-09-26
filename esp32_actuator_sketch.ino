/*
  =============================================================================
  ACMGS v2.0 - ESP-12E (ESP8266) Closed-Loop Actuator Node
  =============================================================================
  WiFi: Harsh / 12345678
  Hardware (ESP-12E / NodeMCU):
    - IRF540 MOSFET Gate -> D1 (GPIO 5) -> Fan PWM Speed
    - LED (Anode + 220Ω) -> D2 (GPIO 4) -> Machine Alert Indicator
    - Buzzer (+)         -> D5 (GPIO 14) -> Emergency Audio Alarm
  =============================================================================
*/

#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ArduinoJson.h>

// ===== Pin Definitions (ESP8266 NodeMCU Mapping) =====
#define FAN_PIN        D1    // GPIO 5  -> IRF540 MOSFET Gate
#define LED_PIN        D2    // GPIO 4  -> Indicator LED
#define BUZZER_PIN     D5    // GPIO 14 -> Alert Buzzer
#define ESTOP_BTN_PIN  D6    // GPIO 12 -> Physical E-STOP Button to GND

// ===== WiFi Configuration =====
const char* ssid = "Harsh";
const char* password = "12345678";
ESP8266WebServer server(80);

// State variables
int currentPwm = 0;              // 0 to 1023 on ESP8266
bool alertActive = false;
bool emergencyStop = false;
unsigned long lastCommandTime = 0;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(50); // Prevent Serial timeout from starving the ESP8266 Watchdog
  delay(500);

  Serial.println("\n╔════════════════════════════════════════════════════════╗");
  Serial.println("║   ACMGS v2.0 - ESP-12E Actuator Node (Harsh WiFi)      ║");
  Serial.println("║   Fan: D1 (GPIO 5) | LED: D2 (GPIO 4) | Buzzer: D5     ║");
  Serial.println("╚════════════════════════════════════════════════════════╝");
  Serial.printf("[BOOT] Reset Reason: %s\n", ESP.getResetReason().c_str());

  // Configure Output & Input Pins
  pinMode(FAN_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(ESTOP_BTN_PIN, INPUT_PULLUP); // Active LOW on press

  // Initial State: Machine running nominally! (Fan ON at ~60% speed)
  currentPwm = 600;
  emergencyStop = false;
  analogWrite(FAN_PIN, currentPwm);
  digitalWrite(LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  // Configure WiFi for ESP8266 stability
  WiFi.persistent(false);
  WiFi.disconnect(true);
  delay(100);
  WiFi.mode(WIFI_STA);
  WiFi.setOutputPower(17.5); // Reduce RF surge current to prevent USB power dips

  // Scan visible 2.4 GHz networks to verify SSID
  Serial.println("[SCAN] Scanning visible 2.4 GHz Wi-Fi networks...");
  int n = WiFi.scanNetworks();
  Serial.printf("[SCAN] Found %d network(s):\n", n);
  bool foundHarsh = false;
  for (int i = 0; i < n; ++i) {
    Serial.printf("  [%d] SSID: '%s' | Signal: %d dBm | Ch: %d\n", i + 1, WiFi.SSID(i).c_str(), WiFi.RSSI(i), WiFi.channel(i));
    if (WiFi.SSID(i) == ssid) foundHarsh = true;
  }
  if (!foundHarsh) {
    Serial.printf("[SCAN] ⚠️ '%s' not visible on 2.4GHz! Hotspot might be on 5GHz or hidden.\n", ssid);
  }

  WiFi.setAutoConnect(true);
  WiFi.setAutoReconnect(true);
  WiFi.begin(ssid, password);
  
  Serial.print("[WIFI] Connecting to '");
  Serial.print(ssid);
  Serial.println("'...");

  int retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 30) {
    delay(400);
    yield(); // Feed ESP8266 Watchdog
    Serial.print(".");
    retries++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WIFI] Connected! Actuator IP: " + WiFi.localIP().toString());
    setupHttpEndpoints();
    server.begin();
    Serial.println("[HTTP] Actuation listener running on Port 80");
  } else {
    Serial.printf("\n[WIFI] Status: %d. Waiting for connection in background.\n", WiFi.status());
  }

  lastCommandTime = millis();
}

void applyActuation(int pwmVal, bool alert, bool eStop, const char* source) {
  // ESP8266 PWM is 10-bit: 0 to 1023 (Convert 0-255 inputs to 0-1023)
  if (pwmVal <= 255) {
    currentPwm = map(pwmVal, 0, 255, 0, 1023);
  } else {
    currentPwm = constrain(pwmVal, 0, 1023);
  }

  alertActive = alert;
  emergencyStop = eStop;
  lastCommandTime = millis();

  // 1. Machine Spindle / Fan Motor Control (MOSFET Gate)
  if (emergencyStop) {
    analogWrite(FAN_PIN, 0); // 🚨 INSTANT EMERGENCY STOP: Cut machine motor to 0 RPM (<20ms)
    digitalWrite(LED_PIN, HIGH);
    digitalWrite(BUZZER_PIN, HIGH); // DC High for active buzzers
    tone(BUZZER_PIN, 2000);         // 2 kHz tone for passive buzzers
  } else if (alertActive) {
    analogWrite(FAN_PIN, max(currentPwm, 700)); // High speed on warning
    digitalWrite(LED_PIN, HIGH);
    noTone(BUZZER_PIN);
    digitalWrite(BUZZER_PIN, LOW);
  } else {
    // Normal operation: MACHINE IS RUNNING! (Guaranteed >= 600 PWM / ~60% speed)
    int runSpeed = max(currentPwm, 600);
    analogWrite(FAN_PIN, runSpeed);
    digitalWrite(LED_PIN, LOW);
    noTone(BUZZER_PIN);
    digitalWrite(BUZZER_PIN, LOW);
  }

  // Echo ACK over Serial
  StaticJsonDocument<256> doc;
  doc["node"] = 2;
  doc["fan_pwm"] = currentPwm;
  doc["fan_pct"] = round((currentPwm / 1023.0) * 100.0);
  doc["alert"] = alertActive;
  doc["estop"] = emergencyStop;
  doc["source"] = source;
  serializeJson(doc, Serial);
  Serial.println();
}

void setupHttpEndpoints() {
  server.on("/actuate", HTTP_POST, []() {
    if (!server.hasArg("plain")) {
      server.send(400, "application/json", "{\"error\":\"Missing body\"}");
      return;
    }
    StaticJsonDocument<256> doc;
    deserializeJson(doc, server.arg("plain"));
    int pwm = doc["pwm"] | 0;
    bool alert = doc["alert"] | false;
    bool estop = doc["feed_hold"] | doc["estop"] | false;

    applyActuation(pwm, alert, estop, "HTTP");
    server.send(200, "application/json", "{\"status\":\"OK\",\"fan_pwm\":" + String(currentPwm) + "}");
  });

  server.on("/status", HTTP_GET, []() {
    String resp = "{\"node\":2,\"fan_pwm\":" + String(currentPwm) + ",\"estop\":" + String(emergencyStop ? "true" : "false") + "}";
    server.send(200, "application/json", resp);
  });
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    server.handleClient();
  }

  // Read incoming Serial JSON commands (e.g. from Python Decision Engine)
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.startsWith("{") && line.endsWith("}")) {
      StaticJsonDocument<256> doc;
      if (!deserializeJson(doc, line)) {
        int pwm = doc["pwm"] | currentPwm;
        bool alert = doc["alert"] | false;
        bool estop = doc["feed_hold"] | doc["estop"] | false;
        bool resetCmd = doc["reset"] | false;

        // Handle E-STOP and Reset commands cleanly
        if (estop) {
          emergencyStop = true;
          pwm = 0; // Cut machine motor to 0 RPM
          alert = true;
        } else if (resetCmd || !estop) {
          emergencyStop = false;
          if (pwm == 0) pwm = 150; // Keep machine motor running at nominal speed
        }

        applyActuation(pwm, alert, emergencyStop, "SERIAL");
      }
    }
  }

  // Visual LED blink during warning alert state
  if (alertActive && !emergencyStop) {
    digitalWrite(LED_PIN, (millis() / 400) % 2);
  }

  // Emergency Siren Alarm (Pulsing sound every 250ms)
  if (emergencyStop) {
    bool beep = (millis() / 250) % 2;
    digitalWrite(BUZZER_PIN, beep ? HIGH : LOW);
    if (beep) tone(BUZZER_PIN, 2200); else noTone(BUZZER_PIN);
  }

  // Physical Hardware E-STOP Push Button (D6 to GND) with Debounce Toggle
  static unsigned long lastBtnPress = 0;
  if (digitalRead(ESTOP_BTN_PIN) == LOW && (millis() - lastBtnPress > 500)) {
    lastBtnPress = millis();
    emergencyStop = !emergencyStop;
    if (emergencyStop) {
      // 🚨 BUTTON PRESSED: STOP FAN IMMEDIATELY! (PWM 0), alert LED ON, siren ON!
      analogWrite(FAN_PIN, 0);
      digitalWrite(LED_PIN, HIGH);
      applyActuation(0, true, true, "PHYSICAL_BTN");
      Serial.println("{\"node\":2,\"fan_pwm\":0,\"estop\":true,\"source\":\"PHYSICAL_BTN\"}");
    } else {
      // 🟢 BUTTON PRESSED AGAIN: RESTART FAN RUNNING (PWM 600 ~ 60% speed), alarm OFF!
      analogWrite(FAN_PIN, 600);
      digitalWrite(LED_PIN, LOW);
      noTone(BUZZER_PIN);
      digitalWrite(BUZZER_PIN, LOW);
      applyActuation(600, false, false, "PHYSICAL_BTN");
      Serial.println("{\"node\":2,\"fan_pwm\":600,\"estop\":false,\"source\":\"PHYSICAL_BTN\"}");
    }
  }

  delay(10);
}

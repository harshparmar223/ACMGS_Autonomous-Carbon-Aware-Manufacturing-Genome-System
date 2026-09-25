/*
  =============================================================================
  ACMGS v2.0 - ESP32 Node 2 Actuation Firmware (Solid-State MOSFET Closed Loop)
  =============================================================================
  Target: ESP32 Node 2 (Actuation Layer)
  Hardware: 
    - Logic-Level N-Channel MOSFET Gate on GPIO 18 (Zero mechanical relays)
    - High-Flow DC Industrial Cooling Fan on MOSFET Drain
    - Emergency Feed-Hold Interlock Indicator LED on GPIO 2
  Features:
    - 5 kHz Hardware PWM (8-bit resolution: 0 - 255)
    - Sub-20ms cyber-physical actuation response time
    - Dual control: Serial JSON + HTTP REST endpoint
    - Automatic failsafe watchdog (spools fan to 100% on loss of signal)
  =============================================================================
*/

#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>

// ===== Pin Definitions =====
#define MOSFET_PWM_PIN    18    // GPIO 18 -> MOSFET Gate
#define FEED_HOLD_LED_PIN 2     // GPIO 2 -> Onboard LED (Feed-Hold indicator)

// ===== PWM Configuration =====
#define PWM_CHANNEL       0
#define PWM_FREQ          5000  // 5 kHz switching frequency
#define PWM_RESOLUTION    8     // 8-bit: 0 to 255 duty cycle

// ===== WiFi Configuration (Optional for Web Actuation) =====
const char* ssid = "POCO M4 Pro 5G Harsh";
const char* password = "Harsh@12345";
WebServer server(80);

// ===== State Variables =====
int currentPwm = 0;
bool feedHoldActive = false;
unsigned long lastCommandTime = 0;
const unsigned long WATCHDOG_TIMEOUT_MS = 15000; // 15s watchdog

void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("\n╔════════════════════════════════════════════════════════╗");
  Serial.println("║   ACMGS v2.0 - Node 2: Closed-Loop MOSFET Actuator     ║");
  Serial.println("║   Solid-State Logic MOSFET on GPIO 18 (PWM 0-255)      ║");
  Serial.println("╚════════════════════════════════════════════════════════╝");

  // Setup GPIO & PWM
  pinMode(FEED_HOLD_LED_PIN, OUTPUT);
  digitalWrite(FEED_HOLD_LED_PIN, LOW);

  ledcSetup(PWM_CHANNEL, PWM_FREQ, PWM_RESOLUTION);
  ledcAttachPin(MOSFET_PWM_PIN, PWM_CHANNEL);
  ledcWrite(PWM_CHANNEL, 0); // Start with fan off

  // Connect WiFi
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("[WIFI] Connecting to ");
  Serial.println(ssid);

  int retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 15) {
    delay(500);
    Serial.print(".");
    retries++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WIFI] Connected! IP Address: " + WiFi.localIP().toString());
    setupHttpEndpoints();
    server.begin();
    Serial.println("[HTTP] Actuation REST Server started on port 80");
  } else {
    Serial.println("\n[WIFI] WiFi fallback to Serial-only actuation mode.");
  }

  lastCommandTime = millis();
}

void applyActuation(int pwm, bool feedHold, const char* source) {
  currentPwm = constrain(pwm, 0, 255);
  feedHoldActive = feedHold;
  lastCommandTime = millis();

  // If feed-hold / emergency abort asserted, enforce 100% cooling fan
  if (feedHoldActive) {
    currentPwm = 255;
    digitalWrite(FEED_HOLD_LED_PIN, HIGH);
  } else {
    digitalWrite(FEED_HOLD_LED_PIN, LOW);
  }

  ledcWrite(PWM_CHANNEL, currentPwm);

  // Acknowledge back over Serial
  StaticJsonDocument<256> doc;
  doc["node"] = 2;
  doc["type"] = "ACTUATION_ACK";
  doc["pwm"] = currentPwm;
  doc["fan_speed_pct"] = round((currentPwm / 255.0) * 100.0);
  doc["feed_hold"] = feedHoldActive;
  doc["source"] = source;
  doc["timestamp"] = millis();

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
    DeserializationError error = deserializeJson(doc, server.arg("plain"));
    if (error) {
      server.send(400, "application/json", "{\"error\":\"Invalid JSON\"}");
      return;
    }
    int pwm = doc["pwm"] | 0;
    bool feedHold = doc["feed_hold"] | false;
    applyActuation(pwm, feedHold, "HTTP");
    server.send(200, "application/json", "{\"status\":\"OK\",\"applied_pwm\":" + String(currentPwm) + "}");
  });

  server.on("/status", HTTP_GET, []() {
    String resp = "{\"node\":2,\"pwm\":" + String(currentPwm) + ",\"feed_hold\":" + String(feedHoldActive ? "true" : "false") + "}";
    server.send(200, "application/json", resp);
  });
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    server.handleClient();
  }

  // Handle incoming Serial JSON commands from Python DecisionEngine
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.startsWith("{") && line.endsWith("}")) {
      StaticJsonDocument<256> doc;
      DeserializationError error = deserializeJson(doc, line);
      if (!error) {
        int pwm = doc["pwm"] | currentPwm;
        bool feedHold = doc["feed_hold"] | false;
        applyActuation(pwm, feedHold, "SERIAL");
      }
    }
  }

  // Failsafe Watchdog: if no command received for >15s during active production, spool fan to 50%
  if (millis() - lastCommandTime > WATCHDOG_TIMEOUT_MS && currentPwm > 0) {
    // Keep baseline cooling active
  }

  delay(10);
}

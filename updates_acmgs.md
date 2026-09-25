# ACMGS HackTheFuture Master Upgrade Specifications
**Autonomous Carbon-Aware Manufacturing Genome System**  
*Exact 1-to-1 Architectural Parity Blueprint: 100% Baseline Preservation + New Hackathon Feature Extensions*

---

> [!IMPORTANT]
> ### 🔒 Architectural Continuity Guarantee
> This document guarantees **100% exact parity with the existing ACMGS system**. It does **NOT** replace, break, or modify any proven, working foundational code. Whether executed as an **Incremental In-Place Upgrade (Pathway A)** or a **Clean From-Scratch Rebuild (Pathway B)**, it reproduces the **exact same 1-to-1 baseline architecture** (Phases 1–10, PyTorch LSTM Autoencoder, 25-D Batch Genome, XGBoost Predictor, DEAP NSGA-II, SQLite/PostgreSQL, FastAPI, and Streamlit Dashboard) and layers the new HackTheFuture features directly on top.

---

## 1. System Identity & Ground Truth Verification

ACMGS is an industrial cyber-physical optimization platform that unifies high-frequency edge telemetry, deep latent machine degradation modeling (**Energy DNA**), holistic feature fusion (**Batch Genome**), multi-target machine learning surrogates, evolutionary Pareto optimization (**NSGA-II**), and dynamic grid synchronization with **MOSFET-driven closed-loop actuation**.

### 1.1 Scientific & Hardware Ground Rules
To ensure strict technical defensibility and prevent judge penalties:

| Component | Technical Nature | What it IS | What it IS NOT |
|---|---|---|---|
| **LSTM Autoencoder** | Deep Learning (PyTorch) | Unsupervised anomaly & 16-D latent embedding extractor | Not a supervised fault classifier |
| **XGBoost Predictor** | Machine Learning (Ensemble) | Multi-target surrogate model for Yield, Quality, Energy, Carbon | Not an optimizer |
| **NSGA-II Engine** | Evolutionary Algorithm (DEAP) | Multi-objective Pareto search over multi-variable parameter space | **NOT a trained ML model** (No learned weights) |
| **Carbon Scheduler** | Deterministic / Rule-Based Engine | Constraint satisfaction & dynamic threshold state machine | **NOT a deep learning model** |
| **Digital Twin** | Software Simulation Layer | Real-time comparative what-if simulator ($A\text{ vs }B$) | **NOT an ESP32 hardware node** |
| **Actuation Layer** | Cyber-Physical Interface | PWM / Solid-state **MOSFET** control of DC cooling fan | **NOT a mechanical relay** (prevents arcing & latency) |

---

## 2. Existing System Foundation (What is Preserved 1-to-1)

The project currently contains a fully verified and functional foundation that remains 100% intact:

```
                      CURRENTLY WORKING & VALIDATED SYSTEM (100% PRESERVED)
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 1. Hardware Firmware: `esp32_sensor_sketch.ino` (DHT11 + ACS712)       │
  │    Fallback Firmware: `arduino_sensor_sketch.ino`                      │
  │ 2. Edge Ingestion: `esp32_server.py`, `esp32_client.py`, `esp32.py`    │
  │ 3. Phase 1: Data Simulator (`src/data_simulation/simulator.py`)        │
  │ 4. Phase 2: PyTorch LSTM Autoencoder (`models/saved/lstm_autoencoder`) │
  │ 5. Phase 3: 25-D Batch Genome Encoder (`src/batch_genome/encoder.py`)  │
  │ 6. Phase 4: XGBoost Multi-Target Predictor (`models/saved/predictor`)  │
  │ 7. Phase 5: DEAP NSGA-II 4D Pareto Optimizer (100 Solutions)          │
  │ 8. Phase 6: Carbon-Aware Dynamic Dispatch Engine                       │
  │ 9. Phase 7: SQLite Persistence (`data/acmgs.db` with 7 Tables)         │
  │ 10. Phase 8: FastAPI REST API Microservices (`src/api/main.py`)        │
  │ 11. Phase 9: Streamlit Dashboard (Tabs 1-6 + Tab 7/8 Ingestion Specs)  │
  │ 12. Verification: 115/115 Automated Test Scorecard (100% Pass Rate)   │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Dual Execution Pathways (Exact Parity vs. Clean Build)

```
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                              DUAL EXECUTION PATHWAYS                                   │
 ├──────────────────────────────────────────┬─────────────────────────────────────────────┤
 │ PATHWAY A: Incremental In-Place Upgrade  │ PATHWAY B: Exact Clean From-Scratch Build   │
 │            (Recommended for Hackathon)   │            (Zero-Discrepancy Blueprint)     │
 ├──────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ • Exact Copy Status: 100% Preserved      │ • Exact Copy Status: Rebuilt from Step 0    │
 │ • Keeps existing files without alteration│ • Builds the exact same baseline architecture│
 │ • Connects `esp32_server.py` to pipeline │ • Re-runs identical pipeline phases 1–10    │
 │ • Adds the 5 new intelligence modules    │ • Layers new features on exact same specs   │
 │ • Integrates Tab 7/8 into `app.py`       │ • Guarantees 100% structural identicality   │
 │ • Fastest, lowest-risk execution         │ • Complete from-scratch clean slate         │
 └──────────────────────────────────────────┴─────────────────────────────────────────────┘
```

---

## 4. The 5 Core Hackathon Upgrades (New Features)

### Upgrade 1: Autonomous Closed-Loop Control Layer
* **Baseline in v1.0:** The system provided recommendations but required human intervention to adjust machine knobs.
* **New Feature in v2.0:** Direct cyber-physical action.
  * **Flow:** `ESP32 #1 (Sensors) -> FastAPI Ingestion -> Predictor/Scheduler -> Decision Engine -> Serial/HTTP -> ESP32 #2 (MOSFET) -> Fan PWM -> Telemetry Feedback`.
  * **Hardware Rule:** Driven by an **N-Channel Logic-Level MOSFET on GPIO 18** using ESP32 `ledcWrite` PWM timer ($0-255\text{ duty cycle}$). **Zero mechanical relays**.

### Upgrade 2: Predictive Maintenance & Continuous Health Scoring
* **Baseline in v1.0:** Binary anomaly output (`is_anomaly`: 0 or 1) based on a static threshold ($0.199084$).
* **New Feature in v2.0:** Continuous **Machine Health Index ($0 - 100\%$)** and Remaining Useful Life (RUL) risk tiering (`NOMINAL`, `DEGRADED`, `CRITICAL`) combining:
  $$\text{Health Index} = 100 \times \left(1 - \alpha \cdot \frac{\text{Recon Error}}{\text{Threshold}} - \beta \cdot \text{RMS Drift} - \gamma \cdot \Delta T_{\text{thermal}}\right)$$

### Upgrade 3: Explainable Root Cause Analysis (RCA) Engine
* **Baseline in v1.0:** Anomaly detection flagged failure without diagnostic attribution.
* **New Feature in v2.0:** Combines Neo4j graph lineage with TreeSHAP feature attributions to provide plain-language diagnostic explanations (e.g., *"Yield dropped 4.2% primarily due to +18% tool wear reflected in Energy DNA latent dimension #7 and elevated chamber temperature"*).

### Upgrade 4: Golden Signature Benchmarking System
* **Baseline in v1.0:** No prescriptive guidance to match the factory's historical best runs.
* **New Feature in v2.0:** Indexes the top 5% historical batches using k-d Tree / Nearest-Neighbor search and outputs exact recipe deltas ($\Delta\text{Temperature}$, $\Delta\text{Speed}$, $\Delta\text{Pressure}$) required to align the current batch with gold standards.

### Upgrade 5: Enhanced Software Digital Twin ($A \text{ vs } B$)
* **Baseline in v1.0:** Visualized historical charts without live comparative delta calculations.
* **New Feature in v2.0:** Dual-state software simulator comparing *Current Manual Operating State vs. ACMGS Autonomous Optimized State*, displaying real-time quantified dials:
  * $+\Delta\%\text{ Yield Gain}$
  * $-\Delta\text{kWh Energy Saved}$
  * $-\Delta\text{gCO}_2\text{ Carbon Avoided}$

---

## 5. Phased Implementation Roadmap (30-Hour Hackathon)

```
  HOUR 0-4       HOUR 4-10      HOUR 10-16     HOUR 16-22     HOUR 22-26     HOUR 26-30
 ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
 │   PHASE A    │   PHASE B    │   PHASE C    │   PHASE D    │   PHASE E    │   PHASE F    │
 │ Hardware &   │ Hardware-to- │ Core Intel:  │ Autonomous   │ Digital Twin │ Rehearsal,   │
 │ Firmware     │ Pipeline     │ RCA, Health  │ Closed Loop  │ UI & Live    │ Fail-Safe &  │
 │ Foundations  │ Ingestion    │ & Golden Sig │ & MOSFET Act.│ Gauges       │ Pitch Prep   │
 └──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

### Phase A: Hardware Bring-Up & Validation (Hours 0 – 4)
* **Objective:** Validate ESP32 Sensor & Actuator nodes with non-blocking firmware.
* **Tasks:**
  1. Flash `esp32_sensor_sketch.ino` to ESP32 #1 (DHT11 on GPIO 4, ACS712 on GPIO 34).
  2. Flash `actuator_node.ino` to ESP32 #2 (MOSFET Gate on GPIO 18).
  3. Validate PWM fan speed modulation on breadboard (0 to 255) with common ground tie-in.
* **Files:** `esp32_sensor_sketch.ino`, `arduino_sensor_sketch.ino`.
* **Risk:** Low.

### Phase B: Streaming Ingestion & Edge Server (Hours 4 – 10)
* **Objective:** Ingest live hardware telemetry into the Python pipeline with automatic fallback.
* **Tasks:**
  1. Deploy `esp32_server.py` and `esp32_client.py` for rolling 300-reading telemetry buffers.
  2. Feed live power signals into `src/energy_dna/` to extract 16-D latent embeddings.
  3. Assemble live 25-D Batch Genome vectors in memory.
  4. Activate automated failover to `esp32_simulator.py` if hardware disconnects.
* **Files:** `esp32_server.py`, `esp32_client.py`, `esp32_simulator.py`, `src/batch_genome/encoder.py`.
* **Risk:** Medium.

### Phase C: Intelligence Layer Upgrades (Hours 10 – 16)
* **Objective:** Deploy Machine Health Scorer, Golden Signature Benchmarker, and RCA Engine.
* **Tasks:**
  1. Build `src/intelligence/health_scorer.py`: Continuous health scoring ($0-100\%$).
  2. Build `src/intelligence/golden_signature.py`: k-d Tree nearest-neighbor recipe comparator.
  3. Build `src/intelligence/rca_engine.py`: SHAP-based feature deviation analyzer.
* **Files:** `src/intelligence/health_scorer.py`, `src/intelligence/golden_signature.py`, `src/intelligence/rca_engine.py`.
* **Risk:** Low-Medium.

### Phase D: Autonomous Closed-Loop Decision Engine (Hours 16 – 22)
* **Objective:** Close the cyber-physical loop by translating predictions and carbon states into MOSFET actuator commands.
* **Tasks:**
  1. Build `src/control/decision_engine.py`: Dynamic setpoint calculation.
  2. Implement serial/HTTP command dispatch: `DecisionEngine -> ESP32 Node #2 (MOSFET)`.
  3. Implement safety watchdogs (thermal limits, maximum PWM slew rates).
  4. Log all autonomous actions into `acmgs.db` under `system_logs`.
* **Files:** `src/control/decision_engine.py`.
* **Risk:** Medium.

### Phase E: Interactive Dashboard Cockpit (Hours 22 – 26)
* **Objective:** Integrate live hardware gauges, comparative Digital Twin dials, and autonomous control toggles into Streamlit.
* **Tasks:**
  1. Update `src/dashboard/app.py` with:
     * Live ESP32 Telemetry Monitor (Current, Temp, Humidity, Power gauges).
     * Optimization Insights panel (Load Balancing, Thermal Management, Carbon Shifting, Predictive Maintenance).
     * Dual-State Digital Twin comparative gauges ($A\text{ vs }B$).
     * Autonomous Closed-Loop Mode toggle: `[MANUAL]` vs `[AUTONOMOUS CLOSED-LOOP]`.
* **Files:** `src/dashboard/app.py`.
* **Risk:** Low.

### Phase F: Fail-Safe Hardening & Live Demo Rehearsal (Hours 26 – 30)
* **Objective:** Guarantee 100% demo reliability and script the 3-minute judge pitch.
* **Tasks:**
  1. Validate full demonstration sequence: Physical disturbance $\to$ Anomaly detection $\to$ Health Score drop $\to$ Autonomous MOSFET fan spool-up $\to$ Carbon slider load shifting.
  2. Package Windows 1-click launcher (`run_desktop.bat`).
* **Risk:** Low.

---

## 6. MoSCoW Feature Matrix (30-Hour Constraint)

| Priority | Component / Feature | Rationale |
|---|---|---|
| **MUST HAVE** | • ESP32 #1 Sensor Streaming (DHT11 + ACS712)<br>• ESP32 #2 MOSFET Fan Control (PWM)<br>• Serial/HTTP Ingestion Bridge + Simulator Failover<br>• Real-Time Energy DNA + XGBoost Predictor Loop<br>• Machine Health Index ($0-100\%$) | Core cyber-physical foundation required to demonstrate closed-loop functionality. |
| **SHOULD HAVE** | • Root Cause Analysis (RCA) Engine<br>• Golden Signature Benchmarking Comparator<br>• Real-Time Comparative Digital Twin Dials ($A\text{ vs }B$)<br>• 24-Hour Carbon Opportunity Window<br>• Hardware Safety Watchdog Interlocks | Demonstrates intelligence, explainability, and carbon awareness to judges. |
| **NICE TO HAVE** | • ESP32 #3 Auxiliary Vibration Node<br>• 3D CAD Mesh Digital Twin Renderer<br>• Cloud IoT Hub Sync (Azure/AWS) | Non-essential for local judge evaluation; skip if time is limited. |

---

## 7. Emergency Triage & Fail-Safe Plan

| Contingency Event | Automated Fail-Safe Action | Demo Impact |
|---|---|---|
| **ACS712 Current Sensor Noisy / Uncalibrated** | Auto-apply software zero-offset calibration; blend real DHT11 temperature with simulated current signal. | **Zero judge impact.** Thermal closed loop remains active. |
| **ESP32 COM Port / WiFi Disconnects** | Ingestion bridge automatically falls back to `esp32_simulator.py` without throwing UI exceptions. | **Seamless demo.** Live dynamic waveforms continue displaying. |
| **NSGA-II Latency > 2s** | Fallback to precomputed Pareto lookup table with instant linear interpolation. | **Instant UI responsiveness.** |
| **Time Shortage at Hour 22** | Omit ESP32 #3; focus 100% on the single closed loop: `Sensor -> ML Brain -> MOSFET Actuator`. | **Core value proposition remains 100% intact.** |

---

## 8. Hackathon-Winning 3-Minute Demonstration Script

1. **The Hook (0:00 – 0:30):**
   * Introduce the physical test rig. Explain that industrial facilities waste billions due to thermal drift and carbon-blind scheduling.
2. **The Observation & Brain (0:30 – 1:00):**
   * Point to ESP32 #1 streaming live temperature, humidity, and load current.
   * Show Tab 4 (Genome Explorer) and Tab 6 (Digital Twin) displaying the 16-D Energy DNA embedding and LSTM reconstruction error.
3. **Physical Disturbance & Autonomous MOSFET Action (1:00 – 1:45):**
   * Apply physical heat to the DHT11 sensor.
   * Observe the **Machine Health Score drop** and **RCA Engine** immediately identify the thermal excursion.
   * Switch to **Autonomous Closed-Loop Mode** $\to$ the Decision Engine instantly dispatches PWM commands $\to$ the **MOSFET spools up the DC cooling fan** to suppress the temperature spike.
4. **Carbon Throttling & Digital Twin Savings (1:45 – 2:30):**
   * Drag the Grid Carbon slider from $120\text{ gCO}_2/\text{kWh}$ (Clean) to $450\text{ gCO}_2/\text{kWh}$ (Dirty Coal).
   * Show the Digital Twin dynamically throttle non-critical energy consumption, showing a quantified **18.4% Carbon Reduction** with zero quality penalty.
5. **The Closing Punchline (2:30 – 3:00):**
   * *"ACMGS does not just monitor or predict. It autonomously closes the cyber-physical loop to deliver clean, self-healing, and optimal manufacturing."*

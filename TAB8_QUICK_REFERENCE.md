# Dashboard Analysis - Quick Visual Summary

## TAB 7: ESP32 Real-Time Data Flow

```
ESP32 Sensors
├── Current (ADC pin)          ──┐
├── Temperature (DHT/TMP)      ──┤
├── Humidity (DHT)             ──├──→ ESP32 Server
└── Power Calculation          ──┤   ├── /api/latest
                                  │   ├── /api/stats
                         (Optional)   └── /api/health
                          Voltage:
                         ADC × (V_ref/1024)
                         Power = Current × 230V AC

                                    ↓
                    st.session_state.esp32_latest
                         (Updated every ~1 sec)
                                    ↓
        ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
        │  Current Gauge  │  Temp Gauge     │  Humidity Gauge │  Power Gauge    │
        │  0-150A range   │  0-100°C range  │  0-100% range   │  0-50kW range   │
        │  3-color zones  │  3-color zones  │  Inverse zones  │  3-color zones  │
        └─────────────────┴─────────────────┴─────────────────┴─────────────────┘
        
        ┌──────────────────────┬──────────────────────┬──────────────────────┐
        │  Statistics Panel    │  Energy Forecast (1h)│  Latest Data Table   │
        │  (Last 60 readings)  │  ├─ Predicted kWh    │  ├─ Current reading  │
        │  ├─ Avg Temp (min)   │  ├─ Carbon impact    │  ├─ Server status    │
        │  ├─ Avg Humidity     │  ├─ Cost estimate    │  └─ Connection URL   │
        │  ├─ Avg Current      │  └─ Avg current      │                      │
        │  ├─ Avg Power        │        (assumptions) │                      │
        │  └─ Total Current    │                      │                      │
        │     integration      │                      │                      │
        └──────────────────────┴──────────────────────┴──────────────────────┘
```

---

## TAB 8: Optimization Insights - Current Flow

```
┌────────────────────────────────────┐
│  Fetch ESP32 Latest from Session   │
│  - current (default: 50A)          │
│  - temperature (default: 35°C)     │
│  - power_watts (default: 11.5kW)   │
│  - humidity (default: 55%)         │
└────────────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│  Define Static Thresholds          │
│  - OPTIMAL_CURRENT = 80A          │
│  - CRITICAL_CURRENT = 120A        │
│  - OPTIMAL_TEMP = 40°C            │
│  - CRITICAL_TEMP = 55°C           │
└────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────────────────────────────────┐
    │                EVALUATE 4 CONDITIONS                           │
    ├────────────────────────────────────────────────────────────────┤
    │                                                                │
    │  IF current > 80A          IF temp > 40°C                     │
    │  └─→ OPP 1:                └─→ OPP 2:                         │
    │      Load Balance              Thermal Mgmt                   │
    │                                                                │
    │  ALWAYS                    IF current > 60A OR temp > 40°C    │
    │  └─→ OPP 3:                └─→ OPP 4:                         │
    │      Carbon Scheduling         Predictive Maint               │
    │                                                                │
    └────────────────────────────────────────────────────────────────┘
         ↓
    ┌────────────────┬────────────────┬────────────────┬────────────────┐
    │   OPP 1        │    OPP 2       │    OPP 3       │    OPP 4       │
    ├────────────────┼────────────────┼────────────────┼────────────────┤
    │ ⚡ Load Balance│ 🌡️ Thermal    │ 🕐 Carbon      │ 🔧 Maintenance│
    │                │                │                │                │
    │Reduction %:    │Cooling %:      │Shifted %:      │Efficiency %:   │
    │((C-80)/C)×100  │(T-40) × 0.5    │Fixed 30%       │Fixed 8%        │
    │                │                │                │                │
    │Savings:        │Savings:        │Savings:        │Savings:        │
    │Calc per opp    │Calc per opp    │Calc per opp    │Calc per opp    │
    │                │                │                │                │
    └────────────────┴────────────────┴────────────────┴────────────────┘
         ↓
    ┌──────────────────────────────────────────────┐
    │  AGGREGATE & DISPLAY                         │
    │  ├─ Total Annual Savings (kWh, $, CO₂)       │
    │  ├─ Payback Period (months)                  │
    │  ├─ Monthly projection graphs                │
    │  └─ Executive summary                        │
    └──────────────────────────────────────────────┘
```

---

## The 4 Optimization Opportunities - Detailed Breakdown

### OPP 1: ⚡ Load Balancing & Peak Shaving
```yaml
Trigger: IF current_current > 80A
Target State: Reduce to 80A

Example (Current = 100A, Power = 23kW):
  ├─ Reduction = ((100-80)/100) × 100 = 20%
  ├─ Energy Saved: 0.20 × 23kW × 24h = 110.4 kWh/day
  ├─ Cost Saved: 110.4 × $0.12 = $13.25/day
  ├─ Carbon Saved: 110.4 × 0.42 = 46.4 kg CO₂/day
  │
  ├─ Annual Impact:
  │  ├─ Energy: 40,296 kWh
  │  ├─ Cost: $4,838
  │  └─ Carbon: 16,924 kg CO₂
  │
  └─ Implementation: Stagger batches, distribute loads, enable demand-response
```

### OPP 2: 🌡️ Thermal Management
```yaml
Trigger: IF current_temp > 40°C
Target State: Reduce to 40°C

Logic: Each °C above 40 adds 0.5% cooling overhead
       (Example assumes linear, but should be non-linear!)

Example (Temp = 50°C, Power = 23kW):
  ├─ Cooling Overhead = (50-40) × 0.5 = 5%
  ├─ Energy Saved: 0.05 × 23kW × 24h = 27.6 kWh/day
  ├─ Cost Saved: 27.6 × $0.12 = $3.31/day
  ├─ Carbon Saved: 27.6 × 0.42 = 11.6 kg CO₂/day
  │
  ├─ Annual Impact:
  │  ├─ Energy: 10,074 kWh
  │  ├─ Cost: $1,209
  │  └─ Carbon: 4,231 kg CO₂
  │
  └─ Implementation: Upgrade cooling, optimize HVAC, predictive control
```

### OPP 3: 🕐 Carbon-Aware Scheduling
```yaml
Trigger: ALWAYS (no condition)
Strategy: Shift 30% of production to off-peak hours

Time Analysis:
  Peak Hours (2-4 PM, 6-8 PM):   0.58 kg CO₂/kWh
  Off-Peak Hours (11 PM-6 AM):   0.28 kg CO₂/kWh
  Difference:                     0.30 kg CO₂/kWh

Example (Power = 23kW constant):
  ├─ Shiftable Load: 23kW × 0.30 = 6.9 kW
  ├─ Daily Carbon Shift: 6.9 × 24 × 0.30 = 49.7 kg CO₂/day
  │   (Or: 23 × 24 × 0.30 × 0.3 = 49.7 kg)
  ├─ Cost Benefit: 23 × 24 × 0.12 × 0.15 = $9.93/day
  │
  ├─ Annual Impact:
  │  ├─ Carbon: 18,141 kg CO₂
  │  └─ Cost: $3,624
  │
  └─ Implementation: Enable scheduler, monitor grid intensity, adjust batches
```

### OPP 4: 🔧 Predictive Maintenance
```yaml
Trigger: IF current > 60A OR temp > 40°C
Assumption: 8% efficiency improvement from maintenance

Example (Power = 23kW):
  ├─ Efficiency Gain: 0.08 × 23kW × 24h = 44.2 kWh/day
  ├─ Cost Saved: 44.2 × $0.12 = $5.30/day
  ├─ Carbon Saved: 44.2 × 0.42 = 18.6 kg CO₂/day
  │
  ├─ Annual Impact:
  │  ├─ Energy: 16,129 kWh
  │  ├─ Cost: $1,935
  │  └─ Carbon: 6,774 kg CO₂
  │
  └─ Implementation: Sensor-based PM, lubrication schedules, component replacement
```

---

## Proposed Improvements to Tab 8

```
CURRENT STATE                           PROPOSED STATE
═══════════════════════════════════════════════════════════════════

1. Static Thresholds                    1. Configurable + Adaptive
   └─ 80A always optimal                   └─ Adjust per equipment
                                           └─ Time-aware scheduling

2. Single-point Snapshot                2. Trend Analysis
   └─ Uses only latest value               ├─ 60-reading window
                                           ├─ Slope detection
                                           ├─ Volatility scoring
                                           └─ Urgency factor

3. Linear Cooling Model                 3. Non-Linear Thermal Curve
   └─ 0.5% per °C always                   ├─ 0.4% for 0-5°C delta
                                           ├─ 0.7% for 5-15°C delta
                                           └─ 1.0% for >15°C delta

4. Single-factor Maintenance            4. Multi-factor Scoring (0-12)
   └─ IF current > 60A                     ├─ Current draw (0-3 pts)
                                           ├─ Temperature (0-3 pts)
                                           ├─ Power volatility (0-2 pts)
                                           └─ Humidity stress (0-1 pt)

5. Unused Humidity                      5. Humidity Optimization
   └─ Collected, not analyzed              ├─ Optimal: 40-70%
                                           ├─ Low: reduces 2-3% efficiency
                                           └─ High: causes condensation

6. Fixed 30% Shiftability               6. Adaptive Shiftability
   └─ Same all day                        ├─ 50% during peak hours
                                          └─ 30% during off-peak

7. No Confidence Scores                 7. Confidence 0.60-0.95
   └─ All recommendations equal           ├─ ✅ High (≥80%)
                                          ├─ ⚠️ Medium (70-80%)
                                          └─ ℹ️ Low (<70%)

8. Hardcoded Defaults                   8. Error Handling
   └─ 50A, 35°C, 11.5kW used if missing   ├─ Use values if available
                                           ├─ Flag as "insufficient data"
                                           └─ Lower confidence scores

9. No Historical Context                9. Payback Calculation
   └─ Single value displayed               ├─ Dynamic based on actual savings
                                           └─ Configurable impl. cost
```

---

## Implementation Priorities

### 🔴 HIGH (Must Have)
- [x] Trend analysis (increasing/decreasing detection)
- [x] Multi-factor maintenance scoring
- [x] Confidence percentages
- [x] Humidity optimization track
- [x] Non-linear thermal curve

### 🟡 MEDIUM (Should Have)
- [ ] Adaptive shiftability (50% peak, 30% off-peak)
- [ ] Real-time grid carbon intensity API integration
- [ ] Configurable threshold UI
- [ ] Historical recommendation tracking
- [ ] Comparison to actual savings

### 🟢 LOW (Nice to Have)
- [ ] Machine learning for threshold optimization
- [ ] Recommendation accuracy feedback loop
- [ ] Facility-specific configuration profiles
- [ ] Predictive anomaly detection
- [ ] Batch-level precision targets

---

## Data Constants Reference

```yaml
ELECTRICITY:
  cost_per_kwh: $0.12  # Consistent value to use
  supply_voltage: 230V AC  # For power calculation

CARBON INTENSITY:
  average: 0.42 kg CO₂/kWh
  peak_hours: 0.58 kg CO₂/kWh (2-4 PM, 6-8 PM)
  off_peak_hours: 0.28 kg CO₂/kWh (11 PM - 6 AM)

OPTIMAL RANGES:
  current: < 80A (warning: > 80A, critical: > 120A)
  temperature: < 40°C (warning: > 40°C, critical: > 55°C)
  humidity: 40-70% (low: < 40%, high: > 70%)

CARBON VALUE:
  $50 per ton CO₂  = $0.05 per kg
  Used: Cost√ = Cost + (Carbon × 0.05)

IMPLEMENTATION COSTS:
  estimated_total: $50,000
  payback_threshold: 24 months
```

---

## Error Handling Strategy

```
┌─────────────────────────────────────────────────────────────┐
│ Data Available?                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  YES ✓                          NO ✗                        │
│  ├─ Use actual values           ├─ Use defaults             │
│  ├─ Generate full report         ├─ Lower all confidences   │
│  ├─ Display confidence 0.7-0.95 ├─ Add warning banner       │
│  └─ Update trends                └─ Set confidence <0.60    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Testing Checklist

### Functional Tests
- [ ] Load Balancing: current 100A → generates $13+/day savings
- [ ] Thermal: temp 50°C → generates $3+/day savings
- [ ] Scheduling: always generates carbon savings
- [ ] Maintenance: current > 60A OR temp > 40°C → triggers
- [ ] Humidity: 35% humidity → includes humidity track

### Threshold Tests
- [ ] current 79A → no Load Balancing track
- [ ] current 81A → generates Load Balancing track
- [ ] temp 39°C → no Thermal track
- [ ] temp 41°C → generates Thermal track

### Trend Tests
- [ ] Rising current (slope > 1) → 20% urgency boost
- [ ] Stable current (slope ≈ 0) → no boost
- [ ] Falling current (slope < -1) → reduced urgency

### Data Quality Tests
- [ ] No ESP32 data → shows defaults with warning
- [ ] Partial ESP32 data → uses what's available
- [ ] API timeout → graceful degradation
- [ ] Invalid JSON → error with recovery

### Calculation Tests
- [ ] Annual extrapolation: daily × 365
- [ ] Payback: $50k / monthly savings × 12
- [ ] Carbon value: kg CO₂ × $0.05
- [ ] Total impact: sum of all opportunities


# Executive Summary: Dashboard Analysis & Optimization Plan

## 📋 Overview

This document summarizes findings from analyzing the ACMGS dashboard (Tab 7 & Tab 8) and provides a comprehensive plan to enhance Tab 8's recommendation engine with dynamic, real-time ESP32 sensor data integration.

---

## 🎯 Key Findings

### Tab 7: ESP32 Real-Time Monitor ✓ (Working Well)
**Status:** Fully functional, captures all required sensor data

**Data Captured:**
- **Current**: 0-150A range (ADC reading)
- **Temperature**: 0-100°C (DHT or TMP sensor)
- **Humidity**: 0-100% (DHT sensor)
- **Power**: 0-50kW (calculated as Current × 230V AC)
- **Timestamp**: ISO format, updates every ~1 second

**Display Format:**
1. Four gauge charts (one per metric)
2. Statistics panel (min, max, avg over 60 readings)
3. Energy forecast (1-hour prediction)
4. Latest readings table with connection status

**API Endpoints:**
- `/api/latest` → Single reading
- `/api/stats?window=60` → Aggregated statistics
- `/api/predict?duration_hours=1` → Energy forecast
- `/api/health` → Connection check

---

### Tab 8: Optimization Insights ⚠️ (Partially Functional, Suboptimal)

**Current Implementation Issues:**

| Issue | Severity | Impact |
|-------|----------|--------|
| Uses hardcoded defaults if ESP32 unavailable | HIGH | Default to 50A/35°C/11.5kW instead of real values |
| Recommendations based on single snapshot | HIGH | No trend analysis = poor decision-making |
| Linear thermal model (0.5%/°C) | MEDIUM | Oversimplified; high temps have non-linear effects |
| Single-condition maintenance checks | MEDIUM | Ignores other warning signs (humidity, volatility) |
| Humidity data captured but unused | MEDIUM | Missing optimization opportunity |
| Fixed 30% production shiftability | MEDIUM | Should adapt based on time of day |
| No confidence metrics | MEDIUM | All recommendations appear equal importance |
| Inconsistent cost data ($0.12 vs $0.15/kWh) | MEDIUM | Creates calculation discrepancies |

---

## 📊 The 4 Current Optimization Opportunities

### 1. ⚡ Load Balancing & Peak Shaving
- **Triggers when:** Current > 80A
- **Target:** Reduce to 80A
- **Example impact (100A → 80A at 23kW):**
  - Daily savings: $13.25 + 46.4 kg CO₂
  - Annual savings: $4,838 + 16,924 kg CO₂
- **Implementation:** Stagger batch starts, distribute loads

### 2. 🌡️ Thermal Management Optimization
- **Triggers when:** Temperature > 40°C
- **Logic:** Cooling overhead = (Temp - 40°C) × 0.5%/°C
- **Example impact (50°C → 40°C at 23kW):**
  - Daily savings: $3.31 + 11.6 kg CO₂
  - Annual savings: $1,209 + 4,231 kg CO₂
- **Implementation:** Upgrade cooling, optimize HVAC

### 3. 🕐 Carbon-Aware Scheduling
- **Always triggers** (no condition)
- **Strategy:** Shift 30% production to off-peak hours (11 PM - 6 AM)
- **Logic:** Peak carbon 0.58 kg CO₂/kWh → Off-peak 0.28 kg CO₂/kWh
- **Example impact (23kW uniform load):**
  - Daily savings: $9.93 + 49.7 kg CO₂
  - Annual savings: $3,624 + 18,141 kg CO₂
- **Implementation:** Enable scheduler, monitor grid intensity

### 4. 🔧 Predictive Maintenance
- **Triggers when:** Current > 60A OR Temperature > 40°C
- **Assumption:** 8% efficiency improvement from maintenance
- **Example impact (23kW):**
  - Daily savings: $5.30 + 18.6 kg CO₂
  - Annual savings: $1,935 + 6,774 kg CO₂
- **Implementation:** Sensor-based PM, replace worn parts

---

## 🔧 Proposed Improvements: 5-Point Enhancement Plan

### 1. **Trend Analysis** (Addresses: single snapshot problem)
- Maintain rolling window of 300 respectable readings (~5 minutes)
- Detect slopes: is current increasing? Temperature warming?
- Add urgency factor: +20% impact boost if trend is adverse
- Display trend indicator in metrics (↑ increasing, ↓ decreasing, → stable)

### 2. **Non-Linear Thermal Model** (Addresses: oversimplified cooling)
```
Current:    Linear 0.5%/°C for any delta
Proposed:   0-5°C delta:    0.4%/°C
            5-15°C delta:   0.7%/°C
            >15°C delta:    1.0%/°C (non-linear acceleration)
```

### 3. **Multi-Factor Maintenance Scoring** (Addresses: incomplete analysis)
Instead of: "IF current > 60A OR temp > 40°C"

Use composite score (0-12 points):
- Current draw (0-3 pts): ≥120A=3, ≥100A=2, ≥80A=1
- Temperature (0-3 pts): ≥55°C=3, ≥45°C=2, ≥40°C=1
- Power volatility (0-2 pts): High variability = mechanical stress
- Humidity deviation (0-1 pt): Out of 40-70% range

Then map score to action: 7-9 pts = 10% efficiency gain (HIGH), 5-6 pts = 8% (MEDIUM), <5 pts = 5% (LOW)

### 4. **Humidity Optimization Track** (Addresses: unused data)
- **New 5th opportunity:** Environmental Humidity Control
- **Triggers when:** Humidity < 40% OR > 70%
- **Impact:** Out-of-range humidity adds 2-3% inefficiency per 10% deviation
- **Implementation:** Install dehumidifier/humidifier, improve ventilation
- **Confidence:** 0.60 (lower, as impact varies by equipment)

### 5. **Adaptive Scheduling & Confidence Scores** (Addresses: rigidity + transparency)
- **Adaptive shiftability:** 50% during peak hours, 30% during off-peak
- **Confidence levels:** Visual indicators (✅ High ≥80%, ⚠️ Medium 70-80%, ℹ️ Low <70%)
- **Confidence based on:**
  - Criticality: Critical state = 0.95, Warning state = 0.85, Normal = 0.70
  - Data quality: Assume 0.9x if data is stale/missing
  - Trend type: Latest trend matches historical pattern = +0.05

---

## 💻 Implementation Details

### New Module: `src/optimization/recommender.py`

**Core Classes:**
```python
DynamicRecommender
├── __init__(config=None)           # Initialize with optional thresholds
├── add_reading(sensor_data)        # Build 300-reading history
├── get_trends(window=60)           # Analyze slopes & volatility
├── evaluate_conditions(latest)     # Check all thresholds
├── generate_recommendations()      # Create all opportunities
└── calculate_payback(daily_cost)   # Financial analysis

Opportunity (dataclass)
├── rank: int (1-5, sorted by impact)
├── title: str
├── description: str
├── impact_pct: float
├── savings_kwh_daily: float
├── savings_cost_daily: float
├── savings_carbon_daily: float
├── implementation: str
├── timeline: str
└── confidence: float (0.60-0.95)
```

**Key Methods:**

1. **`add_reading()`** - Appends sensor data to rolling window
   ```python
   # Input: {'current': 85.2, 'temperature': 42.1, 'humidity': 55.0, ...}
   # Updates: self.recent_readings deque (maxlen=300)
   ```

2. **`get_trends(window=60)`** - Analyzes last N readings
   ```python
   # Output: {
   #   'current_trend': 'increasing|stable|decreasing',
   #   'temp_trend': 'warming|stable|cooling',
   #   'power_trend': 'increasing|stable|decreasing',
   #   'volatility': {'current': 5.2, 'temperature': 1.1, 'power': 450}
   # }
   ```

3. **`generate_recommendations(latest, trends)`** - Main logic
   - Evaluates 5 conditions (load, thermal, scheduling, maintenance, humidity)
   - Calculates impact for each triggered opportunity
   - Assigns confidence scores
   - Sorts by value × confidence
   - Returns list of Opportunity objects

4. **`calculate_payback(daily_cost, impl_cost=50000)`** - ROI
   ```python
   # If daily_cost = $10/day over 30 days = $300/month
   # payback = $50,000 / $300 = 166.7 months
   ```

---

## 📈 Implementation Roadmap

### Phase 1: Module Creation (2-3 days)
1. Create `src/optimization/recommender.py`
2. Implement all 5 opportunity calculations
3. Write unit tests for each calculation
4. Validate against current Tab 8 numbers

### Phase 2: Dashboard Integration (2-3 days)
1. Import DynamicRecommender in Tab 8
2. Replace static opportunity generation
3. Update UI to show confidence indicators
4. Add trend visualization to metrics
5. Integration test with live ESP32 data

### Phase 3: Testing & Optimization (2-3 days)
1. Unit test: Each opportunity type
2. Integration test: ESP32 API connection
3. Regression test: All 4 original opportunities still work
4. Performance test: Recommendation generation < 50ms
5. UAT: Real-world feedback from facility operators

### Phase 4: Documentation (1 day)
1. Update README with new features
2. Document configurable thresholds
3. Create tuning guide for different equipment types
4. Add confidence level interpretation guide

---

## 🎬 Next Steps

### Immediate (Today)
- [ ] Review analysis documents
- [ ] Gather stakeholder feedback on proposed changes
- [ ] Prioritize which improvements to implement first

### Short-term (This Week)
- [ ] Create `recommender.py` module
- [ ] Implement Unit tests
- [ ] Begin Tab 8 integration

### Medium-term (This Sprint)
- [ ] Complete integration & testing
- [ ] Configuration UI for threshold tuning
- [ ] Historical performance comparison
- [ ] Documentation completion

### Long-term (Future Enhancements)
- [ ] Machine learning for dynamic threshold optimization
- [ ] Real-time grid carbon intensity API integration
- [ ] Per-facility configuration profiles
- [ ] Recommendation feedback loop (tracking actual vs predicted savings)

---

## 📚 Documentation Files Created

1. **`DASHBOARD_OPTIMIZATION_ANALYSIS.md`** - Full technical analysis
   - Tab 7 sensor data structure
   - Tab 8 current implementation details
   - 4 opportunity explanations with formulas
   - Complete `recommender.py` code
   - Testing strategy

2. **`TAB8_QUICK_REFERENCE.md`** - Visual reference guide
   - Data flow diagrams
   - Opportunity breakdowns with examples
   - Before/after comparison table
   - Threshold reference sheet
   - Checklist for testing and deployment

3. **`IMPLEMENTATION_DIFF.md`** - Code change guide
   - Side-by-side before/after code
   - Key differences summary
   - Integration checklist
   - Deployment checklist

4. **`EXECUTIVE_SUMMARY.md`** - This document
   - High-level overview
   - Key findings
   - Proposed improvements
   - Roadmap
   - Next steps

---

## 💡 Expected Outcomes

### For End Users
- ✅ More accurate recommendations (based on real data, not defaults)
- ✅ Prioritized opportunities (confidence indicators help decide what to act on)
- ✅ Better insights (trends show if situation is improving or worsening)
- ✅ New optimization opportunity (humidity management)

### For Operations
- ✅ Data-driven decision making (trend analysis + multi-factor scoring)
- ✅ Reduced energy waste (adaptive scheduling based on time)
- ✅ Better equipment health (maintenance scored on 4 factors, not 2)
- ✅ Improved ROI tracking (dynamic payback based on actual savings)

### For Development
- ✅ Modular architecture (recommender.py is separate from UI)
- ✅ Testable calculations (unit tests for each formula)
- ✅ Configurable system (thresholds can be tuned per facility)
- ✅ Extensible design (easy to add 6th, 7th opportunities later)

---

## Cost-Benefit Summary

| Aspect | Investment | Benefit |
|--------|-----------|---------|
| **Development Time** | 1-2 weeks | Improved recommendations for entire user base |
| **Code Complexity** | +500 lines | Modular, well-documented, testable |
| **Performance Impact** | Negligible | Caching + efficient numpy calculations |
| **Maintenance Burden** | Low | Centralized logic in recommender.py |
| **User Training** | Minimal | UI mostly the same, just better results |

**ROI:** Improved recommendations lead to better facility decisions → higher energy savings + lower carbon → better business outcomes.

---

## Questions & Support

For implementation questions, refer to:
- **Module Design:** See `IMPLEMENTATION_DIFF.md` "New Module Creation"
- **Formula Details:** See `DASHBOARD_OPTIMIZATION_ANALYSIS.md` "PART 3.2"
- **Visual Diagrams:** See `TAB8_QUICK_REFERENCE.md`
- **Testing Approach:** See `DASHBOARD_OPTIMIZATION_ANALYSIS.md` "PART 5"

---

**Document Generated:** March 21, 2026  
**Status:** Ready for Implementation  
**Approval:** [Pending stakeholder review]


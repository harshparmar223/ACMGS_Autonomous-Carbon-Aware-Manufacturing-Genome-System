# ACMGS Track A Requirements Analysis

## Executive Summary

The ACMGS codebase implements a **Multi-Objective Evolutionary Optimization** approach focused on carbon-aware manufacturing rather than the Track A **Knowledge Graph Reasoning** approach. 

**Key Finding:** ACMGS uses **ML-based recommendations with energy sensor analysis**, NOT knowledge graph-based reasoning or conversational batch forensics.

---

## Track A Requirements vs. ACMGS Implementation

### 1. Manufacturing Knowledge Graph Reasoning ❌ NOT IMPLEMENTED

#### Required
- Semantic knowledge graph with relationships:
  - Asset –[Produces] → Batch
  - Batch –[Has] → Energy Patterns
  - Energy Patterns –[indicates] → Asset Health Events
  - Process Parameters –[influences] → Energy Patterns
  - Batch –[compared against] → Golden Signature
  - Golden Signature –[optimized for] → Objectives
  - Raw Material –[affects] → Yield Outcome
  - Anomaly –[triggered by] → Process Drift

#### ACMGS Implementation
- **Relational Database Structure** (instead of knowledge graph):
  ```sql
  batches
    ├── energy_embeddings (batch_id FK)
    ├── genome_vectors (batch_id FK)
    ├── predictions (batch_id FK)
    └── pareto_solutions (carbon_intensity reference)
  ```

- **No explicit entity relationships** defined as semantic triples
- **No cross-entity reasoning** (e.g., "Asset X's declining energy pattern indicates wear")
- **No graph traversal** for causal analysis

#### What ACMGS Does Instead
1. **Energy DNA (Phase 2):** LSTM Autoencoder extracts energy signatures, but only for anomaly detection
2. **Genome Encoding (Phase 3):** Merges process params + energy embeddings + material properties into 25-dim vectors
3. **Optimization (Phase 5):** NSGA-II finds Pareto-optimal configurations
4. **Scheduling (Phase 6):** Static rule-based scheduling: 
   - HIGH carbon zone (≥400 gCO₂/kWh) → minimize energy
   - LOW carbon zone (≤150 gCO₂/kWh) → maximize yield
   - MEDIUM zone → balance all objectives

**Verdict:** ACMGS does NOT use knowledge graph reasoning.

---

### 2. Natural Language Forecasting Summaries ⚠️ PARTIAL

#### Required
- Generated natural language explanations of forecasts
- Explanations grounded in domain knowledge
- Narrative descriptions of why predictions were made

#### ACMGS Implementation

**Found in `src/optimization/recommender.py`:**
```python
@dataclass
class Recommendation:
    title: str                    # e.g., "⚡ Load Balancing & Peak Shaving"
    description: str              # Template-based NL description
    icon: str
    daily_savings: Dict[str, float]
    confidence: str              # "HIGH", "MEDIUM", "LOW"
    factors: List[str]           # Evidence list
```

**Example from Dashboard (Tab 8):**
```
Title: "⚡ Load Balancing & Peak Shaving"
Description: "Reduce current draw from 95.2A to 50A through load distribution. 
             Currently drawing 90% above optimal."
Factors:
  - Current 95.2A (target: 50A)
  - Reduction potential: 90%
  - Current trend: increasing
```

**Limitations:**
- ✅ Descriptions are human-readable
- ✅ Confidence levels included
- ❌ Descriptions are **parameterized templates**, not generated from reasoning
- ❌ No NLG engine (no semantic-to-text generation)
- ❌ No explanation chain (why parameters led to recommendation)

**Verdict:** ACMGS has **natural language outputs but NOT natural language generation from reasoning**.

---

### 3. Conversational Batch Forensics ❌ NOT IMPLEMENTED

#### Required
- Why-why analysis tools in plain English
- Conversational interface for investigating batch issues
- Root cause identification dialogue
- Interactive questioning (chatbot-style)

#### ACMGS Implementation

**What exists:**
- Batch replay feature in Dashboard (Tab 6):
  ```python
  # Can navigate through 2,000-batch history
  # Shows process parameters, yield, quality, energy, carbon
  # BUT: No interactive debugging
  ```

- Anomaly detection (Energy DNA):
  ```python
  # Flags batches where reconstruction error > threshold
  # But NO explanation of WHY anomaly occurred
  ```

- Energy embedding visualization:
  ```python
  # Shows batch health via reconstruction error
  # BUT: No interpretable reasoning about health degradation
  ```

**What's MISSING:**
- ❌ No conversational interface
- ❌ No "why-why" questioning engine
- ❌ No batch forensics module
- ❌ No root cause analysis beyond anomaly score
- ❌ No reasoning about parameter drift → performance impact

**Verdict:** ACMGS does NOT implement conversational batch forensics.

---

### 4. Knowledge-Based Recommendations ⚠️ PARTIAL (ML-Based Instead)

#### Required
- Recommendations grounded in expert knowledge base
- Evidence/reasoning for each recommendation
- Knowledge-derived confidence levels

#### ACMGS Implementation

**Recommender System (`src/optimization/recommender.py`):**
```python
def get_recommendations(current_a, temp_c, humidity_pct, power_w):
    """Generate 5 recommendation tracks:
    1. Load Balancing & Peak Shaving
    2. Thermal Management
    3. Carbon-Aware Scheduling
    4. Predictive Maintenance
    5. Humidity Optimization
    """
```

**Evidence Included:**
```python
Recommendation(
    title="🌡️ Thermal Management",
    description=f"Optimize cooling to lower from {temp_c:.1f}°C",
    confidence="HIGH",  # Based on multi-factor score
    factors=[
        "Temperature 51.2°C (optimal <35°C)",
        "Efficiency gain potential: 16%",
        "Temperature trending: increasing"
    ]
)
```

**How Recommendations Work:**
1. Read ESP32 sensor data (current, temperature, humidity, power)
2. Analyze trends (is current increasing/decreasing?)
3. Score maintenance likelihood (0-12 points):
   - Current draw stress (0-4 pts)
   - Temperature (0-4 pts)
   - Humidity (0-2 pts)
   - Trend analysis (0-2 pts)
4. Generate recommendations with confidence based on score
5. Calculate daily/annual savings impact

**Strengths:**
- ✅ Includes "factors" showing why recommendation triggered
- ✅ Confidence levels (HIGH/MEDIUM/LOW)
- ✅ Evidence-based on sensor readings
- ✅ Includes savings quantification

**Limitations:**
- ❌ **Not knowledge-based** (no expert knowledge rules)
- ❌ **Purely threshold & heuristic-driven** (if current > 50A then recommend load balancing)
- ❌ **No knowledge representation** (no rules engine, no inference)
- ❌ **Not reasoning** (just matching patterns to thresholds)
- ❌ **No reasoning trace** explaining decision chain

**Verdict:** ACMGS provides **recommendations with evidence, but NOT knowledge-based reasoning**.

---

## What ACMGS Actually Implements

### Core Architecture: 9 Phases

```
Phase 1: Data Simulation
  ↓
Phase 2: Energy DNA (LSTM Autoencoder)
  ↓
Phase 3: Batch Genome Encoding (25-dim vectors)
  ↓
Phase 4: Multi-Target Prediction (XGBoost)
  ├─ Predicts: yield, quality, energy consumption
  ↓
Phase 5: NSGA-II Evolutionary Optimization
  ├─ Objectives: maximize yield/quality, minimize energy/carbon
  ├─ Output: Pareto frontier
  ↓
Phase 6: Carbon-Aware Scheduling
  ├─ Classifies grid carbon intensity (HIGH/MEDIUM/LOW)
  ├─ Selects optimal config from Pareto frontier
  ↓
Phase 7: SQLite Database
  ├─ Stores all batch data, embeddings, genomes, predictions
  ↓
Phase 8: FastAPI REST Layer
  ├─ Endpoints: /health, /batches/{id}, /schedule/{carbon}
  ↓
Phase 9: Streamlit Dashboard
  ├─ Real-time monitoring
  ├─ Pareto frontier visualization
  ├─ Optimization recommendations (dynamic recommender)
  ├─ ESP32 sensor data integration
```

### Key Features Implemented

| Feature | Status | Location |
|---------|--------|----------|
| Manufacturing data simulation | ✅ | `src/data_simulation/simulator.py` |
| Energy anomaly detection (LSTM) | ✅ | `src/energy_dna/trainer.py` |
| Batch genome construction | ✅ | `src/batch_genome/encoder.py` |
| ML prediction (yield, quality, energy) | ✅ | `src/prediction/predictor.py` |
| Multi-objective optimization | ✅ | `src/optimization/optimizer.py` (NSGA-II) |
| Carbon-aware scheduling | ✅ | `src/carbon_scheduler/scheduler.py` |
| Dynamic recommendations | ✅ | `src/optimization/recommender.py` |
| Real-time ESP32 integration | ✅ | `src/esp32/realtime_processor.py` |
| Knowledge graph | ❌ | N/A |
| Conversational interface | ❌ | N/A |
| Batch forensics reasoning | ❌ | N/A |
| Natural language generation | ❌ | N/A |

---

## Detailed Feature Mapping

### ✅ IMPLEMENTED: Dynamic Energy Optimization Recommender

**File:** `src/optimization/recommender.py`

**5 Recommendation Tracks:**
1. **Load Balancing & Peak Shaving**
   - Triggered: current draw > 50A
   - Action: Reduce to 50A via load distribution
   - Savings: Percentage-based on overage

2. **Thermal Management**
   - Triggered: temperature > 35°C
   - Formula: Non-linear efficiency gain (0.4-1.0% per °C)
   - Savings: Exponential with temperature delta

3. **Carbon-Aware Scheduling**
   - Triggered: current > 40A
   - Action: Shift 30-50% to off-peak hours
   - Benefit: Lower grid carbon intensity + cost

4. **Predictive Maintenance**
   - Triggered: multi-factor maintenance score ≥ 6
   - Score factors: current stress, temperature, humidity, trends
   - Savings: 2-8% efficiency recovery

5. **Humidity Optimization**
   - Triggered: humidity < 30% or > 80%
   - Action: Stabilize to 40-70% range
   - Savings: 3% efficiency

**Integration:**
- Real-time in Dashboard Tab 8
- Uses ESP32 sensor data (300-reading history)
- Calculates cumulative annual impact
- Includes ROI analysis (payback period)

---

### ⚠️ PARTIAL: Carbon-Aware Decision Logic

**File:** `src/carbon_scheduler/scheduler.py`

**3-Zone Classification:**
```
LOW (≤150 gCO₂/kWh):
  → Maximize yield + quality
  Strategy: Run at full production

MEDIUM (150-400):
  → Balance all objectives
  Strategy: Optimize efficiency

HIGH (≥400):
  → Minimize energy + carbon
  Strategy: Energy conservation
```

**Selection Logic:**
- Loads Pareto-optimal solutions from Phase 5
- Applies zone-specific fitness: 
  - HIGH: minimize (energy + carbon)
  - LOW: maximize (yield + quality)
  - MEDIUM: maximize (yield + quality - energy - carbon)

**Limitations:**
- Static thresholds (not adaptive)
- No reasoning about *why* this schedule is best
- No trace of decision-making

---

### ❌ NOT IMPLEMENTED: Batch Forensics

**Missing Capabilities:**
- No "root cause analysis" module
- No batch-level anomaly investigation
- No parameter-to-outcome tracing
- No "what-if" analysis for past batches
- No conversational debugging interface

**What could be built:**
```
User: "Why did Batch_0042 have poor yield?"
System: "Analyzing batch parameters...
  - Temperature 280°C (optimal: 200-250°C)
  - Pressure 8.5 bar (nominal: 5.0 bar)
  - Both exceed optimal ranges
  - Prediction: yield should be 0.65 (actual: 0.58)
  - Root cause: Over-constrained process parameters
  Recommendation: Reduce temperature to 230°C and pressure to 6 bar"
```

The infrastructure exists to build this, but no implementation present.

---

## Assessment by Track A Requirement

### 1. Manufacturing Knowledge Graph Reasoning
- **Status:** ❌ NOT IMPLEMENTED
- **Alternative Used:** Relational database + NSGA-II optimization
- **Gap:** No semantic representation of manufacturing knowledge

### 2. Natural Language Forecasting Summaries
- **Status:** ⚠️ PARTIAL (Text output, not generated from reasoning)
- **Alternative Used:** Template-based descriptions
- **Gap:** No natural language generation engine

### 3. Conversational Batch Forensics
- **Status:** ❌ NOT IMPLEMENTED
- **Alternative Used:** Static batch replay in dashboard
- **Gap:** No interactive investigation tools

### 4. Knowledge-Based Recommendations
- **Status:** ⚠️ PARTIAL (Recommendations exist, but ML/heuristic-based not knowledge-based)
- **Alternative Used:** Sensor thresholds + multi-factor scoring
- **Gap:** No explicit knowledge base or inference engine

---

## Architectural Comparison

### ACMGS Approach (Actual)
```
Sensor Data → Thresholds → Score → Confidence → Recommendation
```
- Data-driven and heuristic-based
- Optimizes for carbon-aware manufacturing
- Uses evolutionary algorithms (NSGA-II)
- Focus: Finding Pareto-optimal process parameters

### Track A Approach (Required)
```
Knowledge Graph → Entity Relationships → Reasoning Engine 
  → Inference Chain → Explanation → Recommendation
```
- Knowledge-driven
- Focuses on semantic understanding
- Uses symbolic reasoning
- Focus: Explaining manufacturing decisions

---

## Recommendation for Track A Alignment

If the goal is to implement Track A requirements, the following would need to be added to ACMGS:

### 1. Knowledge Graph Layer
```python
# Implement semantic triples:
Asset_001 --produces--> Batch_042
Batch_042 --has-energy-pattern--> pattern_sine_50hz
pattern_sine_50hz --indicates--> wear_indication_level_3
Batch_042 --compared-against--> golden_signature_asset_001
```

### 2. Reasoning Engine
```python
# Example inference rule:
IF batch.energy_pattern.drift > threshold 
   AND batch.material.grade == high
   THEN anomaly.root_cause = "asset_degradation"
   WITH confidence = 0.85
```

### 3. Natural Language Generation
```python
# Semantic-to-text:
Entity: Batch_042
Relations: [high_energy_drift, material_incompatibility, asset_wear]
Generated: "Batch 42 consumed 15% more energy than golden signature 
           due to potential asset bearing wear and material hardness mismatch"
```

### 4. Conversational Interface
```python
# Interactive batch forensics:
User: "Analyze batch performance"
System: Loads batch graph, applies inference rules, generates NL explanation
User: "Why did yield decrease?"
System: Traces causal chain through knowledge graph, explains
```

---

## Summary Table

| Requirement | Track A Asks | ACMGS Implements | Match |
|-------------|-------------|-----------------|-------|
| Knowledge Graph | Semantic triples | Relational DB | ❌ No |
| Entity Relationships | Asset→Batch→Patterns | Part of genome vector | ❌ No |
| Causal Reasoning | Why-why analysis | Threshold matching | ❌ No |
| NL Explanations | Generated from reasoning | Template text | ⚠️ Partial |
| Batch Forensics | Interactive investigation | Static replay | ❌ No |
| Conversational | Chat-based Q&A | Web dashboard | ❌ No |
| Recommendations | Knowledge-based | ML/Sensor-based | ⚠️ Partial |
| Evidence Tracing | Decision chain | Single confidence | ⚠️ Partial |

---

## Conclusion

**ACMGS is fundamentally an ML-based multi-objective optimization system, NOT a knowledge graph reasoning system.**

The system excels at:
- ✅ Finding Pareto-optimal manufacturing configurations
- ✅ Carbon-aware scheduling based on grid conditions
- ✅ Real-time sensor-based optimization recommendations
- ✅ Batch anomaly detection via energy patterns

The system does NOT implement Track A's:
- ❌ Manufacturing knowledge graphs with semantic relationships
- ❌ Conversational batch forensics
- ❌ Knowledge-based reasoning engines
- ❌ Natural language generation from semantic representations

**To align with Track A, a completely separate "Knowledge Reasoning" module would need to be developed, operating alongside (or replacing) the current ML optimization system.**

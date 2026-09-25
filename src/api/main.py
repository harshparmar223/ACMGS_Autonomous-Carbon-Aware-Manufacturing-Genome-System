"""
Phase 8: ACMGS FastAPI REST API

Endpoints
---------
GET  /health                      — system status + DB table counts
GET  /batches/{batch_id}          — fetch one batch from DB
GET  /genome/{batch_id}           — fetch 25-dim genome vector from DB
GET  /schedule/{carbon_intensity} — carbon-aware manufacturing recommendation
GET  /pareto                      — all Pareto-optimal manufacturing solutions
POST /predict                     — run XGBoost prediction on a genome vector
GET  /db/summary                  — DB row counts + file size

Run with:
    python -m src.api.main
    or
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import pickle
import sqlite3
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from config.settings import DB_PATH, MODELS_DIR, PROCESSED_DIR
from src.api.schemas import (
    BatchResponse,
    DBSummaryResponse,
    GenomeResponse,
    HealthResponse,
    ParetoResponse,
    PredictRequest,
    PredictResponse,
    ScheduleResponse,
)
from src.carbon_scheduler import classify_carbon_zone
from src.database import (
    get_batch,
    get_db_summary,
    get_genome,
    get_latest_schedule,
    get_pareto_solutions,
)
from src.utils.logger import get_logger

logger = get_logger("api")

# ─── App creation ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="ACMGS API",
    description=(
        "Autonomous Carbon-aware Manufacturing Genome System — "
        "REST interface for all pipeline phases (1-7)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Lazy-loaded predictor ────────────────────────────────────────────────────

_predictor = None

def get_predictor():
    """Load the XGBoost predictor once and cache it."""
    global _predictor
    if _predictor is None:
        pkl_path = os.path.join(MODELS_DIR, "predictor.pkl")
        if not os.path.isfile(pkl_path):
            raise HTTPException(status_code=503, detail="Predictor model not found. Run Phase 4 first.")
        with open(pkl_path, "rb") as f:
            _predictor = pickle.load(f)
        logger.info("Predictor loaded from %s" % pkl_path)
    return _predictor


# =============================================================================
#  GET /health
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """
    System health check.
    Returns API status, DB connectivity, and table row counts.
    """
    try:
        summary = get_db_summary()
        db_ok = True
    except Exception as e:
        logger.error("DB health check failed: %s" % e)
        summary = {}
        db_ok = False

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        phase=8,
        db_connected=db_ok,
        table_counts=summary,
        message="ACMGS API is running — Phase 8 active",
    )


# =============================================================================
#  GET /batches/{batch_id}
# =============================================================================

@app.get("/batches/{batch_id}", response_model=BatchResponse, tags=["Data"])
def get_batch_endpoint(batch_id: str):
    """
    Retrieve a single manufacturing batch record by its ID.

    - **batch_id**: e.g. `BATCH_0042`
    """
    row = get_batch(batch_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Batch '%s' not found" % batch_id)
    # map 'yield' key to 'yield_' to avoid Python keyword clash
    row_out = dict(row)
    return BatchResponse(**row_out)


# =============================================================================
#  GET /genome/{batch_id}
# =============================================================================

@app.get("/genome/{batch_id}", response_model=GenomeResponse, tags=["Data"])
def get_genome_endpoint(batch_id: str):
    """
    Retrieve the 25-dimensional genome vector for a batch.

    Genome layout:
    - `[0:5]`  — process params (temperature, pressure, speed, feed_rate, humidity)
    - `[5:8]`  — material props (density, hardness, grade)
    - `[8:24]` — energy DNA (16-dim LSTM latent vector from Phase 2)
    - `[24]`   — carbon intensity
    """
    genome = get_genome(batch_id)
    if genome is None:
        raise HTTPException(status_code=404, detail="Genome for '%s' not found" % batch_id)
    return GenomeResponse(batch_id=batch_id, genome=genome, dims=len(genome))


# =============================================================================
#  GET /schedule/{carbon_intensity}
# =============================================================================

@app.get("/schedule/{carbon_intensity}", response_model=ScheduleResponse, tags=["Scheduling"])
def get_schedule_endpoint(carbon_intensity: float):
    """
    Get the optimal manufacturing schedule for the given grid carbon intensity.

    - Carbon ≤ 150 gCO2/kWh → **LOW** zone: maximize production
    - 150 < Carbon < 400   → **MEDIUM** zone: balance efficiency
    - Carbon ≥ 400 gCO2/kWh → **HIGH** zone: minimize energy/carbon

    - **carbon_intensity**: current grid carbon in gCO2/kWh (e.g. `250.0`)
    """
    if carbon_intensity < 0 or carbon_intensity > 1000:
        raise HTTPException(
            status_code=422,
            detail="carbon_intensity must be between 0 and 1000 gCO2/kWh"
        )

    zone = classify_carbon_zone(carbon_intensity)
    sched = get_latest_schedule(zone)

    if sched is None:
        raise HTTPException(
            status_code=503,
            detail="No schedule found for zone '%s'. Run Phase 6 first." % zone
        )

    # Build human-readable recommendation
    recs = {
        "LOW"    : "Grid is clean — run at FULL PRODUCTION. Maximize yield and quality.",
        "MEDIUM" : "Grid is moderate — BALANCED MODE. Optimize for efficiency.",
        "HIGH"   : "Grid is dirty — CONSERVATION MODE. Minimize energy and carbon emissions.",
    }

    return ScheduleResponse(
        zone=zone,
        carbon_intensity=carbon_intensity,
        schedule_temperature=sched.get("schedule_temperature"),
        schedule_pressure=sched.get("schedule_pressure"),
        schedule_speed=sched.get("schedule_speed"),
        schedule_feed_rate=sched.get("schedule_feed_rate"),
        schedule_humidity=sched.get("schedule_humidity"),
        schedule_pred_yield=sched.get("schedule_pred_yield"),
        schedule_pred_quality=sched.get("schedule_pred_quality"),
        schedule_pred_energy=sched.get("schedule_pred_energy"),
        schedule_pred_carbon=sched.get("schedule_pred_carbon"),
        recommendation=recs[zone],
    )


# =============================================================================
#  GET /pareto
# =============================================================================

@app.get("/pareto", response_model=ParetoResponse, tags=["Optimization"])
def list_pareto(
    limit: int = Query(100, ge=1, le=500, description="Max solutions to return"),
    min_yield: float = Query(0.0, ge=0.0, le=1.0, description="Filter: min predicted yield"),
    max_carbon: float = Query(1000.0, ge=0.0, description="Filter: max predicted carbon (gCO2/kWh)"),
):
    """
    List Pareto-optimal manufacturing configurations from Phase 5.

    Optional filters:
    - **min_yield**: only return solutions where `pred_yield >= min_yield`
    - **max_carbon**: only return solutions where `pred_carbon <= max_carbon`
    - **limit**: cap the number of results
    """
    df = get_pareto_solutions()
    if df is None or df.empty:
        raise HTTPException(status_code=503, detail="No Pareto solutions found. Run Phase 5 first.")

    # Apply filters
    df = df[df["pred_yield"] >= min_yield]
    df = df[df["pred_carbon"] <= max_carbon]
    df = df.head(limit)

    # Round floats for clean JSON
    df = df.round(4)

    run_id = str(df["run_id"].iloc[0]) if "run_id" in df.columns else None

    return ParetoResponse(
        count=len(df),
        run_id=run_id,
        solutions=df.to_dict(orient="records"),
    )


# =============================================================================
#  POST /predict
# =============================================================================

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_endpoint(req: PredictRequest):
    """
    Run the XGBoost predictor on a genome vector.

    Supply EITHER:
    - **batch_id**: looks up the stored genome from the DB
    - **genome**: a raw 25-dimensional float list

    Returns predicted yield, quality, and energy consumption.
    """
    model = get_predictor()

    if req.genome is not None:
        # Use provided genome directly
        if len(req.genome) != 25:
            raise HTTPException(
                status_code=422,
                detail="genome must have exactly 25 dimensions, got %d" % len(req.genome)
            )
        genome_arr = np.array(req.genome, dtype=np.float32).reshape(1, -1)
        batch_id = req.batch_id
    elif req.batch_id is not None:
        genome_vals = get_genome(req.batch_id)
        if genome_vals is None:
            raise HTTPException(
                status_code=404,
                detail="Genome for batch '%s' not found" % req.batch_id
            )
        genome_arr = np.array(genome_vals, dtype=np.float32).reshape(1, -1)
        batch_id = req.batch_id
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'batch_id' or 'genome' in the request body."
        )

    preds = model.predict(genome_arr)[0]

    return PredictResponse(
        batch_id=batch_id,
        pred_yield=round(float(preds[0]), 4),
        pred_quality=round(float(preds[1]), 4),
        pred_energy=round(float(preds[2]), 2),
    )


# =============================================================================
#  GET /db/summary
# =============================================================================

@app.get("/db/summary", response_model=DBSummaryResponse, tags=["System"])
def db_summary():
    """
    Database summary — row counts for all tables + DB file size in bytes.
    """
    try:
        summary = get_db_summary()
    except Exception as e:
        raise HTTPException(status_code=503, detail="DB unavailable: %s" % str(e))

    db_size = os.path.getsize(DB_PATH) if os.path.isfile(DB_PATH) else 0

    return DBSummaryResponse(
        batches=summary.get("batches", 0),
        energy_embeddings=summary.get("energy_embeddings", 0),
        genome_vectors=summary.get("genome_vectors", 0),
        predictions=summary.get("predictions", 0),
        pareto_solutions=summary.get("pareto_solutions", 0),
        carbon_schedules=summary.get("carbon_schedules", 0),
        pipeline_runs=summary.get("pipeline_runs", 0),
        db_size_bytes=db_size,
    )


# =============================================================================
#  v2.0 Intelligence & Cyber-Physical Endpoints
# =============================================================================

@app.get("/api/machine-health", tags=["v2.0 Intelligence"])
def get_machine_health(
    recon_error: float = 0.045,
    current_rms: float = 12.5,
    temperature: float = 38.0
):
    """
    Computes Continuous Machine Health Index (0-100%) and 3 Operational Tiers:
    NOMINAL (>=75%), DEGRADED (45-74%), CRITICAL (<45%).
    """
    from src.intelligence.health_scorer import MachineHealthScorer
    scorer = MachineHealthScorer()
    rep = scorer.evaluate(recon_error=recon_error, current_rms=current_rms, temperature=temperature)
    return {
        "health_index": rep.health_index,
        "tier": rep.tier.value,
        "recon_penalty": rep.recon_penalty,
        "current_penalty": rep.current_penalty,
        "temp_penalty": rep.temp_penalty,
        "status_summary": rep.status_summary,
        "action_recommendation": rep.action_recommendation,
        "color_hex": rep.color_hex,
        "maintenance_window_hours": rep.maintenance_window_hours
    }


@app.get("/api/rca", tags=["v2.0 Intelligence"])
def get_root_cause_analysis(
    temperature: float = 245.0,
    pressure: float = 4.2,
    speed: float = 1600.0,
    feed_rate: float = 1.10,
    recon_error: float = 0.42
):
    """
    TreeSHAP Explainable Root Cause Analysis.
    Translates 25-D genome variances into plain-English operator diagnostics.
    """
    from src.intelligence.rca_engine import RCAEngine
    rca = RCAEngine()
    genome = np.array([
        temperature, pressure, speed, feed_rate, 45.0,
        7.85, 200.0, 2.0,
        0.1, -0.2, 0.3, -0.1, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        250.0
    ], dtype=np.float32)
    rep = rca.explain(genome, target_index=0, recon_error=recon_error)
    return {
        "target": rep.target_name,
        "predicted_value": rep.predicted_value,
        "baseline_value": rep.baseline_value,
        "primary_driver": rep.primary_driver_text,
        "plain_english": rep.plain_english_diagnosis,
        "top_attributions": [
            {
                "feature": a.display_name,
                "value": a.feature_value,
                "attribution_pct": a.attribution_pct,
                "direction": a.direction,
                "interpretation": a.interpretation
            }
            for a in rep.top_attributions
        ]
    }


@app.get("/api/golden-signature", tags=["v2.0 Intelligence"])
def get_golden_signature(
    temperature: float = 245.0,
    pressure: float = 4.2,
    speed: float = 1600.0,
    feed_rate: float = 1.10
):
    """
    Golden Signature Nearest-Neighbor Comparator.
    Calculates prescriptive parameter adjustments to match top 5% historical runs.
    """
    from src.intelligence.golden_signature import GoldenSignatureEngine
    engine = GoldenSignatureEngine()
    rec = engine.find_nearest_golden_recipe(
        temperature=temperature,
        pressure=pressure,
        speed=speed,
        feed_rate=feed_rate
    )
    return {
        "nearest_batch_id": rec.nearest_batch_id,
        "target_yield": rec.target_yield,
        "target_quality": rec.target_quality,
        "target_energy": rec.target_energy,
        "current_params": rec.current_params,
        "target_params": rec.target_params,
        "deltas": rec.deltas,
        "prescriptive_text": rec.prescriptive_text,
        "similarity_score_pct": rec.similarity_score_pct
    }


@app.post("/api/digital-twin/simulate", tags=["v2.0 Intelligence"])
def simulate_digital_twin(
    carbon_intensity: float = 350.0,
    inject_defect: bool = False,
    defect_minute: float = 18.0
):
    """
    Dual-State Industrial Digital Twin (Plan A vs Plan B Matrix).
    Simultaneously computes live comparative economic deltas and sunk energy saved.
    """
    from src.digital_twin.twin_engine import DigitalTwinEngine
    twin = DigitalTwinEngine()
    comp = twin.compute_dual_state(
        grid_carbon_intensity=carbon_intensity,
        defect_injected=inject_defect,
        defect_minute=defect_minute
    )
    return {
        "plan_a": {
            "name": comp.plan_a.name,
            "yield": comp.plan_a.yield_rate,
            "quality": comp.plan_a.quality_score,
            "energy_kwh": comp.plan_a.energy_kwh,
            "carbon_kg": comp.plan_a.carbon_emissions_kg,
            "scrap_rate_pct": comp.plan_a.scrap_rate_pct
        },
        "plan_b": {
            "name": comp.plan_b.name,
            "yield": comp.plan_b.yield_rate,
            "quality": comp.plan_b.quality_score,
            "energy_kwh": comp.plan_b.energy_kwh,
            "carbon_kg": comp.plan_b.carbon_emissions_kg,
            "scrap_rate_pct": comp.plan_b.scrap_rate_pct
        },
        "deltas": {
            "yield_gain_pct": comp.delta_yield_pct,
            "energy_reduced_pct": comp.delta_energy_pct,
            "carbon_avoided_pct": comp.delta_carbon_pct,
            "scrap_energy_preserved_kwh": comp.scrap_energy_preserved_kwh
        },
        "economic_summary": comp.economic_benefit_summary,
        "sunk_report": {
            "abort_triggered": comp.sunk_report.abort_triggered,
            "abort_minute": comp.sunk_report.abort_minute,
            "sunk_energy_saved_kwh": comp.sunk_report.sunk_energy_saved_kwh,
            "sunk_carbon_avoided_kg": comp.sunk_report.sunk_carbon_avoided_kg,
            "material_status": comp.sunk_report.material_status,
            "summary": comp.sunk_report.status_summary
        } if comp.sunk_report else None
    }


# =============================================================================
#  Production Continuity & Human-in-the-Loop Endpoints
# =============================================================================

# In-memory singletons for live API demo session
from src.services.machine_state import MachineState, SensorReadingContract
from src.services.production_continuity import ProductionContinuityManager
from src.services.recovery import RecoveryManager
from src.services.command_service import CommandService
from src.safety.safety_rules import SafetyRuleEngine

_live_machine_state = MachineState(
    machine_id="MACHINE_01",
    temperature=42.0,
    humidity=48.0,
    voltage=230.0,
    current=12.2,
    power=2806.0,
    energy_kwh=112.5,
    carbon_emissions_kg=28.12,
    anomaly_score=0.035,
    health_index=94.5,
    quality_score=0.94,
    golden_similarity=96.2,
    batch_progress_pct=65.0
)
_continuity_mgr = ProductionContinuityManager()
_recovery_mgr = RecoveryManager()
_command_svc = CommandService()
_safety_engine = SafetyRuleEngine()


@app.get("/api/machine/{machine_id}/state", tags=["Production Continuity"])
def get_machine_state(machine_id: str):
    """Returns the unified MachineState (single source of truth)."""
    if machine_id != _live_machine_state.machine_id and machine_id != "latest":
        raise HTTPException(status_code=404, detail=f"Machine {machine_id} not registered.")
    return _live_machine_state.to_dict()


@app.post("/api/production-continuity", tags=["Production Continuity"])
def evaluate_production_continuity(
    temperature: Optional[float] = None,
    current_a: Optional[float] = None,
    health_index: Optional[float] = None,
    progress_pct: Optional[float] = None,
    is_sustained_defect: bool = False
):
    """
    Evaluates in-flight production problems: Continue vs Correct vs Controlled Stop vs Resume.
    Prioritizes safe corrective trimming to prevent unnecessary downtime & sunk batch loss.
    """
    if temperature is not None: _live_machine_state.temperature = temperature
    if current_a is not None: _live_machine_state.current = current_a
    if health_index is not None: _live_machine_state.health_index = health_index
    if progress_pct is not None: _live_machine_state.batch_progress_pct = progress_pct

    decision = _continuity_mgr.evaluate_decision(
        state=_live_machine_state,
        is_sustained_defect=is_sustained_defect
    )
    
    req_id = None
    if decision.requires_human_approval:
        req_id = _command_svc.submit_for_approval(_live_machine_state.machine_id, decision)

    res = decision.to_dict()
    res["approval_request_id"] = req_id
    return res


@app.post("/api/approval", tags=["Production Continuity"])
def operator_approval(request_id: str, approve: bool, operator_id: str = "OPERATOR_01"):
    """
    Human-in-the-loop interface: Operator APPROVES or REJECTS an AI process recommendation.
    Approved commands are re-validated by the Safety Rule Engine before physical dispatch.
    """
    success, msg = _command_svc.process_operator_action(
        request_id=request_id,
        approve=approve,
        operator_id=operator_id
    )
    if not success:
        return {"status": "BLOCKED_OR_FAILED", "message": msg}
    return {"status": "SUCCESS", "message": msg}


@app.post("/api/resume", tags=["Production Continuity"])
def resume_production_endpoint(machine_id: str = "MACHINE_01"):
    """
    Executes pre-flight verification and resumes production from valid saved checkpoint.
    """
    verif = _recovery_mgr.verify_pre_flight(
        machine_id=machine_id,
        current_temp=_live_machine_state.temperature,
        current_a=_live_machine_state.current
    )
    success, msg = _recovery_mgr.resume_production(machine_id, _live_machine_state, verif)
    return {
        "resumed": success,
        "message": msg,
        "verification": {
            "ready_to_resume": verif.ready_to_resume,
            "notes": verif.verification_notes
        }
    }


@app.get("/api/machine/{machine_id}/recovery-state", tags=["Production Continuity"])
def get_recovery_state(machine_id: str):
    """Fetches saved checkpoint context and pre-flight readiness for a machine."""
    chk = _recovery_mgr._active_checkpoints.get(machine_id)
    verif = _recovery_mgr.verify_pre_flight(
        machine_id=machine_id,
        current_temp=_live_machine_state.temperature,
        current_a=_live_machine_state.current
    )
    return {
        "has_active_checkpoint": chk is not None,
        "checkpoint": {
            "checkpoint_id": chk.checkpoint_id,
            "batch_id": chk.batch_id,
            "batch_progress_pct": chk.batch_progress_pct,
            "elapsed_minutes": chk.elapsed_minutes,
            "energy_kwh": chk.energy_kwh_accumulated,
            "stop_reason": chk.stop_reason
        } if chk else None,
        "pre_flight_readiness": verif.ready_to_resume,
        "verification_notes": verif.verification_notes
    }



# =============================================================================
#  Entry point — run directly
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting ACMGS API server on http://0.0.0.0:8000")
    logger.info("Docs available at: http://localhost:8000/docs")
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )

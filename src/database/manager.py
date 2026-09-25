"""
Phase 7: Database Manager

SQLite database for ACMGS — stores all pipeline data for persistence,
querying, and API access.

Tables:
    batches          — raw batch process + material + target data (Phase 1)
    energy_embeddings — per-batch 16-dim LSTM DNA vectors (Phase 2)
    genome_vectors   — per-batch 25-dim genome vectors (Phase 3)
    predictions      — model predictions per batch (Phase 4)
    pareto_solutions — Pareto-optimal manufacturing configs (Phase 5)
    carbon_schedules — carbon-zone scheduling decisions (Phase 6)
    pipeline_runs    — audit log: when each phase ran and its outcome
"""

import os
import sqlite3
import json
from datetime import datetime
from contextlib import contextmanager
from typing import List, Optional

import numpy as np
import pandas as pd

from config.settings import DB_PATH, SIMULATED_DIR, PROCESSED_DIR
from src.utils.logger import get_logger

logger = get_logger("database")


# ─── Connection helper ────────────────────────────────────────────────────────

@contextmanager
def get_connection():
    """Yield a SQLite connection that auto-commits or rolls back."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row          # dict-like rows
    conn.execute("PRAGMA journal_mode=WAL")  # safer concurrent access
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─── Schema creation ──────────────────────────────────────────────────────────

def create_tables():
    """Create all tables if they don't already exist."""
    with get_connection() as conn:
        conn.executescript("""
            -- Phase 1: raw batch data
            CREATE TABLE IF NOT EXISTS batches (
                batch_id          TEXT PRIMARY KEY,
                temperature       REAL,
                pressure          REAL,
                speed             REAL,
                feed_rate         REAL,
                humidity          REAL,
                material_density  REAL,
                material_hardness REAL,
                material_grade    INTEGER,
                yield             REAL,
                quality           REAL,
                energy_consumption REAL,
                carbon_intensity  REAL,
                created_at        TEXT DEFAULT (datetime('now'))
            );

            -- Phase 2: LSTM energy DNA embeddings (stored as JSON array)
            CREATE TABLE IF NOT EXISTS energy_embeddings (
                batch_id     TEXT PRIMARY KEY,
                embedding    TEXT NOT NULL,   -- JSON array of 16 floats
                is_anomaly   INTEGER DEFAULT 0,
                recon_error  REAL,
                created_at   TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
            );

            -- Phase 3: 25-dim genome vectors (stored as JSON array)
            CREATE TABLE IF NOT EXISTS genome_vectors (
                batch_id    TEXT PRIMARY KEY,
                genome      TEXT NOT NULL,    -- JSON array of 25 floats
                created_at  TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
            );

            -- Phase 4: multi-target predictions
            CREATE TABLE IF NOT EXISTS predictions (
                batch_id           TEXT PRIMARY KEY,
                pred_yield         REAL,
                pred_quality       REAL,
                pred_energy        REAL,
                actual_yield       REAL,
                actual_quality     REAL,
                actual_energy      REAL,
                created_at         TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
            );

            -- Phase 5: Pareto-optimal solutions
            CREATE TABLE IF NOT EXISTS pareto_solutions (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id            TEXT NOT NULL,
                temperature       REAL,
                pressure          REAL,
                speed             REAL,
                feed_rate         REAL,
                humidity          REAL,
                material_density  REAL,
                material_hardness REAL,
                material_grade    REAL,
                pred_yield        REAL,
                pred_quality      REAL,
                pred_energy       REAL,
                pred_carbon       REAL,
                created_at        TEXT DEFAULT (datetime('now'))
            );

            -- Phase 6: carbon-aware scheduling decisions
            CREATE TABLE IF NOT EXISTS carbon_schedules (
                id                        INTEGER PRIMARY KEY AUTOINCREMENT,
                carbon_intensity          REAL NOT NULL,
                zone                      TEXT NOT NULL,
                schedule_temperature      REAL,
                schedule_pressure         REAL,
                schedule_speed            REAL,
                schedule_feed_rate        REAL,
                schedule_humidity         REAL,
                schedule_pred_yield       REAL,
                schedule_pred_quality     REAL,
                schedule_pred_energy      REAL,
                schedule_pred_carbon      REAL,
                schedule_material_density  REAL,
                schedule_material_hardness REAL,
                schedule_material_grade    REAL,
                created_at                TEXT DEFAULT (datetime('now'))
            );

            -- Audit log: one row per pipeline phase execution
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                phase       INTEGER NOT NULL,
                phase_name  TEXT NOT NULL,
                status      TEXT NOT NULL,   -- 'success' | 'failed'
                details     TEXT,            -- JSON with metrics / error msg
                started_at  TEXT,
                finished_at TEXT DEFAULT (datetime('now'))
            );

            -- Phase 10 / v2.0: Cyber-Physical Actuator & Decision Logs
            CREATE TABLE IF NOT EXISTS actuator_logs (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp         TEXT DEFAULT (datetime('now')),
                temperature       REAL,
                current_rms       REAL,
                recon_error       REAL,
                predicted_quality REAL,
                pwm_duty          INTEGER,
                fan_speed_pct     REAL,
                feed_hold         INTEGER,
                status            TEXT,
                sunk_energy_kwh   REAL,
                sunk_carbon_kg    REAL,
                reason            TEXT
            );
        """)
    logger.info("Database tables created/verified at %s", DB_PATH)


# ─── Phase 1 loader ───────────────────────────────────────────────────────────

def load_batches_from_csv(csv_path: Optional[str] = None) -> int:
    """Insert all rows from batch_data.csv into the batches table.

    Returns number of rows inserted.
    """
    path = csv_path or os.path.join(SIMULATED_DIR, "batch_data.csv")
    df = pd.read_csv(path)

    cols = [
        "batch_id", "temperature", "pressure", "speed", "feed_rate",
        "humidity", "material_density", "material_hardness", "material_grade",
        "yield", "quality", "energy_consumption", "carbon_intensity",
    ]
    df = df[cols]

    with get_connection() as conn:
        # Delete child rows first to satisfy FK constraints, then parent
        conn.execute("DELETE FROM predictions")
        conn.execute("DELETE FROM genome_vectors")
        conn.execute("DELETE FROM energy_embeddings")
        conn.execute("DELETE FROM batches")
        df.to_sql("batches", conn, if_exists="append", index=False,
                  method="multi", chunksize=500)

    logger.info("Loaded %d batches into DB", len(df))
    return len(df)


# ─── Phase 2 loader ───────────────────────────────────────────────────────────

def load_embeddings_from_npy(
    emb_path: Optional[str] = None,
    batch_ids_path: Optional[str] = None,
) -> int:
    """Insert energy DNA embeddings into energy_embeddings table."""
    emb_path = emb_path or os.path.join(SIMULATED_DIR, "energy_embeddings.npy")
    ids_path = batch_ids_path or os.path.join(PROCESSED_DIR, "batch_ids.npy")

    embeddings = np.load(emb_path)                          # (2000, 16)
    batch_ids = np.load(ids_path, allow_pickle=True)        # (2000,)

    rows = [
        (str(bid), json.dumps(emb.tolist()), 0, None)
        for bid, emb in zip(batch_ids, embeddings)
    ]

    with get_connection() as conn:
        conn.execute("DELETE FROM energy_embeddings")
        conn.executemany(
            "INSERT INTO energy_embeddings (batch_id, embedding, is_anomaly, recon_error)"
            " VALUES (?, ?, ?, ?)",
            rows,
        )

    logger.info("Loaded %d energy embeddings into DB", len(rows))
    return len(rows)


# ─── Phase 3 loader ───────────────────────────────────────────────────────────

def load_genomes_from_npy(
    genome_path: Optional[str] = None,
    batch_ids_path: Optional[str] = None,
) -> int:
    """Insert genome vectors into genome_vectors table."""
    genome_path = genome_path or os.path.join(PROCESSED_DIR, "genome_vectors.npy")
    ids_path = batch_ids_path or os.path.join(PROCESSED_DIR, "batch_ids.npy")

    genomes = np.load(genome_path)                          # (2000, 25)
    batch_ids = np.load(ids_path, allow_pickle=True)        # (2000,)

    rows = [
        (str(bid), json.dumps(genome.tolist()))
        for bid, genome in zip(batch_ids, genomes)
    ]

    with get_connection() as conn:
        conn.execute("DELETE FROM genome_vectors")
        conn.executemany(
            "INSERT INTO genome_vectors (batch_id, genome) VALUES (?, ?)",
            rows,
        )

    logger.info("Loaded %d genome vectors into DB", len(rows))
    return len(rows)


# ─── Phase 4 loader ───────────────────────────────────────────────────────────

def load_predictions_from_csv(pred_path: Optional[str] = None) -> int:
    """Insert Phase 4 predictions into the predictions table."""
    path = pred_path or os.path.join(PROCESSED_DIR, "predictions.csv")
    if not os.path.exists(path):
        logger.warning("predictions.csv not found at %s — skipping", path)
        return 0
    df = pd.read_csv(path)
    cols = ["batch_id", "pred_yield", "pred_quality", "pred_energy",
            "actual_yield", "actual_quality", "actual_energy"]
    df = df[cols]
    with get_connection() as conn:
        conn.execute("DELETE FROM predictions")
        df.to_sql("predictions", conn, if_exists="append", index=False,
                  method="multi", chunksize=500)
    logger.info("Loaded %d predictions into DB", len(df))
    return len(df)


# ─── Phase 5 loader ───────────────────────────────────────────────────────────

def load_pareto_from_csv(
    pareto_path: Optional[str] = None,
    run_id: Optional[str] = None,
) -> int:
    """Insert Pareto solutions into pareto_solutions table."""
    path = pareto_path or os.path.join(SIMULATED_DIR, "pareto_solutions.csv")
    df = pd.read_csv(path)

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    df["run_id"] = run_id

    with get_connection() as conn:
        conn.execute("DELETE FROM pareto_solutions WHERE run_id = ?", (run_id,))
        df.to_sql("pareto_solutions", conn, if_exists="append", index=False,
                  method="multi", chunksize=200)

    logger.info("Loaded %d Pareto solutions (run_id=%s) into DB", len(df), run_id)
    return len(df)


# ─── Phase 6 loader ───────────────────────────────────────────────────────────

def load_schedules_from_csv(schedule_path: Optional[str] = None) -> int:
    """Insert carbon schedule decisions into carbon_schedules table."""
    path = schedule_path or os.path.join(SIMULATED_DIR, "carbon_schedule_demo.csv")
    df = pd.read_csv(path)

    with get_connection() as conn:
        conn.execute("DELETE FROM carbon_schedules")
        df.to_sql("carbon_schedules", conn, if_exists="append", index=False,
                  method="multi", chunksize=200)

    logger.info("Loaded %d schedule rows into DB", len(df))
    return len(df)


# ─── Audit log ────────────────────────────────────────────────────────────────

def log_pipeline_run(
    phase: int,
    phase_name: str,
    status: str,
    details: Optional[dict] = None,
    started_at: Optional[str] = None,
) -> int:
    """Record a pipeline phase execution in the audit log.

    Returns the new row id.
    """
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO pipeline_runs
               (phase, phase_name, status, details, started_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                phase,
                phase_name,
                status,
                json.dumps(details) if details else None,
                started_at or datetime.now().isoformat(),
            ),
        )
    return cur.lastrowid


# ─── Query helpers ────────────────────────────────────────────────────────────

def get_batch(batch_id: str) -> Optional[dict]:
    """Return a single batch row as a dict, or None if not found."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
        ).fetchone()
    return dict(row) if row else None


def get_genome(batch_id: str) -> Optional[List[float]]:
    """Return the 25-dim genome vector for a batch as a Python list."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT genome FROM genome_vectors WHERE batch_id = ?", (batch_id,)
        ).fetchone()
    return json.loads(row["genome"]) if row else None


def get_latest_schedule(zone: Optional[str] = None) -> Optional[dict]:
    """Return the most recent carbon schedule, optionally filtered by zone."""
    query = "SELECT * FROM carbon_schedules"
    params: tuple = ()
    if zone:
        query += " WHERE zone = ?"
        params = (zone.upper(),)
    query += " ORDER BY id DESC LIMIT 1"
    with get_connection() as conn:
        row = conn.execute(query, params).fetchone()
    return dict(row) if row else None


def get_pareto_solutions(run_id: Optional[str] = None) -> pd.DataFrame:
    """Return Pareto solutions as a DataFrame, optionally for a specific run."""
    query = "SELECT * FROM pareto_solutions"
    params: tuple = ()
    if run_id:
        query += " WHERE run_id = ?"
        params = (run_id,)
    query += " ORDER BY id"
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


def get_db_summary() -> dict:
    """Return row counts for all tables — useful for health checks."""
    tables = [
        "batches", "energy_embeddings", "genome_vectors",
        "predictions", "pareto_solutions", "carbon_schedules", "pipeline_runs",
    ]
    summary = {}
    with get_connection() as conn:
        for table in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            summary[table] = count
    return summary


# ─── Full pipeline loader ─────────────────────────────────────────────────────

def run_database_pipeline() -> dict:
    """
    Phase 7 entry point.

    Creates all tables, then loads output from Phases 1–6 into the database.
    Returns a summary dict of rows loaded per table.
    """
    started = datetime.now().isoformat()
    logger.info("=" * 60)
    logger.info("PHASE 7: DATABASE PIPELINE STARTING")
    logger.info("=" * 60)
    logger.info("DB path: %s", DB_PATH)

    create_tables()

    results = {}

    # Phase 1
    try:
        results["batches"] = load_batches_from_csv()
        log_pipeline_run(1, "Data Simulation", "success",
                         {"rows": results["batches"]}, started)
    except Exception as e:
        logger.error("Failed to load batches: %s", e)
        results["batches"] = 0

    # Phase 2
    try:
        results["energy_embeddings"] = load_embeddings_from_npy()
        log_pipeline_run(2, "Energy DNA", "success",
                         {"rows": results["energy_embeddings"]}, started)
    except Exception as e:
        logger.error("Failed to load embeddings: %s", e)
        results["energy_embeddings"] = 0

    # Phase 3
    try:
        results["genome_vectors"] = load_genomes_from_npy()
        log_pipeline_run(3, "Batch Genome", "success",
                         {"rows": results["genome_vectors"]}, started)
    except Exception as e:
        logger.error("Failed to load genomes: %s", e)
        results["genome_vectors"] = 0

    # Phase 4
    try:
        results["predictions"] = load_predictions_from_csv()
        log_pipeline_run(4, "Prediction Models", "success",
                         {"rows": results["predictions"]}, started)
    except Exception as e:
        logger.error("Failed to load predictions: %s", e)
        results["predictions"] = 0

    # Phase 5 — wipe old pareto rows first so re-runs don't accumulate
    try:
        with get_connection() as conn:
            conn.execute("DELETE FROM pareto_solutions")
        results["pareto_solutions"] = load_pareto_from_csv()
        log_pipeline_run(5, "Optimization", "success",
                         {"rows": results["pareto_solutions"]}, started)
    except Exception as e:
        logger.error("Failed to load pareto solutions: %s", e)
        results["pareto_solutions"] = 0

    # Phase 6
    try:
        results["carbon_schedules"] = load_schedules_from_csv()
        log_pipeline_run(6, "Carbon Scheduler", "success",
                         {"rows": results["carbon_schedules"]}, started)
    except Exception as e:
        logger.error("Failed to load schedules: %s", e)
        results["carbon_schedules"] = 0

    summary = get_db_summary()

    logger.info("=" * 60)
    logger.info("PHASE 7: COMPLETE")
    logger.info("=" * 60)
    logger.info("Database summary:")
    for table, count in summary.items():
        logger.info("  %-25s : %d rows", table, count)

    return summary


if __name__ == "__main__":
    summary = run_database_pipeline()

    print("\n" + "=" * 50)
    print("DATABASE LOADED SUCCESSFULLY")
    print("=" * 50)
    for table, count in summary.items():
        print(f"  {table:<25} : {count} rows")

    # Quick spot-check queries
    print("\n--- Spot Checks ---")
    batch = get_batch("BATCH_0000")
    if batch:
        print(f"BATCH_0000 carbon_intensity : {batch['carbon_intensity']:.1f} gCO2/kWh")
        print(f"BATCH_0000 yield            : {batch['yield']:.4f}")

    genome = get_genome("BATCH_0000")
    if genome:
        print(f"BATCH_0000 genome dim       : {len(genome)} (expected 25)")

    sched = get_latest_schedule("LOW")
    if sched:
        print(f"Latest LOW schedule yield   : {sched['schedule_pred_yield']:.4f}")
    print("=" * 50)

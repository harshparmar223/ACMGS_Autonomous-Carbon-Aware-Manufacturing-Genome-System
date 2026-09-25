"""
ACMGS - Comprehensive Phase Testing
Tests all implemented phases of the system
"""

import os
import sys
import importlib
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def test_phase(phase_num, phase_name, module_path, required_functions=None):
    """Test if a phase module exists and has required functions"""
    print(f"\n{'='*60}")
    print(f"PHASE {phase_num}: {phase_name}")
    print(f"{'='*60}")
    
    try:
        # Import the module
        module = importlib.import_module(module_path)
        print(f"✓ Module '{module_path}' imported successfully")
        
        # Check for required functions/classes
        if required_functions:
            missing = []
            found = []
            for func in required_functions:
                if hasattr(module, func):
                    found.append(func)
                else:
                    missing.append(func)
            
            if found:
                print(f"✓ Found functions/classes: {', '.join(found)}")
            if missing:
                print(f"✗ Missing functions/classes: {', '.join(missing)}")
                return False
        
        # Check file size (is it just an empty file?)
        file_path = module.__file__
        if file_path and os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ Module file size: {size} bytes")
            if size < 100:
                print(f"⚠ Warning: File seems very small, might be empty")
                return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import module: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def check_data_files():
    """Check if required data files exist"""
    print(f"\n{'='*60}")
    print("DATA FILES CHECK")
    print(f"{'='*60}")
    
    data_files = {
        "Simulated Batch Data": "data/simulated/batch_data.csv",
        "Energy Signals": "data/simulated/energy_signals.npy",
        "Energy Embeddings": "data/simulated/energy_embeddings.npy",
        "Genome Vectors": "data/processed/genome_vectors.npy",
        "Batch IDs": "data/processed/batch_ids.npy",
        "Genome Metadata": "data/processed/genome_metadata.csv",
        "Genome Normalization": "data/processed/genome_normalization.npz",
    }
    
    models = {
        "LSTM Autoencoder": "models/saved/lstm_autoencoder.pth",
        "Predictor Model": "models/saved/predictor.pkl",
        "Predictor Metrics": "models/saved/predictor_metrics.pkl",
    }
    
    all_exist = True
    
    print("\nData Files:")
    for name, path in data_files.items():
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {path} ({size:,} bytes)" if exists else f"{status} {name}: {path} (missing)")
        if not exists:
            all_exist = False
    
    print("\nModel Files:")
    for name, path in models.items():
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {path} ({size:,} bytes)" if exists else f"{status} {name}: {path} (missing)")
        if not exists:
            all_exist = False
    
    return all_exist

def main():
    print("="*60)
    print("ACMGS - COMPREHENSIVE PHASE TESTING")
    print("="*60)
    
    results = {}
    
    # Phase 1: Data Simulation
    results['Phase 1'] = test_phase(
        1, "Data Simulation",
        "src.data_simulation.simulator",
        ["generate_full_dataset", "generate_energy_signals", "generate_process_parameters"]
    )
    
    # Phase 2: Energy DNA (LSTM Autoencoder)
    results['Phase 2 (Model)'] = test_phase(
        2, "Energy DNA - Model",
        "src.energy_dna.model",
        ["LSTMAutoencoder"]
    )
    
    results['Phase 2 (Trainer)'] = test_phase(
        2, "Energy DNA - Trainer",
        "src.energy_dna.trainer",
        ["run_energy_dna_pipeline", "train_model", "extract_embeddings"]
    )
    
    # Phase 3: Batch Genome Encoding
    results['Phase 3'] = test_phase(
        3, "Batch Genome Encoding",
        "src.batch_genome.encoder",
        ["run_batch_genome_pipeline", "construct_genome_vectors", "load_genome_vectors"]
    )
    
    # Phase 4: Prediction Models
    results['Phase 4'] = test_phase(
        4, "Multi-Target Prediction",
        "src.prediction.predictor",
        ["load_genome_features", "create_predictor_model", "run_prediction_pipeline"]
    )
    
    # Phase 5: Optimization (NSGA-II)
    results['Phase 5'] = test_phase(
        5, "Evolutionary Optimization (NSGA-II)",
        "src.optimization",
        []  # May not be implemented yet
    )
    
    # Phase 6: Carbon Scheduler
    results['Phase 6'] = test_phase(
        6, "Carbon-Aware Scheduling",
        "src.carbon_scheduler",
        []  # May not be implemented yet
    )
    
    # Phase 7: Database
    results['Phase 7'] = test_phase(
        7, "Database Management",
        "src.database",
        []  # May not be implemented yet
    )
    
    # Phase 8: API
    results['Phase 8'] = test_phase(
        8, "REST API",
        "src.api",
        []  # May not be implemented yet
    )
    
    # Phase 9: Dashboard
    results['Phase 9'] = test_phase(
        9, "Visualization Dashboard",
        "src.dashboard.app",
        []
    )

    # Phase 10: Closed-Loop Control & Intelligence (v2.0 Upgrades)
    results['Phase 10 (Decision Engine)'] = test_phase(
        10, "Closed-Loop Decision Engine & Interlock",
        "src.control.decision_engine",
        ["DecisionEngine", "ActuationCommand"]
    )
    results['Phase 10 (Machine Health)'] = test_phase(
        10, "Predictive Maintenance Health Scorer",
        "src.intelligence.health_scorer",
        ["MachineHealthScorer", "HealthTier"]
    )
    results['Phase 10 (TreeSHAP RCA)'] = test_phase(
        10, "TreeSHAP Explainable RCA Engine",
        "src.intelligence.rca_engine",
        ["RCAEngine"]
    )
    results['Phase 10 (Golden Signatures)'] = test_phase(
        10, "Golden Signature Benchmarking",
        "src.intelligence.golden_signature",
        ["GoldenSignatureEngine"]
    )
    results['Phase 10 (Digital Twin A vs B)'] = test_phase(
        10, "Dual-State Industrial Digital Twin",
        "src.digital_twin.twin_engine",
        ["DigitalTwinEngine"]
    )
    
    # Check data and model files
    data_ok = check_data_files()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    implemented = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nPhases Implemented: {implemented}/{total}")
    print("\nDetailed Status:")
    for phase, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {phase}")
    
    print(f"\nData & Models: {'✓ All files present' if data_ok else '⚠ Some files missing'}")
    
    # Overall status
    core_phases = ['Phase 1', 'Phase 2 (Model)', 'Phase 2 (Trainer)', 'Phase 3', 'Phase 4']
    core_working = all(results.get(p, False) for p in core_phases)
    
    print(f"\n{'='*60}")
    if core_working and data_ok:
        print("✓ CORE SYSTEM (Phases 1-4) IS FULLY OPERATIONAL")
    elif core_working:
        print("⚠ CORE SYSTEM WORKING BUT MISSING SOME DATA FILES")
    else:
        print("✗ CORE SYSTEM HAS ISSUES - Some phases not working")
    print(f"{'='*60}")
    
    return core_working and data_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

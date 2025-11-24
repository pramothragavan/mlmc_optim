"""Test all dataset/model combinations from the paper using MLMC training.

This runs train_mlmc.py with test overrides.
"""

import sys
import subprocess
from pathlib import Path

FRACTION = 0.02  # Use 10% of data for quick tests

def run_mlmc_training_test(config_path, epochs=2):
    """Run MLMC training script with test overrides."""
    project_root = Path(__file__).parent.parent
    train_script = project_root / "examples/examples_src/train_mlmc.py"

    cmd = [
        sys.executable,
        str(train_script),
        "--config", str(config_path),
        "--epochs", str(epochs),
        f"--train_subset", str(FRACTION),
        f"--test_subset", str(FRACTION),
        "--device", "cpu",
    ]
    
    print(f"\n  Running: python train_mlmc.py --config {config_path.name} --epochs {epochs} --train_subset {FRACTION} ...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
    
    if result.returncode == 0:
        print("  ✓ MLMC training completed successfully")
        return True
    else:
        print(f"  ✗ MLMC training failed with exit code {result.returncode}")
        if result.stderr:
            print(f"  Error output:\n{result.stderr[-1000:]}")
        return False


def test_darcy_fno():
    """Test Darcy flow with FNO2d using MLMC."""
    print("\n" + "="*80)
    print("TEST: Darcy Flow + FNO2d (MLMC)")
    print("="*80)
    
    test_dir = Path(__file__).parent.parent
    config_path = test_dir / "examples/pdes/darcy_flow/configs/darcy_fno_mlmc.yaml"
    return run_mlmc_training_test(config_path)


def test_adr_parabolic():
    """Test ADR with ParabolicCNN using MLMC."""
    print("\n" + "="*80)
    print("TEST: Advection-Diffusion-Reaction + ParabolicCNN (MLMC)")
    print("="*80)
    
    test_dir = Path(__file__).parent.parent
    config_path = test_dir / "examples/pdes/adr/configs/adr_parabolic_mlmc.yaml"
    return run_mlmc_training_test(config_path)


def test_navier_stokes_fno3d():
    """Test Navier-Stokes with FNO3d using MLMC."""
    print("\n" + "="*80)
    print("TEST: Navier-Stokes + FNO3d (MLMC)")
    print("="*80)
    
    test_dir = Path(__file__).parent.parent
    config_path = test_dir / "examples/pdes/navier_stokes/configs/ns_fno3d_mlmc.yaml"
    return run_mlmc_training_test(config_path)


def test_flow_past_cylinder_gnn():
    """Test FlowPastCylinder with MP_PDE using MLMC."""
    print("\n" + "="*80)
    print("TEST: FlowPastCylinder + MP_PDE (GNN, MLMC)")
    print("="*80)
    
    test_dir = Path(__file__).parent.parent
    config_path = test_dir / "examples/pdes/flow_past_cylinder/configs/fpc_mp_pde_mlmc.yaml"
    return run_mlmc_training_test(config_path)


def test_jeb_ginot():
    """Test JEB with GINOT using MLMC."""
    print("\n" + "="*80)
    print("TEST: JEB + GINOT (Point Clouds, MLMC)")
    print("="*80)
    
    test_dir = Path(__file__).parent.parent
    config_path = test_dir / "examples/pdes/jeb/configs/jeb_ginot_mlmc.yaml"
    return run_mlmc_training_test(config_path)


def main():
    """Run all paper experiment tests with MLMC."""
    print("\n" + "="*80)
    print("TESTING ALL PAPER EXPERIMENTS (MLMC)")
    print("="*80)
    print("\nThis runs train_mlmc.py for each config with:")
    print("  - Multi-resolution MLMC training")
    print("  - 2 epochs")
    print(f"  - {FRACTION*100}% of training data")
    print(f"  - {FRACTION*100}% of test data")
    print("  - CPU device")
    
    tests = [
        ("Darcy + FNO", test_darcy_fno),
        ("Navier-Stokes + FNO3d", test_navier_stokes_fno3d),
        ("ADR + ParabolicCNN", test_adr_parabolic),
        ("FlowPastCylinder + GNN", test_flow_past_cylinder_gnn),
        ("JEB + GINOT", test_jeb_ginot),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "PASS" if success else "FAIL", None))
        except Exception as e:
            results.append((name, "FAIL", str(e)))
            print(f"✗ Exception: {e}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for name, status, error in results:
        if status == "PASS":
            print(f"✓ PASS {name:50}")
        else:
            error_msg = f": {error}" if error else ""
            print(f"✗ FAIL {name:50}{error_msg}")
    
    passed = sum(1 for _, status, _ in results if status == "PASS")
    print(f"\nPassed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("\n✓ All paper experiments verified!")
        return 0
    else:
        print(f"\n✗ {len(tests) - passed} experiments failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

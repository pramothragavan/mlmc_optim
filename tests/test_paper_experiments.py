"""Test all dataset/model combinations from the paper using single-resolution baseline training.

This runs train.py with override_res for baseline comparisons.
"""

import sys
import subprocess
from pathlib import Path


def run_baseline_training_test(config_path, override_res, epochs=2):
    """Run single-resolution baseline training script with test overrides."""
    project_root = Path(__file__).parent.parent
    train_script = project_root / "examples/examples_src/train.py"
    
    cmd = [
        sys.executable,
        str(train_script),
        "--config", str(config_path),
        "--override_res", str(override_res),
        "--epochs", str(epochs),
        "--train_subset", "0.1",
        "--test_subset", "0.1",
    ]
    
    print(f"\n  Running: python train.py --config {config_path.name} --override_res {override_res} --epochs {epochs} ...")
    # result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
    result = subprocess.run(cmd,  cwd=str(project_root))

    if result.returncode == 0:
        print("  ✓ Baseline training completed successfully")
        return True
    else:
        print(f"  ✗ Baseline training failed with exit code {result.returncode}")
        if result.stderr:
            print(f"  Error output:\n{result.stderr[-1000:]}")
        return False


def test_darcy_fno():
    """Test Darcy flow with FNO2d at single resolution."""
    print("\n" + "="*80)
    print("TEST: Darcy Flow + FNO2d (Baseline)")
    print("="*80)
    
    test_dir = Path(__file__).parent.parent
    config_path = test_dir / "examples/pdes/darcy_flow/configs/darcy_fno_baseline.yaml"
    return run_baseline_training_test(config_path, override_res=120)


def test_adr_parabolic():
    """Test ADR with ParabolicCNN at single resolution."""
    print("\n" + "="*80)
    print("TEST: Advection-Diffusion-Reaction + ParabolicCNN (Baseline)")
    print("="*80)
    
    test_dir = Path(__file__).parent.parent
    config_path = test_dir / "examples/pdes/adr/configs/adr_parabolic_baseline.yaml"
    return run_baseline_training_test(config_path, override_res=300)


def test_navier_stokes_fno3d():
    """Test Navier-Stokes with FNO3d at single resolution."""
    print("\n" + "="*80)
    print("TEST: Navier-Stokes + FNO3d (Baseline)")
    print("="*80)
    
    test_dir = Path(__file__).parent.parent
    config_path = test_dir / "examples/pdes/navier_stokes/configs/ns_fno3d_baseline.yaml"
    # Test at resolution 64 (finest)
    return run_baseline_training_test(config_path, override_res=64)


def test_flow_past_cylinder_gnn():
    """Test Flow Past Cylinder with MP-PDE (GNN) at single resolution."""
    print("\n" + "="*80)
    print("TEST: FlowPastCylinder + MP_PDE (Baseline)")
    print("="*80)
    
    test_dir = Path(__file__).parent.parent
    config_path = test_dir / "examples/pdes/flow_past_cylinder/configs/fpc_mp_pde_baseline.yaml"
    # Test at resolution level 4
    return run_baseline_training_test(config_path, override_res=4)


def test_jeb_ginot():
    """Test JEB with GINOT (Transformer) at single resolution."""
    print("\n" + "="*80)
    print("TEST: JEB + GINOT (Baseline)")
    print("="*80)
    
    test_dir = Path(__file__).parent.parent
    config_path = test_dir / "examples/pdes/jeb/configs/jeb_ginot_baseline.yaml"
    # Test at resolution level 6
    return run_baseline_training_test(config_path, override_res=6)


def main():
    """Run all paper experiment baseline tests."""
    print("\n" + "="*80)
    print("TESTING ALL PAPER EXPERIMENTS (BASELINE)")
    print("="*80)
    print("\nThis runs train.py for each config with:")
    print("  - Single-resolution training (override_res)")
    print("  - 2 epochs")
    print("  - 1% of training data")
    print("  - 1% of test data")

    tests = [
        ("Darcy + FNO (Baseline)", test_darcy_fno),
        ("ADR + ParabolicCNN (Baseline)", test_adr_parabolic),
        ("Navier-Stokes + FNO3d (Baseline)", test_navier_stokes_fno3d),
        ("FlowPastCylinder + MP_PDE (Baseline)", test_flow_past_cylinder_gnn),
        ("JEB + GINOT (Baseline)", test_jeb_ginot),
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
        print("\n✓ All baseline experiments verified!")
        return 0
    else:
        print(f"\n✗ {len(tests) - passed} experiments failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

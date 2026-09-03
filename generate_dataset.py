"""
generate_dataset.py — Master Unified Phase 2 Dataset Generator Entry Point
Supports both Reference Benchmark and Heavy Stress Validation Dataset generation with --mode, --pairs, --seed, and --output.

Usage:
    # Generate 220-Pair Reference Benchmark
    python generate_dataset.py --mode reference --output submission_dataset/phase2_reference_220pairs --pairs 220 --seed 42

    # Generate 220-Pair Heavy Stress Suite
    python generate_dataset.py --mode stress --output submission_dataset/phase2_stress_220pairs --pairs 220 --seed 42

    # Generate 20-Pair Reference Sample Set
    python generate_dataset.py --mode reference --output submission_dataset/phase2_reference_20pairs --pairs 20 --seed 42

    # Generate 20-Pair Stress Sample Set
    python generate_dataset.py --mode stress --output submission_dataset/phase2_stress_20pairs --pairs 20 --seed 42
"""

import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Master Unified Phase 2 Dataset Generator")
    parser.add_argument("--mode", choices=["reference", "stress"], default="reference", help="Dataset generation mode (reference or stress)")
    parser.add_argument("--output", "--output-dir", "--out_dir", type=str, default="submission_dataset/phase2_reference_220pairs", help="Output directory path")
    parser.add_argument("--pairs", type=int, default=220, help="Total number of dataset pairs to generate (e.g. 220 or 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for provably reproducible dataset generation")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.mode == "reference":
        ref_script = os.path.join(base_dir, "test_generate_dataset_p2.py")
        if not os.path.exists(ref_script):
            ref_script = os.path.join(base_dir, "generate_phase2_reference.py")
        if not os.path.exists(ref_script):
            ref_script = os.path.join(base_dir, "phase_2_reference_generator", "generator", "generate_phase2_samples.py")
        
        print(f"=========================================================================================")
        print(f" GENERATING PHASE 2 REFERENCE DATASET ({args.pairs} PAIRS, SEED={args.seed})")
        print(f" Output Directory: {args.output}")
        print(f"=========================================================================================")

        cmd = f'python "{ref_script}" --output-dir "{args.output}" --pairs {args.pairs} --seed {args.seed}'
        subprocess.run(cmd, shell=True, cwd=base_dir)

    elif args.mode == "stress":
        stress_script = os.path.join(base_dir, "generate_phase2_dataset.py")
        if not os.path.exists(stress_script):
            stress_script = os.path.join(base_dir, "generate_phase2_stress.py")

        print(f"=========================================================================================")
        print(f" GENERATING PHASE 2 HEAVY STRESS DATASET ({args.pairs} PAIRS, SEED={args.seed})")
        print(f" Output Directory: {args.output}")
        print(f"=========================================================================================")

        cmd = f'python "{stress_script}" --output "{args.output}" --pairs {args.pairs} --seed {args.seed}'
        subprocess.run(cmd, shell=True, cwd=base_dir)

if __name__ == "__main__":
    main()

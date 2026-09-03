"""
generate_phase2_stress.py — CLI Wrapper for Phase 2 Heavy Stress Dataset Generation
"""
import sys
import argparse
from generate_phase2_dataset import generate_phase2_dataset, PHASE2_SEED

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Heavy Stress Phase 2 Dataset")
    parser.add_argument("--output", "--output-dir", "--out_dir", type=str, default="submission_dataset/phase2_stress_220pairs", help="Output directory")
    parser.add_argument("--pairs", type=int, default=220, help="Total pairs to generate")
    parser.add_argument("--cores", type=int, default=8, help="CPU cores for parallel rendering")
    parser.add_argument("--seed", type=int, default=PHASE2_SEED, help="Random seed")
    args = parser.parse_args()

    out_dir = args.output or args.output_dir or args.out_dir
    generate_phase2_dataset(out_dir=out_dir, total_pairs=args.pairs, num_workers=args.cores)

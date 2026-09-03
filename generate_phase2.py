"""
generate_phase2.py — Official Section 7 Entry Point for Phase 2 Dataset Generation
Supports CLI flags: --output-dir / --out_dir, --seed, --pairs, --rgb
"""

import argparse
import sys
import os

from generate_phase2_dataset import generate_phase2_dataset

def main():
    parser = argparse.ArgumentParser(description="Generate Phase 2 Synthetic SEM Metrology Dataset")
    parser.add_argument("--output-dir", "--out_dir", type=str, default="phase2_dataset", help="Output directory for generated dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic generation")
    parser.add_argument("--pairs", type=int, default=20, help="Number of pairs to generate (default 20 for core set)")
    parser.add_argument("--rgb", action="store_true", help="Include Set D 3-channel RGB Optical Bonus set")
    parser.add_argument("--cores", "--num-workers", type=int, default=1, help="Optional CPU worker cores for parallel multiprocessing (default 1 for evaluator safety)")

    args = parser.parse_args()

    print(f"Generating Phase 2 Dataset: output_dir={args.output_dir}, seed={args.seed}, pairs={args.pairs}, include_rgb={args.rgb}, cores={args.cores}")
    generate_phase2_dataset(out_dir=args.output_dir, total_pairs=args.pairs, include_rgb=args.rgb, num_workers=args.cores)

if __name__ == "__main__":
    main()

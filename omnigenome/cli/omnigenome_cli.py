# -*- coding: utf-8 -*-
# file: omnigenome_cli.py
# time: 12:51 05/02/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.
import argparse
import json


def main():
    """
    Command-line interface (CLI) entry point for OmniGenome.

    Supports sub-commands; currently implements 'rna_design' to design RNA sequences
    matching a given secondary structure using a genetic algorithm.

    Command 'rna_design' arguments:
        --structure (str, required): Target RNA secondary structure in dot-bracket notation.
        --model (str): Pre-trained model path or HuggingFace model ID (default: "yangheng/OmniGenome-186M").
        --mutation-ratio (float): Mutation ratio in genetic algorithm, between 0 and 1 (default: 0.5).
        --num-population (int): Number of individuals in the population (default: 100).
        --num-generation (int): Number of generations for evolution (default: 100).
        --output-file (str): Optional JSON file path to save design results.

    Validates input parameters, runs RNA design via OmniGenomeModelForRNADesign,
    prints the best sequences found, and optionally saves results to JSON.

    Raises:
        ValueError: If mutation ratio is out of range or population/generation counts are non-positive.
    """
    parser = argparse.ArgumentParser(description="OmniGenome CLI")
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Sub-command help"
    )

    # Design command
    design_parser = subparsers.add_parser(
        "rna_design", help="Design RNA sequences for a given secondary structure"
    )
    design_parser.add_argument(
        "--structure",
        type=str,
        required=True,
        help='Target RNA structure in dot-bracket notation (e.g., "(((...)))")',
    )
    design_parser.add_argument(
        "--model",
        type=str,
        default="yangheng/OmniGenome-186M",
        help="Path to the pre-trained model (default: yangheng/OmniGenome-186M)",
    )
    design_parser.add_argument(
        "--mutation-ratio",
        type=float,
        default=0.5,
        help="Mutation ratio for genetic algorithm (0.0-1.0, default: 0.5)",
    )
    design_parser.add_argument(
        "--num-population",
        type=int,
        default=100,
        help="Number of individuals in population (default: 100)",
    )
    design_parser.add_argument(
        "--num-generation",
        type=int,
        default=100,
        help="Number of generations to evolve (default: 100)",
    )
    design_parser.add_argument(
        "--output-file", type=str, help="Output JSON file to save results"
    )

    args = parser.parse_args()

    if args.command == "rna_design":
        from omnigenome import OmniGenomeModelForRNADesign

        # Validate parameters
        if not 0 <= args.mutation_ratio <= 1:
            raise ValueError("--mutation-ratio must be between 0.0 and 1.0")
        if args.num_population <= 0 or args.num_generation <= 0:
            raise ValueError(
                "Population and generation numbers must be positive integers"
            )

        # Run RNA design
        model = OmniGenomeModelForRNADesign(model=args.model)
        best_sequences = model.design(
            structure=args.structure,
            mutation_ratio=args.mutation_ratio,
            num_population=args.num_population,
            num_generation=args.num_generation,
        )

        # Output results
        print(f"Best RNA sequences for {args.structure}:")
        for seq in best_sequences:
            print(f"- {seq}")

        # Save to file if specified
        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(
                    {
                        "structure": args.structure,
                        "parameters": {
                            "mutation_ratio": args.mutation_ratio,
                            "population": args.num_population,
                            "generations": args.num_generation,
                        },
                        "best_sequences": best_sequences,
                    },
                    f,
                    indent=2,
                )
            print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# coding=utf-8
# Copyright 2025 The Lightblue Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convenience script to run evaluation on a model served via OpenAI-compatible API.
This script handles setting environment variables, generating responses, and running evaluations.
"""

import argparse
import os
import subprocess
import sys
from glob import glob


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation for a model on an OpenAI-compatible server"
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name to evaluate")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port where the server is running (default: 8000)")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Base URL for the API (default: http://localhost:{port}/v1)")
    parser.add_argument("--api_key", type=str, default="EMPTY",
                        help="API key for the server (default: EMPTY)")
    parser.add_argument("--languages", type=str, default="ja,en,es,fr",
                        help="Comma-separated list of language codes to evaluate (default: ja,en,es,fr)")
    parser.add_argument("--output_dir", type=str, default="./evaluation/",
                        help="Output directory for evaluation results (default: ./evaluation/)")
    parser.add_argument("--skip_generate", action="store_true",
                        help="Skip response generation (useful if responses already exist)")

    args = parser.parse_args()

    # Build base URL if not provided
    if args.base_url is None:
        args.base_url = f"http://localhost:{args.port}/v1"

    # Set environment variables
    os.environ["OPENAI_COMPATIBLE_BASE_URL"] = args.base_url
    os.environ["OPENAI_COMPATIBLE_API_KEY"] = args.api_key

    print("=" * 80)
    print(f"Running evaluation for model: {args.model_name}")
    print(f"API Endpoint: {args.base_url}")
    print(f"Languages: {args.languages}")
    print("=" * 80)
    print()

    # Step 1: Generate responses (unless skipped)
    if not args.skip_generate:
        print("[1/2] Generating model responses...")
        print("-" * 80)

        generate_cmd = [
            sys.executable, "-m", "get_responses",
            "--model_name", args.model_name,
            "--provider", "openai_compatible",
            "--languages", args.languages
        ]

        result = subprocess.run(generate_cmd, check=False)

        if result.returncode != 0:
            print("\n❌ Error generating responses. Exiting.")
            sys.exit(1)

        print("\n✓ Response generation complete")
        print()
    else:
        print("[1/2] Skipping response generation (--skip_generate)")
        print()

    # Step 2: Run evaluations for each language
    print("[2/2] Running evaluations...")
    print("-" * 80)

    # Get safe model name (replace / with __)
    safe_model_name = args.model_name.replace("/", "__")

    # Parse languages
    lang_codes = [lang.strip() for lang in args.languages.split(",")]

    for lang in lang_codes:
        input_data = f"./data/{lang}_input_data.jsonl"
        response_data = f"./data/{lang}_input_response_data_{safe_model_name}.jsonl"

        # Check if files exist
        if not os.path.exists(input_data):
            print(f"⚠️  Skipping {lang}: input file not found ({input_data})")
            continue

        if not os.path.exists(response_data):
            print(f"⚠️  Skipping {lang}: response file not found ({response_data})")
            continue

        print(f"\nEvaluating {lang.upper()}...")

        eval_cmd = [
            sys.executable, "-m", "evaluation_main",
            f"--input_data={input_data}",
            f"--input_response_data={response_data}",
            f"--output_dir={args.output_dir}"
        ]

        result = subprocess.run(eval_cmd, check=False)

        if result.returncode != 0:
            print(f"❌ Error evaluating {lang}")
        else:
            print(f"✓ {lang.upper()} evaluation complete")

    print()
    print("=" * 80)
    print("All evaluations complete!")
    print(f"Results saved to:")
    print(f"  - {args.output_dir} (evaluation results)")
    print(f"  - ./scores/ (aggregated scores)")
    print("=" * 80)


if __name__ == "__main__":
    main()

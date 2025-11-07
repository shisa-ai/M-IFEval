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
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_env_from_dotenv(dotenv_path: Path = Path(".env")) -> None:
    """Load environment variables from a .env file if present and not already set."""
    if not dotenv_path.exists():
        return

    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config if config else {}


def merge_config_and_args(config: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    """Merge config values with command-line arguments. CLI args take precedence."""
    # Model configuration
    model_config = config.get("model", {})

    if args.model_name is None:
        args.model_name = model_config.get("name")
    if args.provider is None:
        args.provider = model_config.get("provider", "openai_compatible")

    # API configuration - check both top-level and nested client configuration
    client_config = model_config.get("client", {})

    if args.base_url is None:
        # Try nested client.base_url first, then top-level base_url
        args.base_url = client_config.get("base_url") or model_config.get("base_url")
    if args.port is None:
        # Try nested client.port first, then top-level port
        args.port = client_config.get("port") or model_config.get("port")
    if args.api_key is None:
        # Try nested client.api_key first, then top-level api_key
        args.api_key = client_config.get("api_key") or model_config.get("api_key")

    # Evaluation configuration
    if args.languages is None:
        languages = config.get("languages")
        if languages:
            args.languages = ",".join(languages) if isinstance(languages, list) else languages
    if args.output_dir is None:
        args.output_dir = config.get("output_dir", "./evaluation/")

    return args


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation for a model on an OpenAI-compatible server"
    )
    # Config file
    parser.add_argument("--config", type=Path,
                        help="Path to YAML config file. Command-line arguments override config values.")

    # Model configuration
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name to evaluate (required if not in config)")
    parser.add_argument("--provider", type=str, default=None,
                        help="Provider type: openai, anthropic, vllm, openai_compatible (default: openai_compatible)")

    # API configuration
    parser.add_argument("--port", type=int, default=None,
                        help="Port where the server is running (default: 8000, or from config/.env)")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Base URL for the API (default: http://localhost:{port}/v1, or from config/.env)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key for the server (default: EMPTY, or from config/.env)")

    # Evaluation configuration
    parser.add_argument("--languages", type=str, default=None,
                        help="Comma-separated list of language codes to evaluate (default: ja,en,es,fr, or from config)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for evaluation results (default: ./evaluation/, or from config)")

    # Options
    parser.add_argument("--skip_generate", action="store_true",
                        help="Skip response generation (useful if responses already exist)")
    parser.add_argument("--env_file", type=Path, default=Path(".env"),
                        help="Path to .env file (default: .env)")

    args = parser.parse_args()

    # Load environment variables from .env file (if it exists)
    load_env_from_dotenv(args.env_file)

    # Load config file if provided and merge with args
    if args.config:
        try:
            config = load_config(args.config)
            args = merge_config_and_args(config, args)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            return 1

    # Apply defaults when no config is provided
    if args.languages is None:
        args.languages = "ja,en,es,fr"
    if args.output_dir is None:
        args.output_dir = "./evaluation/"
    if args.provider is None:
        args.provider = "openai_compatible"

    # Validate required arguments
    if args.model_name is None:
        print("Error: --model_name is required (either via CLI or config file)")
        return 1

    # Determine base_url (priority: CLI arg > env var > default with port)
    if args.base_url is None:
        if "OPENAI_COMPATIBLE_BASE_URL" in os.environ:
            args.base_url = os.environ["OPENAI_COMPATIBLE_BASE_URL"]
        else:
            port = args.port if args.port is not None else int(os.environ.get("OPENAI_COMPATIBLE_PORT", "8000"))
            args.base_url = f"http://localhost:{port}/v1"

    # Determine API key (priority: CLI arg > env var > default)
    if args.api_key is None:
        args.api_key = os.environ.get("OPENAI_COMPATIBLE_API_KEY", "EMPTY")

    # Set environment variables for child processes
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
            "--provider", args.provider,
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

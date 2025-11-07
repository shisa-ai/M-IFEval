# IMPORTANT NOTE
Activate the mamba M-IFEval to use this project.


# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

M-IFEval is a multilingual instruction-following evaluation benchmark for large language models. It extends the original Google IFEval benchmark to support French, Japanese, and Spanish, in addition to English. The codebase evaluates how well LLMs follow specific instructions across different languages.

## Architecture

### Core Components

1. **Instruction System** - Language-specific instruction checkers organized by type:
   - `instructions_registry.py`: Central registry mapping instruction IDs to checker classes for each language
   - `instructions/{lang}_instructions.py`: Language-specific instruction implementations (en, ja, fr, es)
   - `instruction_utils/{lang}_instructions_util.py`: Helper utilities for language-specific checking logic
   - Instruction types: keywords, language, length_constraints, detectable_content, detectable_format, combination, startend, change_case, punctuation, letters, special_character

2. **Response Generation** (`get_responses.py`):
   - Abstract `ResponseGenerator` base class
   - Provider-specific implementations: `AnthropicResponseGenerator`, `OpenaiResponseGenerator`, `VllmResponseGenerator`, `OpenaiCompatibleResponseGenerator`
   - `SUPPORTED_MODELS` dict maps model names to their provider type
   - Supports parallel inference for OpenAI-compatible API mode (15 workers)

3. **Evaluation Pipeline** (`evaluation_main.py`):
   - Two evaluation modes: strict and loose (tries variations like removing first/last line, removing asterisks)
   - Reads prompts from `{lang}_input_data.jsonl` and responses from `response_data_{model_name}.jsonl`
   - Outputs results to `eval_results_strict.jsonl` and `eval_results_loose.jsonl`
   - Generates scores in `scores/{model_name}_scores.json` and `scores/{model_name}_answers.jsonl`

### Data Flow

1. Input prompts → `data/{lang}_input_data.jsonl` (contains: key, instruction_id_list, prompt, kwargs)
2. Model responses → `data/{lang}_input_response_data_{model_name}.jsonl` (contains: prompt, response)
3. Evaluation → checks each instruction using corresponding checker class
4. Results → JSON files with prompt-level and instruction-level accuracy

## Common Commands

### Setup

```bash
# Install dependencies
pip3 install --user -r requirements.txt

# Download required Spacy models
python -m spacy download es_core_news_sm --quiet
python -m spacy download xx_sent_ud_sm --quiet
```

### Generate Model Responses

```bash
# For a single language
python3 -m get_responses.py --model_name {model_name} --languages ja

# For all languages
python3 -m get_responses.py --model_name {model_name}

# Supported languages: en, ja, es, fr (comma-separated)
```

**Environment variables required:**
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- HuggingFace: `HUGGINGFACE_TOKEN` (login via `huggingface-cli login`)
- OpenAI-Compatible APIs: `OPENAI_COMPATIBLE_BASE_URL`, `OPENAI_COMPATIBLE_API_KEY` (defaults to localhost:8000, also supports legacy `VLLM_BASE_URL` and `VLLM_API_KEY`)
- Optional: `MAX_MODEL_LEN` for vLLM (default: 4096)

### Run Evaluations

#### Quick Evaluation (Convenience Script)

For OpenAI-compatible API endpoints, use the convenience script:

```bash
# Run end-to-end evaluation (generate + eval)
python3 shisa_run_eval.py \
  --model_name your-model-name \
  --port 8000 \
  --languages ja,en,es,fr

# With custom base URL and API key
python3 shisa_run_eval.py \
  --model_name your-model-name \
  --base_url http://localhost:8001/v1 \
  --api_key your-api-key \
  --languages ja

# Skip generation if responses already exist
python3 shisa_run_eval.py \
  --model_name your-model-name \
  --skip_generate
```

#### Manual Evaluation

```bash
# Evaluate a model's responses for a specific language
python3 -m evaluation_main \
  --input_data=./data/{lang}_input_data.jsonl \
  --input_response_data=./data/{lang}_input_response_data_{model_name}.jsonl \
  --output_dir=./evaluation/

# Example for Japanese with a specific model
python3 -m evaluation_main \
  --input_data=./data/ja_input_data.jsonl \
  --input_response_data=./data/ja_input_response_data_claude-3-5-sonnet-20240620.jsonl \
  --output_dir=./evaluation/
```

## Key Implementation Details

### Adding New Models

**Option 1: Dynamic (No code changes needed)**

Use the `--provider` flag when generating responses:
```bash
python3 -m get_responses --model_name your-model --provider openai_compatible
```

Or use the convenience script:
```bash
python3 shisa_run_eval.py --model_name your-model --port 8000
```

**Option 2: Static (Permanent addition)**

Add model to `SUPPORTED_MODELS` dict in `get_responses.py`:
```python
SUPPORTED_MODELS = {
    "your-model-name": "openai",  # or "anthropic", "vllm", "openai_compatible"
}
```

**Provider types:**
- `openai`: Official OpenAI API (gpt-4o, o1, etc.)
- `anthropic`: Anthropic API (Claude models)
- `vllm`: Local vLLM instance (loads model into memory)
- `openai_compatible`: Any OpenAI-compatible API endpoint (vLLM server, Together AI, Anyscale, etc.)

**Adding new providers:** Implement a `ResponseGenerator` subclass with `get_response(input_texts)` method

### Model Name Handling

- Model names with `/` are converted to `__` for safe filenames (e.g., `Qwen/Qwen2.5-7B` → `Qwen__Qwen2.5-7B`)
- This conversion happens in `get_safe_model_name()` in evaluation_main.py:202

### Language-Specific Instructions

Each language has unique instructions beyond the common set:
- **Japanese**: Furigana, Kanji limits, Hiragana/Katakana only, sentence endings, Kansuuji (漢数字)
- **French**: Accent handling, informal address (tutoiement), no digits, quotation marks (« »)
- **Spanish**: Letter frequency (ñ, ü), tildes, inverted punctuation marks

See `instructions/README.md` for complete details.

### Instruction Registry Pattern

The registry uses prefixed keys: `"{lang}:{category}:{instruction_type}"` (e.g., `"ja:letters:furigana"`)
- Instruction classes inherit from base instruction types
- Each checker implements `build_description(**kwargs)` and `check_following(response)`

### Output Structure

Evaluation generates:
- `eval_results_strict.jsonl` / `eval_results_loose.jsonl`: Full results with all fields
- `scores/{model_name}_scores.json`: Combined strict/loose scores with tier0/tier1 breakdowns
- `scores/{model_name}_answers.jsonl`: Answer data without aggregated scores

Scores include:
- `prompt_level_accuracy`: Percentage of prompts where all instructions were followed
- `instruction_level_accuracy`: Percentage of individual instructions followed
- `tier0_scores`: Accuracy by instruction category (e.g., "keywords:existence")
- `tier1_scores`: Accuracy by full instruction ID

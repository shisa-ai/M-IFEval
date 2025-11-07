# M-IFEval Configuration Files

This directory contains YAML configuration files for running M-IFEval evaluations with different models.

## Configuration Structure

### Simple Format (Recommended for basic use)

```yaml
# Model configuration
model:
  name: model-identifier          # Model name/path
  provider: openai_compatible     # Provider type: openai, anthropic, vllm, openai_compatible
  base_url: http://localhost:9000/v1  # API endpoint (for API-based providers)
  # OR use port shorthand for localhost:
  # port: 9000
  api_key: EMPTY                  # Optional: API key (or reference env var)

# Languages to evaluate (list or comma-separated string)
languages:
  - ja
  - en
  - es
  - fr

# Output directory for evaluation results
output_dir: ./evaluation/
```

### Detailed Format (Matches translation-improvement repo structure)

```yaml
# Output configuration
output_dir: ./evaluation/

# Parallel processing (reserved for future use)
workers: 15

# Model configuration
model:
  name: model-identifier
  provider: openai_compatible

  client:
    base_url: http://localhost:9000/v1
    api_key_env: OPENAI_API_KEY  # Name of env var containing API key
    request_timeout: 60          # Reserved for future use

  generation:
    temperature: 0.0             # Sampling temperature (0.0-2.0)
    top_p: 1.0                  # Nucleus sampling threshold
    reasoning_effort: null      # For o1/Gemini: low, medium, high
    max_tokens: 2048            # Maximum response length

  metadata:
    description: Model description for documentation

# Languages to evaluate
languages:
  - ja
  - en
  - es
  - fr
```

**Note:** Both formats are supported. The detailed format is useful for:
- Consistency with other projects (matches translation-improvement repo structure)
- Full control over generation parameters (temperature, top_p, reasoning_effort, max_tokens)
- Better documentation with metadata fields

## Available Configs

### Local Models (OpenAI-Compatible API)

- **`shisa-unphi.yaml`** - Shisa v2 unphi4 14B on port 9000 (detailed format, matches translation-improvement repo)
- **`local-vllm-ja-only.yaml`** - Generic local model template, Japanese only (simple format)

### API-Based Models

- **`claude-sonnet.yaml`** - Claude 3.5 Sonnet via Anthropic API
- **`gpt4o.yaml`** - GPT-4o via OpenAI API
- **`gemini-2.5-flash.yaml`** - Gemini 2.5 Flash via Google AI API (uses `api_key_env: GEMINI_API_KEY`)

## Usage

### Basic Usage

```bash
# Run evaluation with a config file
python3 shisa_run_eval.py --config configs/shisa-unphi.yaml
```

### Override Config Values

CLI arguments take precedence over config file values:

```bash
# Override languages
python3 shisa_run_eval.py \
  --config configs/shisa-unphi.yaml \
  --languages ja

# Override port
python3 shisa_run_eval.py \
  --config configs/shisa-unphi.yaml \
  --port 8001

# Skip response generation (evaluate existing responses)
python3 shisa_run_eval.py \
  --config configs/shisa-unphi.yaml \
  --skip_generate
```

## Creating New Configs

To evaluate a new model:

1. Copy an existing config file as a template
2. Update the `model` section with your model details
3. Adjust `languages` and `output_dir` as needed
4. Run with `--config configs/your-new-config.yaml`

### Example: Local Model Config

```yaml
model:
  name: your-org/your-model-name
  provider: openai_compatible
  port: 8000  # Assumes http://localhost:8000/v1

languages: [ja, en, es, fr]
output_dir: ./evaluation/
```

### Example: API Model Config

```yaml
model:
  name: gpt-4o-mini
  provider: openai
  # base_url not needed for official OpenAI API
  # API key comes from OPENAI_API_KEY env var

languages: [ja, en, es, fr]
output_dir: ./evaluation/
```

## Priority Order

Configuration values are resolved in this priority order:

1. **Command-line arguments** (highest priority)
2. **Config file values**
3. **Environment variables** (from `.env` file)
4. **Defaults** (lowest priority)

## Environment Variables

Config files can reference environment variables for sensitive data. Use the `api_key_env` field to specify which environment variable contains your API key:

```yaml
model:
  client:
    api_key_env: GEMINI_API_KEY  # Will read API key from $GEMINI_API_KEY
```

**Common environment variables:**
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `GEMINI_API_KEY` - Google Gemini API key
- `OPENAI_COMPATIBLE_BASE_URL` - Custom base URL
- `OPENAI_COMPATIBLE_API_KEY` - Custom API key (fallback when `api_key_env` not specified)

Set these in a `.env` file (see `.env.example` in the project root) or export them in your shell.

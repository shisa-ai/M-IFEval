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

import os
import argparse
from glob import glob
from tqdm.auto import tqdm
from datasets import load_dataset

class ResponseGenerator:
    def __init__(self, model_name):
        raise NotImplementedError
    
    def get_response(self, input_texts):
        raise NotImplementedError

######## Anthropic ########

class AnthropicResponseGenerator(ResponseGenerator):

    def __init__(self, model_name):
        import anthropic
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
        self.model_name = model_name
    
    def get_response(self, input_texts):
        return [
            self.anthropic_client.messages.create(
                model=self.model_name,
                max_tokens=2048,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": input_text
                            }
                        ]
                    }
                ]
            ).content[0].text for input_text in tqdm(input_texts)
        ]

######## OpenAI ########

class OpenaiResponseGenerator(ResponseGenerator):
    def __init__(self, model_name):
        from openai import OpenAI

        self.openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_name = model_name
    
    def get_single_response(self, input_text):
        try:
            return self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": input_text
                        }
                    ]
                    }
                ],
                # temperature=0,
                # # max_tokens=None if "o1" in self.model_name else 2048,
                # # top_p=1,
                # frequency_penalty=0,
                # presence_penalty=0,
                # response_format={"type": "text"}
            ).choices[0].message.content
        except Exception as e:
            print(e)
            return None
    
    def get_response(self, input_texts):
        return [
            self.get_single_response(input_text) for input_text in tqdm(input_texts)
        ]

######## OpenAI-Compatible API (vLLM, Together AI, etc.) ########

class OpenaiCompatibleResponseGenerator(ResponseGenerator):
    def __init__(self, model_name, base_url="http://localhost:8000/v1", api_key="EMPTY",
                 temperature=0.0, top_p=1.0, reasoning_effort=None, max_tokens=2048):
        from openai import OpenAI

        # Use passed parameters, only fall back to env vars if not provided
        # Check OPENAI_COMPATIBLE_* env vars first, then legacy VLLM_* vars
        if base_url == "http://localhost:8000/v1":  # Default value, check env vars
            self.base_url = os.environ.get("OPENAI_COMPATIBLE_BASE_URL",
                                          os.environ.get("VLLM_BASE_URL", base_url))
        else:
            self.base_url = base_url

        if api_key == "EMPTY":  # Default value, check env vars
            self.api_key = os.environ.get("OPENAI_COMPATIBLE_API_KEY",
                                         os.environ.get("VLLM_API_KEY", api_key))
        else:
            self.api_key = api_key

        print(f"DEBUG in OpenaiCompatibleResponseGenerator.__init__:")
        print(f"  Passed base_url parameter: {base_url}")
        print(f"  Passed api_key parameter: {api_key[:10]}..." if len(api_key) > 10 else f"  Passed api_key parameter: {api_key}")
        print(f"  Final self.base_url: {self.base_url}")
        print(f"  Final self.api_key: {self.api_key[:10]}..." if len(self.api_key) > 10 else f"  Final self.api_key: {self.api_key}")

        self.openai_client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        self.model_name = model_name

        # Generation parameters
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens

    def get_single_response(self, input_text):
        try:
            # Build kwargs for API call
            kwargs = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": input_text
                            }
                        ]
                    }
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
            }

            # Only add reasoning_effort if specified (some models don't support it)
            if self.reasoning_effort is not None:
                kwargs["reasoning_effort"] = self.reasoning_effort

            return self.openai_client.chat.completions.create(**kwargs).choices[0].message.content
        except Exception as e:
            print(e)
            return None

    def get_response(self, input_texts):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        responses = [None] * len(input_texts)
        with ThreadPoolExecutor(max_workers=15) as executor:
            # Submit all tasks and map them to their indices
            future_to_idx = {
                executor.submit(self.get_single_response, text): idx
                for idx, text in enumerate(input_texts)
            }

            # Process completed futures with progress bar
            for future in tqdm(as_completed(future_to_idx), total=len(input_texts)):
                idx = future_to_idx[future]
                responses[idx] = future.result()

        return responses

######## VertexAI ########

# TO DO: Add Support for VertexAI
# class VertexResponseGenerator(ResponseGenerator):
#     def __init__(self, model_name):
#         self.model_name = model_name
    
#     def get_response(self, input_texts):
#         import vertexai
#         from vertexai.generative_models import GenerativeModel

#         generation_config = {
#             "max_output_tokens": 2048,
#             "temperature": 0,
#         }

#         safety_settings = [
#         ]

#         vertexai.init(project="dev-llab", location="asia-south1")
#         model = GenerativeModel(
#             self.model_name,
#         )

#         def get_vertex_response(input_text):
#             chat = model.start_chat(response_validation=False)

#             return chat.send_message(
#                 [input_text],
#                 generation_config=generation_config,
#                 safety_settings=safety_settings
#             ).candidates[0].content.parts[0].text

#         return [get_vertex_response(input_text) for input_text in tqdm(input_texts)]
        


######## vLLM ########

class VllmResponseGenerator(ResponseGenerator):
    def __init__(self, model_name):
        from vllm import LLM, SamplingParams
        self.model_name = model_name
        self.llm = LLM(model=self.model_name, max_model_len=os.environ.get("MAX_MODEL_LEN", 4096))
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

    def get_response(self, input_texts):
        input_conversations = [[{
            "role": "user",
            "content": input_text
        }] for input_text in input_texts]

        outputs = self.llm.chat(input_conversations,
                   sampling_params=self.sampling_params,
                   use_tqdm=True)
        return [output.outputs[0].text for output in outputs]

######## Main ########

SUPPORTED_MODELS = {
    'gpt-4o-mini-2024-07-18': 'openai',
    'gpt-4o-2024-08-06': 'openai',
    'o1-preview-2024-09-12': 'openai',
    'o1-mini-2024-09-12': 'openai',
    'claude-3-haiku-20240307': 'anthropic',
    'claude-3-5-sonnet-20240620': 'anthropic',
    'claude-3-opus-20240229': 'anthropic',
    # 'gemini-1.5-pro-002': 'gemini',
    # 'gemini-1.5-flash-002': 'gemini',
    'CohereForAI/c4ai-command-r-plus-4bit': 'vllm',
    'CohereForAI/c4ai-command-r-v01-4bit': 'vllm',
    'CohereForAI/aya-23-8B': 'vllm',
    'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4': 'vllm',
    'Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4': 'vllm',
    'Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4': 'vllm',
    'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4': 'vllm',
    'Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4': 'vllm',
    'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4': 'vllm',
    'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4': 'vllm',
    'hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4': 'vllm',
    'hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4': 'vllm',
    'mistralai/Mistral-7B-Instruct-v0.3': 'vllm',
    'deepseek-ai/deepseek-llm-7b-chat': 'vllm',
    'shisa-ai/chotto-14b-20250922-W8A8-INT8-smooth': 'openai_compatible',
    'Unbabel/Tower-Plus-9B': 'openai_compatible',
    'shisa-ai/shisa-v2-unphi4-14b-denoted-W8A8-INT8': 'openai_compatible'
}

MODEL_CLASS_DICT = {
    "openai": OpenaiResponseGenerator,
    "anthropic": AnthropicResponseGenerator,
    # "gemini": VertexResponseGenerator,
    "vllm": VllmResponseGenerator,
    "openai_compatible": OpenaiCompatibleResponseGenerator,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--provider", type=str, default=None,
                        help="Provider type (openai, anthropic, vllm, openai_compatible). If not specified, will look up in SUPPORTED_MODELS.")
    parser.add_argument("--languages", type=str, default=None,
                        help="Comma-separated list of language codes (e.g., 'ja' or 'ja,en,es,fr'). If not specified, all languages will be processed.")

    # API configuration (for openai_compatible provider)
    parser.add_argument("--base_url", type=str, default=None,
                        help="Base URL for API endpoint")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key directly (overrides --api_key_env)")
    parser.add_argument("--api_key_env", type=str, default=None,
                        help="Environment variable name containing API key (e.g., GEMINI_API_KEY)")

    # Generation parameters (for openai_compatible provider)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for sampling (default: 0.0)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p for nucleus sampling (default: 1.0)")
    parser.add_argument("--reasoning_effort", type=str, default=None,
                        choices=["low", "medium", "high"],
                        help="Reasoning effort for models that support it (o1, Gemini thinking)")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Maximum tokens in response (default: 2048)")

    args = parser.parse_args()

    model_name = args.model_name

    # Determine provider type
    if args.provider:
        provider_type = args.provider
        assert provider_type in MODEL_CLASS_DICT, f"Provider {provider_type} not supported. Available providers: {list(MODEL_CLASS_DICT.keys())}"
    else:
        assert model_name in SUPPORTED_MODELS, f"Model {model_name} not supported, update SUPPORTED_MODELS dictionary in get_responses.py to support it, or use --provider flag."
        provider_type = SUPPORTED_MODELS[model_name]

    # Get all input data files
    paths = sorted(glob("./data/*_input_data.jsonl"))

    # Filter by language if specified
    if args.languages:
        lang_codes = [lang.strip() for lang in args.languages.split(",")]
        paths = [p for p in paths for lang in lang_codes if f"/{lang}_input_data.jsonl" in p]

    model_class = MODEL_CLASS_DICT[provider_type]

    # Create response generator with generation parameters for openai_compatible
    if provider_type == "openai_compatible":
        # Determine API key (priority: --api_key > env var from --api_key_env > default env vars)
        api_key = "EMPTY"
        if args.api_key:
            api_key = args.api_key
            print(f"Using API key from --api_key argument: {api_key[:10]}..." if len(api_key) > 10 else f"Using API key: {api_key}")
        elif args.api_key_env:
            # Read from specified environment variable
            api_key = os.environ.get(args.api_key_env, "EMPTY")
            print(f"Using API key from env var ${args.api_key_env}: {api_key[:10]}..." if len(api_key) > 10 else f"Using API key from ${args.api_key_env}: {api_key}")
        else:
            print("No --api_key or --api_key_env specified, using default env var logic")
        # else: let OpenaiCompatibleResponseGenerator use its default logic

        # Determine base_url
        base_url = args.base_url if args.base_url else "http://localhost:8000/v1"
        print(f"Using base_url: {base_url}")

        response_generator = model_class(
            model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            top_p=args.top_p,
            reasoning_effort=args.reasoning_effort,
            max_tokens=args.max_tokens
        )
    else:
        response_generator = model_class(model_name)

    for path in paths:
        print(path + " - " + model_name)
        ds = load_dataset("json", data_files={"train": path}, split="train")
        ds = ds.add_column("response", response_generator.get_response(ds["prompt"]))
        ds.select_columns(["prompt", "response"]).to_json(
            path[:-10] + "response_data_" + model_name.replace("/", "__") + ".jsonl"
        )
# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Binary of evaluating instruction following. See README.md."""

import collections
import dataclasses
import json
import os
from typing import Dict, Optional, Sequence, Union

from absl import app
from absl import flags
from absl import logging

import instructions_registry


_INPUT_DATA = flags.DEFINE_string(
    "input_data", None, "path to input data", required=True
)

_INPUT_RESPONSE_DATA = flags.DEFINE_string(
    "input_response_data", None, "path to input response data", required=False
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Output directory for inference and eval results.",
    required=True,
)


@dataclasses.dataclass
class InputExample:
  key: int
  instruction_id_list: list[str]
  prompt: str
  kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
  instruction_id_list: list[str]
  prompt: str
  response: str
  follow_all_instructions: bool
  follow_instruction_list: list[bool]


def read_prompt_list(input_jsonl_filename):
  """Read inputs from jsonl."""
  inputs = []
  with open(input_jsonl_filename, "r", encoding='utf-8') as f:
    for l in f:
      example = json.loads(l)
      inputs.append(
          InputExample(key=example["key"],
                       instruction_id_list=example["instruction_id_list"],
                       prompt=example["prompt"],
                       kwargs=example["kwargs"]))
  return inputs


def write_outputs(output_jsonl_filename, outputs):
  """Writes outputs to jsonl."""
  assert outputs
  
  # Ensure the entire directory path exists
  directory = os.path.dirname(output_jsonl_filename)
  if directory and not os.path.exists(directory):
      os.makedirs(directory, exist_ok=True)
  
  with open(output_jsonl_filename, "w") as f:
    for o in outputs:
      f.write(
          json.dumps(
              {
                  attr_name: o.__getattribute__(attr_name)
                  for attr_name in [
                      name for name in dir(o) if not name.startswith("_")
                  ]
              },
              ensure_ascii=False
          )
      )
      f.write("\n")


def test_instruction_following_strict(
    inp,
    prompt_to_response,
):
  """Tests response to see if instrutions are followed."""
  response = prompt_to_response[inp.prompt]
  instruction_list = inp.instruction_id_list
  is_following_list = []

  for index, instruction_id in enumerate(instruction_list):
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)

    instruction.build_description(**inp.kwargs[index])
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=inp.prompt)

    if isinstance(response, str) and response.strip() and instruction.check_following(response):
      is_following_list.append(True)
    else:
      is_following_list.append(False)

  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response=response,
      follow_all_instructions=all(is_following_list),
      follow_instruction_list=is_following_list,
  )


def test_instruction_following_loose(
    inp,
    prompt_to_response,
):
  """Tests response for an upper bound for following instructions."""
  response = prompt_to_response[inp.prompt]
  if isinstance(response, str):
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_quotation = response.replace('\"', '')
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
      response,
      revised_response,
      response_remove_first,
      response_remove_last,
      response_remove_both,
      revised_response_remove_first,
      revised_response_remove_last,
      revised_response_remove_both,
    ]
  else:
    all_responses=[]
    
  instruction_list = inp.instruction_id_list
  is_following_list = []

  for index, instruction_id in enumerate(instruction_list):
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)

    instruction.build_description(**inp.kwargs[index])
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=inp.prompt)

    is_following = False
    for r in all_responses:
      if r.strip() and instruction.check_following(r):
        is_following = True
        break

    is_following_list.append(is_following)

  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response=response,
      follow_all_instructions=all(is_following_list),
      follow_instruction_list=is_following_list,
  )


def read_prompt_to_response_dict(input_jsonl_filename):
  """Creates dictionary matching prompt and response."""
  return_dict = {}
  with open(input_jsonl_filename, "r", encoding='utf-8') as f:
    for l in f:
      example = json.loads(l)
      return_dict[example["prompt"]] = example["response"]
  return return_dict


def get_safe_model_name(input_response_data_path):
  """Extract model name from response data path and make it safe for filenames."""
  # Extract filename from path
  filename = os.path.basename(input_response_data_path)
  # Remove extension and 'response_data_' prefix if present
  model_name = filename.replace('.jsonl', '').replace('response_data_', '')
  # Replace / with __ for safe filenames
  safe_name = model_name.replace('/', '__')
  return safe_name


def write_answers(output_dir, model_name, outputs):
  """Writes answer data (without scores) to jsonl."""
  scores_dir = "scores"
  os.makedirs(scores_dir, exist_ok=True)

  answers_file = os.path.join(scores_dir, f"{model_name}_answers.jsonl")
  with open(answers_file, "w", encoding='utf-8') as f:
    for o in outputs:
      answer_data = {
        "instruction_id_list": o.instruction_id_list,
        "prompt": o.prompt,
        "response": o.response,
        "follow_all_instructions": o.follow_all_instructions,
        "follow_instruction_list": o.follow_instruction_list,
      }
      f.write(json.dumps(answer_data))
      f.write("\n")

  logging.info("Saved answers to: %s", answers_file)
  return answers_file


def calculate_scores(outputs):
  """Calculates accuracy scores and returns them as a dictionary."""
  prompt_total = 0
  prompt_correct = 0
  instruction_total = 0
  instruction_correct = 0

  tier0_total = collections.defaultdict(int)
  tier0_correct = collections.defaultdict(int)

  tier1_total = collections.defaultdict(int)
  tier1_correct = collections.defaultdict(int)

  for example in outputs:
    follow_instruction_list = example.follow_instruction_list
    instruction_id_list = example.instruction_id_list

    prompt_total += 1
    if all(follow_instruction_list):
      prompt_correct += 1

    instruction_total += len(instruction_id_list)
    instruction_correct += sum(follow_instruction_list)

    for instruction_id, followed_or_not in zip(
        instruction_id_list, follow_instruction_list
    ):
      instruction_id = instruction_id.split(":")[0]
      tier0_total[instruction_id] += 1
      if followed_or_not:
        tier0_correct[instruction_id] += 1

    for instruction_id, followed_or_not in zip(
        instruction_id_list, follow_instruction_list
    ):
      tier1_total[instruction_id] += 1
      if followed_or_not:
        tier1_correct[instruction_id] += 1

  # Calculate accuracies
  tier0_scores = {
    instruction_id: tier0_correct[instruction_id] / tier0_total[instruction_id]
    for instruction_id in tier0_total.keys()
  }

  tier1_scores = {
    instruction_id: tier1_correct[instruction_id] / tier1_total[instruction_id]
    for instruction_id in tier1_total.keys()
  }

  return {
    "prompt_level_accuracy": prompt_correct / prompt_total if prompt_total > 0 else 0,
    "instruction_level_accuracy": instruction_correct / instruction_total if instruction_total > 0 else 0,
    "tier0_scores": tier0_scores,
    "tier1_scores": tier1_scores,
    "total_prompts": prompt_total,
    "total_instructions": instruction_total,
  }


def print_report(outputs):
  """Prints a report on accuracy scores."""
  scores = calculate_scores(outputs)

  print(f"prompt-level: {scores['prompt_level_accuracy']}")
  print(f"instruction-level: {scores['instruction_level_accuracy']}")
  print()
  for instruction_id in sorted(scores['tier0_scores'].keys()):
    accuracy = scores['tier0_scores'][instruction_id]
    print(f"{instruction_id} {accuracy}")
  print()
  for instruction_id in sorted(scores['tier1_scores'].keys()):
    accuracy = scores['tier1_scores'][instruction_id]
    print(f"{instruction_id} {accuracy}")

  return scores


def write_scores(output_dir, model_name, strict_scores, loose_scores):
  """Writes combined strict and loose scores to json."""
  scores_dir = "scores"
  os.makedirs(scores_dir, exist_ok=True)

  scores_file = os.path.join(scores_dir, f"{model_name}_scores.json")
  combined_scores = {
    "strict": strict_scores,
    "loose": loose_scores,
  }

  with open(scores_file, "w", encoding='utf-8') as f:
    json.dump(combined_scores, f, indent=2)

  logging.info("Saved scores to: %s", scores_file)
  return scores_file


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  inputs = read_prompt_list(_INPUT_DATA.value)
  prompt_to_response = read_prompt_to_response_dict(
      _INPUT_RESPONSE_DATA.value)

  # Get safe model name for saving scores
  model_name = get_safe_model_name(_INPUT_RESPONSE_DATA.value)

  # Store scores for both strict and loose evaluation
  all_scores = {}

  # get instruction following results
  for func, output_file_name in [
      (test_instruction_following_strict, "eval_results_strict"),
      (test_instruction_following_loose, "eval_results_loose"),
  ]:
    logging.info("Generating %s...", output_file_name)
    outputs = []
    for inp in inputs:
      outputs.append(func(inp, prompt_to_response))
    follow_all_instructions = [o.follow_all_instructions for o in outputs]
    accuracy = sum(follow_all_instructions) / len(outputs)
    logging.info("Accuracy: %f", accuracy)

    # Write old-style outputs (keeping backwards compatibility)
    output_file_name_full = os.path.join(
        _OUTPUT_DIR.value, output_file_name + ".jsonl"
    )
    write_outputs(output_file_name_full, outputs)
    logging.info("Generated: %s", output_file_name_full)

    # Prints instruction following accuracy report and collect scores
    print("=" * 64)
    print(f"{output_file_name_full} Accuracy Scores:")
    scores = print_report(outputs)

    # Store scores for combined output
    eval_type = "strict" if "strict" in output_file_name else "loose"
    all_scores[eval_type] = scores

    # Save answers once (using strict results since responses are identical)
    if eval_type == "strict":
      write_answers(_OUTPUT_DIR.value, model_name, outputs)

  # Write combined scores to single JSON file
  write_scores(_OUTPUT_DIR.value, model_name, all_scores.get("strict"), all_scores.get("loose"))


if __name__ == "__main__":
  app.run(main)

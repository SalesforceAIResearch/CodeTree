# LICENSE HEADER MANAGED BY add-license-header
#
# /*
#  * Copyright (c) 2023, Salesforce, Inc.
#  * SPDX-License-Identifier: Apache-2
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  * you may not use this file except in compliance with the License.
#  * You may obtain a copy of the License at
#  *
#  *     http://www.apache.org/licenses/LICENSE-2.0
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS,
#  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  * See the License for the specific language governing permissions and
#  * limitations under the License.
#  */
#

import os
import gzip
import json
import openai
import jsonlines

from typing import List

openai.api_key = os.getenv("OPENAI_API_KEY")


def make_printv(verbose: bool):
    def print_v(*args, **kwargs):
        if verbose:
            kwargs["flush"] = True
            print(*args, **kwargs)
        else:
            pass
    return print_v

def merge_test_subsets(test_subsets):
    # Initialize the lists for full_tests inputs and outputs
    full_inputs = []
    full_outputs = []

    # Iterate through each subset and extend the full_tests lists
    for subset in ['public_tests', 'private_tests']: #, 'generated_tests']:
        # Check if the subset exists in the test_subsets to avoid KeyError
        if subset in test_subsets:
            full_inputs.extend(test_subsets[subset]['input'])
            full_outputs.extend(test_subsets[subset]['output'])

    # Construct the full_tests dictionary
    full_tests = {
            'input': full_inputs,
            'output': full_outputs
        }
    return full_tests

def read_json(path:str):
    with open(path) as f:
        data = json.load(f)
    for i in range(len(data)):
        data[i]["prompt"] = data[i]["description"].replace("\n\n\n","\n\n").replace("\n\n\n","\n\n")
        data[i]["given_tests"] = data[i]["public_tests"]
        data[i]["test"] =data[i]["private_tests"] # merge_test_subsets(data[i])
        data[i]["prev"] = ""
        if "cf_rating" not in data[i].keys(): data[i]["cf_rating"] = 0
    return data

def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File `{path}` does not exist.")
    elif not path.endswith(".jsonl"):
        raise ValueError(f"File `{path}` is not a jsonl file.")
    items = []
    with jsonlines.open(path) as reader:
        for item in reader:
            items += [item]
    return items


def write_jsonl(path: str, data: List[dict], append: bool = False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with jsonlines.open(path, mode='a' if append else 'w') as writer:
        for item in data:
            writer.write(item)


def read_jsonl_gz(path: str) -> List[dict]:
    if not path.endswith(".jsonl.gz"):
        raise ValueError(f"File `{path}` is not a jsonl.gz file.")
    with gzip.open(path, "rt") as f:
        data = [json.loads(line) for line in f]
    return data


# generator that returns the item and the index in the dataset.
# if the results_path exists, it will skip all items that have been processed
# before.
def enumerate_resume(dataset, results_path):
    if not os.path.exists(results_path):
        for i, item in enumerate(dataset):
            yield i, item
    else:
        count = 0
        with jsonlines.open(results_path) as reader:
            for item in reader:
                count += 1

        for i, item in enumerate(dataset):
            # skip items that have been processed before
            if i < count:
                continue
            yield i, item


def resume_success_count(dataset) -> int:
    count = 0
    for item in dataset:
        if "is_solved" in item and item["is_solved"]:
            count += 1
    return count


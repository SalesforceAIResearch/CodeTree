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

from utils import  make_printv, write_jsonl
from executors import executor_factory
from generators import generator_factory, model_factory
from typing import List, Dict, Tuple, Any
import math
import sys
from collections import Counter
from common import gen_test_eval
sys.set_int_max_str_digits(100000)  # Increase the limit to 10000 digits
from config import get_parsed_args
args = get_parsed_args()
Codecontests = False if args.function else True
ADD_HINT = "\nTo solve the problem, You can follow the hint given by an expert: "


class Node:
    def __init__(self, solution: str, parent=None, context="", depth=0):
        self.solution = solution
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.context = ""
        self.depth = depth
        self.reflection = ""
        self.test_feedback = ""

    def uct(self, exploration_weight=1.0):
        if self.visits == 0:
            return self.value
        return (self.value / self.visits) + exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self):
        if not self.children:  # Check if children list is empty
            return None
        return max(self.children, key=lambda child: child.uct())

    def best_child_value(self):
        if not self.children:  # Check if children list is empty
            return None
        return max(self.children, key=lambda child: child.value)

    def sort_children_by_value(self):
        self.children.sort(key=lambda x: x.value)

    def update(self, reward: float):
        self.visits += 1
        self.value += reward

def strategy_guide(
        dataset: List[dict],
        model_name: str,
        language: str,
        log_path: str,
        verbose: bool,
        max_iters: int,
        Codecontests: bool,
        is_leetcode: bool = False
) -> None:
    if Codecontests:
        exe = executor_factory("code_contests")
    else: exe = executor_factory(language, is_leet=is_leetcode)
    
    gen = generator_factory(language)
    model = model_factory(model_name)
    print_v = make_printv(verbose)

    num_items = len(dataset)
    num_success, skip, weak_success = 0, 0, 0  # Counter for successful solutions
    passed_at_sample, solve_or_not = [], []
    is_weaker_solved = False
    pass_problem_subset = []

    for idx, item in enumerate(dataset):

        tests_i = item["given_tests"]
        if Codecontests:
            item["entry_point"] = ""
        else:
            tests_i = [test for test in tests_i if item['entry_point'] in test and 'assert False' not in test]

        hints = gen.strategy(item["prompt"], model, num_strategy=20)

        root = Node("")
        stack = [root] # implementations
        is_solved = False
        num_try = 0
        if len(hints) > max_iters: hints = hints[:max_iters]
        elif len(hints) < max_iters: hints = hints + gen.strategy(item["prompt"]+"\nThink Carefully.", model, num_strategy=20 - len(hints))
        for hint in hints:
            cur_func_impl = None
            while cur_func_impl is None:
                cur_func_impl = gen.func_impl(item["prompt"] + f"{ADD_HINT}{hint}\n", model, "simple")
                stack.append(Node(cur_func_impl))
                stack[0].children.append(stack[-1]) # adding children to root
                is_passing, feedback, reward = gen_test_eval(exe, cur_func_impl, tests_i, prev=item["prev"])
                num_try += 1
                stack[-1].update(reward)
                stack[-1].test_feedback = feedback
            if is_passing:
                is_solved = exe.evaluate(
                    item["entry_point"], cur_func_impl, item["test"], timeout=1,prev=item["prev"])  # early exit
                if "weaker_test" in item.keys():
                    is_weaker_solved = exe.evaluate(
                    item["entry_point"], cur_func_impl, item["weaker_test"], timeout=1, prev=item["prev"])
                break
        # Exit when passed public test cases.
        if is_passing:
            if is_solved:
                num_success += int(is_solved)
                passed_at_sample.append(num_try)
                if "difficulty" in item.keys(): pass_problem_subset.append(item["difficulty"])
            if is_weaker_solved:
                weak_success += int(is_weaker_solved)
            item["weak_acc"] = round(weak_success / (idx + 1), 3)
            item["acc"] = round(num_success / (idx + 1), 3)
            write_jsonl(log_path, [item], append=True)
            print_v(f'completed {idx + 1}/{num_items}: acc = {round(num_success / (idx + 1), 3)}, weak_acc={item["weak_acc"]}')
            continue  # early stop on this case if passsed
    print("_______________________________")
    print(passed_at_sample)
    print(sorted(passed_at_sample))
    print(len(passed_at_sample))
    print(Counter(passed_at_sample))
    print(Counter(pass_problem_subset))
    # write_jsonl(log_path, [item], append=True)
    print_v(f'completed {idx + 1}/{num_items}: acc = {round(num_success / (idx + 1), 3)}, weak_acc={round(weak_success / (idx + 1), 3)}')

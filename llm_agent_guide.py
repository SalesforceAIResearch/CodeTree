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

import openai
from utils import make_printv, write_jsonl
from executors import executor_factory
from generators import generator_factory, model_factory
from typing import List
import sys
from common import gen_test_eval
from collections import Counter

sys.set_int_max_str_digits(100000)  # Increase the limit to 10000 digits

ADD_HINT = "To solve the problem, you can refer the hint given by an expert, and complete the details by analyzing it first.\nHint:"

# TODO: From sample to list
class Node:
    def __init__(self, solution: str, parent=None, strategy="", reflection="", depth=0):
        self.solution = solution
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.context = ""
        self.depth = depth
        self.reflection = reflection
        self.test_feedback = ""
        self.strategy=strategy
        self.pass_visible = False

    def best_child(self):
        if not self.children:  # Check if children list is empty
            return None
        return max(self.children, key=lambda child: child.uct())

    def best_child_value(self):
        if not self.children:  # Check if children list is empty
            return None
        return max(self.children, key=lambda child: child.value)

    def sort_children_by_value(self):
        self.children.sort(key=lambda x: x.value, reverse=True)

    def update(self, reward: float):
        self.visits += 1
        self.value += reward

    # only keeps the most recent blocks

def eval_node(prompt, node:Node, gen, model, max_depth=3):
    """
    The evaluation shows not fully correct, decide whether to go on.
    """
    if node.parent is None: parent_value=0
    else: parent_value =  node.parent.value
    if node.depth >= max_depth: return False, node.value
    agent_reward, analysis = gen.agent_eval(prompt, model, prev_func_impl=node.solution, task="tests",
                                  feedback=node.test_feedback.split("[additional review]:")[0].strip())
    node.value += float(agent_reward) / 15
    if node.value < parent_value: return False, node.value
    elif node.value == parent_value and node.depth > agent_reward: return False, node.value
    return True, node.value, analysis

def step_verify(gen, exe, item, solution, model):
    """
    if pass all public test cases, run one agent review step
    """
    is_passing, feedback, reward = gen_test_eval(exe, solution, item["given_tests"], prev=item["prev"])
    if not is_passing:
        return False, feedback, reward
    else:
        reward = 1
        option, analysis = gen.agent_eval(item["prompt"], model, prev_func_impl=solution,
                                      task="stop", feedback=feedback, temperature=0)
        if option: return True, feedback, reward
        else:
            return False, f"{feedback}\n\n[additional review]:\n\n{analysis}", reward


def rerank_list_of_nodes(list_of_nodes):
    return sorted(list_of_nodes, key=lambda x:x.value, reverse=True) # small value in the front

def agent_guide(
        dataset: List[dict],
        model_name: str,
        language: str,
        log_path: str,
        verbose: bool,
        max_depth: int = 3,
        search_width: int = 10,
        max_iters: int=20,
        Codecontests: bool = False
) -> None:
    print("max_depth", max_depth, "search_width", search_width)
    pass_problem_subset = []
    if Codecontests:
        exe = executor_factory("code_contests")
    else:
        exe = executor_factory(language, is_leet=False)

    print("Len(dataset)", len(dataset), dataset[0].keys())
    gen = generator_factory(language)
    model = model_factory(model_name)
    print_v = make_printv(verbose)
    count, sad_case, debug_thoroughed_case, enter_debug_case = 0, 0, 0, 0
    num_items = len(dataset)
    num_success, weak_success = 0, 0  # Counter for successful solutions
    passed_at_sample, solve_or_not = [], []
    for idx, item in enumerate(dataset):
        print("STARTING EXAMPLE", idx)
        if Codecontests:
            item["entry_point"] = ""
        else: item["given_tests"] = [test for test in item["given_tests"] if 'assert False' not in test]
        # Thinker Agent Preparation
        hints = gen.strategy(item["prompt"], model, num_strategy="multiple", task="strategy", temperature=0)
        if len(hints) > search_width: hints = hints[:search_width] # width cut
        stack = []
        is_passing, is_solved, is_weaker_solved = False, False, False
        num_try = 0
        for hint in reversed(hints):
            new_node = Node("", strategy=hint, depth=1)
            stack.append(new_node) # initial placeholders for new nodes

        # Tree Search Start
        found_solution = None
        candidate_solution = None
        while stack and num_try < max_iters and not is_passing:
            if len(stack) == 0: break
            this_node = stack.pop()
            if this_node.depth > max_depth: continue
            # Solver Agent
            if not this_node.solution:
                cur_solution = gen.func_impl(item["prompt"] + f"{ADD_HINT}{this_node.strategy}\n",
                                                   model, "simple", temperature=0)
                if not candidate_solution: candidate_solution = cur_solution

            # Debugger Agent
            else:
                cur_solution = gen.func_impl(
                    func_sig=item["prompt"],
                    model=model,
                    strategy="reflexion",
                    prev_func_impl=this_node.solution,
                    feedback=this_node.test_feedback.split("[additional review]:")[0].strip(),
                    self_reflection=this_node.reflection,
                    temperature=0
                )
            num_try += 1

            # Execute and get Feedback
            is_passing, feedback, reward = step_verify(gen, exe, item, cur_solution, model)
            print("cur solution judge", is_passing)

            # Update node information as parent
            this_node.solution = cur_solution # update the solution to real solution
            this_node.test_feedback = feedback # With additional critic feedback
            this_node.value = reward

            if reward > 0.99:
                this_node.pass_visible = True
                if this_node.parent and this_node.parent.pass_visible and this_node.depth==max_depth:
                        is_passing = True
                this_node.value += 5/15 # reward for passing all visible
            if is_passing:
                found_solution = cur_solution
                break

            elif reward <= 0.99: # didn't pass, need debugging
                candidate_solution = cur_solution
                go_on, values, analysis = eval_node(prompt=item["prompt"],node=this_node,gen=gen,model=model,max_depth=max_depth)
                this_node.value = values

            # Continue on this node
            else: go_on = this_node.pass_visible

            if go_on:
                # Thinker Agent, init startegies for potential agents
                reflections = gen.strategy(item["prompt"],
                                           model, task="reflection",
                                           num_strategy="one or multiple (if there is)",
                                           prev_func_impl=this_node.solution,
                                           feedback=this_node.test_feedback,
                                           temperature=0,
                                           given_strategy=this_node.strategy)
                if len(reflections) > search_width: reflections = reflections[:search_width]
                for reflection in reversed(reflections):
                    if not reflection: continue
                    new_node = Node(cur_solution, reflection=reflection, parent=this_node, strategy=this_node.strategy, depth=this_node.depth + 1) # init with previous code
                    new_node.test_feedback = this_node.test_feedback
                    this_node.children.append(new_node) # children in a reverse order
            stack.extend(this_node.children)
        if num_try >= max_iters: debug_thoroughed_case += 1
        # Verify that values are actually fair for all nodes.

        if found_solution is None: found_solution = candidate_solution

        is_solved = exe.evaluate(
            item["entry_point"], found_solution, item["test"], timeout=10, prev=item["prev"])  # early exit
        if "weaker_test" in item.keys():
            is_weaker_solved = exe.evaluate(
                item["entry_point"], found_solution, item["weaker_test"], timeout=10, prev=item["prev"])
        if is_solved:
            num_success += int(is_solved)
            passed_at_sample.append(num_try)
            if "difficulty" in item.keys(): pass_problem_subset.append(item["difficulty"])
        else:
            sad_case += 1
            print("Sad, Pass but not solve")

        if is_weaker_solved:
            weak_success += int(is_weaker_solved)
        item["acc"] = round(num_success / (idx + 1), 3)
        item["weak_acc"] = round(weak_success / (idx + 1), 3)
        write_jsonl(log_path, [item], append=True)
        print_v(
            f'completed {idx + 1}/{num_items}: acc = {round(num_success / (idx + 1), 3)}, weak_acc={item["weak_acc"]}, pass no solve: {sad_case}, exhaust: {debug_thoroughed_case}')

    print("_______________________________")
    print(passed_at_sample)
    print(sorted(passed_at_sample))
    print(len(passed_at_sample))
    print(Counter(passed_at_sample))
    print("Passed but not solved case", sad_case)
    print(f"{max_iters} tries used still not solve:", debug_thoroughed_case)
    print(Counter(pass_problem_subset))
    print_v(
        f'completed {idx + 1}/{num_items}: acc = {round(num_success / (idx + 1), 3)}, weak_acc={round(weak_success / (idx + 1), 3)}')

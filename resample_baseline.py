from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
import sys
from common import gen_test_eval
from generators import generator_factory, model_factory
from typing import List, Dict, Tuple, Any
import math
import sys
from collections import Counter
sys.set_int_max_str_digits(100000)  # Increase the limit to 10000 digits

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
            # return float('inf')
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

def resample(
        dataset: List[dict],
        model_name: str,
        language: str,
        max_iters: int,
        log_path: str,
        verbose: bool,
        is_leetcode: bool = False,
        Codecontests: bool = False
) -> None:
    if Codecontests: 
        exe = executor_factory("code_contests")
    else: exe = executor_factory(language, is_leet=is_leetcode)

    pass_problem_subset = []
    gen = generator_factory(language)
    model = model_factory(model_name)
    print_v = make_printv(verbose)

    num_items = len(dataset)
    num_success, weak_success = 0, 0  # Counter for successful solutions
    passed_at_sample, solve_or_not = [], []

    for idx, item in enumerate(dataset):
        print("STARTING EXAMPLE", idx)
        tests_i = item["given_tests"]
        if Codecontests:
            item["entry_point"] = ""
        else:
            tests_i = [test for test in tests_i if item['entry_point'] in test and 'assert False' not in test]
        root = Node("")
        stack = [root] # implementations
        is_solved, is_weaker_solved = False, False
        num_try = 0
        for i in range(max_iters):
            cur_func_impl = None
            while cur_func_impl is None:
                cur_func_impl = gen.func_impl(item["prompt"], model, "simple",  temperature=1.0)
                stack.append(Node(cur_func_impl))
                stack[0].children.append(stack[-1])
                is_passing, feedback, reward = gen_test_eval(exe, cur_func_impl, tests_i, prev=item["prev"])
                num_try += 1
                stack[-1].update(reward)
                stack[-1].test_feedback = feedback
            if is_passing:
                is_solved = exe.evaluate(
                    item["entry_point"], cur_func_impl, item["test"], timeout=1, prev=item["prev"])  # early exit
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
            else: print("sad, passed but no solve.")
            if is_weaker_solved:
                weak_success += int(is_weaker_solved)
            item["acc"] = round(num_success / (idx + 1), 3)
            item["weak_acc"] = round(weak_success / (idx + 1), 3)

            write_jsonl(log_path, [item], append=True)
            print_v(f'completed {idx + 1}/{num_items}: acc = {round(num_success / (idx + 1), 3)}, weak_acc={round(weak_success / (idx + 1), 3)}')
            continue  # early stop on this case if passsed
    print("_______________________________")
    print(passed_at_sample)
    print(sorted(passed_at_sample))
    print(len(passed_at_sample))
    print(Counter(passed_at_sample))
    print(Counter(pass_problem_subset))


    # write_jsonl(log_path, [item], append=True)
    print_v(f'completed {idx + 1}/{num_items}: acc = {round(num_success / (idx + 1), 3)}, weak_acc={round(weak_success / (idx + 1), 3)}')

import openai
from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory
from typing import List, Dict, Tuple, Any
import math
import re
import sys
from collections import Counter
from common import gen_test_eval
# bfs real
sys.set_int_max_str_digits(100000)  # Increase the limit to 10000 digits
# Many are passed but not solved, so if one has passed, use the agreement function to select one

ADD_HINT = "To solve the problem, you can refer the hint given by an expert, and complete the details by analyzing it first.\nHint:"

# TODO: From sample to list
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
        self.strategy=""

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
        self.children.sort(key=lambda x: x.value, reverse=True)

    def update(self, reward: float):
        self.visits += 1
        self.value += reward


def rerank_list_of_nodes(list_of_nodes):
    return sorted(list_of_nodes, key=lambda x:x.value, reverse=True) # small value in the front

def run_bfs(
        dataset: List[dict],
        model_name: str,
        language: str,
        log_path: str,
        verbose: bool,
        max_iters: int,
        is_leetcode: bool = False,
        max_depth: int = 3,
        search_width: int = 3,
        Codecontests: bool = False
) -> None:
    if Codecontests: exe = executor_factory("code_contests")
    else: exe = executor_factory(language, is_leet=is_leetcode)

    gen = generator_factory(language)
    model = model_factory(model_name)
    print_v = make_printv(verbose)
    count, sad_case, debug_thoroughed_case, enter_debug_case = 0, 0, 0, 0
    num_items = len(dataset)
    num_success, weak_success = 0, 0  # Counter for successful solutions
    passed_at_sample, solve_or_not = [], []
    debug_case, skip = 0, 0
    pass_problem_subset = []
    for idx, item in enumerate(dataset):
        tests_i = item["given_tests"]
        if Codecontests:
            item["entry_point"] = ""
        else:
            tests_i = [test for test in tests_i if item['entry_point'] in test and 'assert False' not in test]

        hints = gen.strategy(item["prompt"], model, num_strategy=search_width, task="strategy", temperature=0)
        if len(hints) > search_width: hints = hints[:search_width]
        stack, memory_stack = [], []
        is_solved, is_weaker_solved = False, False
        num_try = 0
        if len(hints) < search_width:
            count += 1
        for hint in hints:
            cur_func_impl = gen.func_impl(item["prompt"] + f"{ADD_HINT} {hint}\n", model, "simple",
                                          temperature=0)
            new_node = Node(cur_func_impl)
            num_try += 1
            is_passing, feedback, reward = gen_test_eval(exe, cur_func_impl, tests_i, prev=item["prev"])
            if is_passing:
                is_solved = exe.evaluate(
                    item["entry_point"], cur_func_impl, item["test"], timeout=1, prev=item["prev"])  # early exit
                if "weaker_test" in item.keys():
                    is_weaker_solved = exe.evaluate(
                    item["entry_point"], cur_func_impl, item["weaker_test"], timeout=1, prev=item["prev"])
                break
            new_node.test_feedback = feedback
            new_node.update(reward)
            new_node.strategy = hint
            stack.append(new_node)
        # Exit when passed public test cases.

        if is_passing:
            if is_solved:
                num_success += int(is_solved)
                passed_at_sample.append(num_try)
                if "difficulty" in item.keys(): pass_problem_subset.append(item["difficulty"])
            else:
                print("SAD, passed but not solved.")
                sad_case += 1
            if is_weaker_solved:
                weak_success += int(is_weaker_solved)
            item["weak_acc"] = round(weak_success / (idx + 1), 3)
            item["acc"] = round(num_success / (idx + 1), 3)
            write_jsonl(log_path, [item], append=True)
            print_v(f'completed {idx + 1}/{num_items}: acc = {round(num_success / (idx + 1), 3)},  weak_acc={item["weak_acc"]}, pass no solve: {sad_case}, enter debug: {debug_case}')
            continue  # early stop on this case if passsed

        print("Entering Debugging Stage")
        debuged = True; debug_case += 1
        stack = rerank_list_of_nodes(stack) # out of all stack
        print("Stack after sorting: ", [a.value for a in stack])
        while stack and num_try < max_iters and not is_passing:
            this_node = stack.pop(0)
            if this_node.depth >= max_depth: continue
            this_node.visits += 1
            reflections = gen.strategy(item["prompt"],
                                                model, task="reflection",
                                                num_strategy=search_width,
                                                prev_func_impl=this_node.solution,
                                                feedback=this_node.test_feedback,
                                                temperature=0,
                                                given_strategy=this_node.strategy)
            if len(reflections) < 2: "print not enough reflections!"
            for reflection in reflections:
                if num_try >= max_iters: break
                new_solution, while_cnt = None, 0
                while new_solution is None and while_cnt < 3:
                    while_cnt += 1
                    new_solution = gen.func_impl(
                        func_sig=item["prompt"],
                        model=model,
                        strategy="reflexion",
                        prev_func_impl=this_node.solution,
                        feedback=this_node.test_feedback,
                        self_reflection=reflection,
                        temperature=0
                    )
                is_passing, feedback, reward = gen_test_eval(exe, new_solution, tests_i, prev=item["prev"])
                num_try += 1
                if is_passing:
                    is_solved = exe.evaluate(
                        item["entry_point"], new_solution, item["test"], timeout=1, prev=item["prev"])
                    if "weaker_test" in item.keys():
                        is_weaker_solved = exe.evaluate(
                            item["entry_point"], new_solution, item["weaker_test"], timeout=1, prev=item["prev"])
                    break
                new_node = Node(new_solution, depth=this_node.depth + 1)
                new_node.test_feedback = feedback
                new_node.update(reward)
                new_node.strategy = this_node.strategy
                this_node.children.append(new_node)
            if is_passing: break
            this_node.sort_children_by_value()
            stack.extend(this_node.children)
            print("Children after sorting: ", [a.value for a in stack])
        if num_try >= max_iters: debug_thoroughed_case += 1
        if is_passing:
            if debuged: enter_debug_case += 1
            if is_solved:
                num_success += int(is_solved)
                passed_at_sample.append(num_try)
                if "difficulty" in item.keys(): pass_problem_subset.append(item["difficulty"])
            else:
                sad_case += 1
                print("Sad, pass but not solve")
            if is_weaker_solved:
                weak_success += int(is_weaker_solved)
            item["weak_acc"] = round(weak_success / (idx + 1), 3)
            item["acc"] = round(num_success / (idx + 1), 3)
            write_jsonl(log_path, [item], append=True)
            print_v(f'completed {idx + 1}/{num_items}: acc = {round(num_success / (idx + 1), 3)}, weak_acc={item["weak_acc"]}, pass no solve: {sad_case}, enter debug: {debug_case}')
            continue

    print("_______________________________")
    print(passed_at_sample)
    print(sorted(passed_at_sample))
    print(len(passed_at_sample))
    print(Counter(passed_at_sample))
    print("Passed but not solved case", sad_case)
    print("not sample 2 even when asked: ", count)
    print("20 tries used still not solve:", debug_thoroughed_case)
    print("Pass not solve after debugging", enter_debug_case)
    print(Counter(pass_problem_subset))
    print_v(f'completed {idx + 1}/{num_items}: acc = {round(num_success / (idx + 1), 3)}, weak_acc={round(weak_success / (idx + 1), 3)}')


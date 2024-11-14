from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory
from typing import List, Dict, Tuple, Any

import sys
from collections import Counter
sys.set_int_max_str_digits(100000)  # Increase the limit to 10000 digits
from common import gen_test_eval
def run_reflexion(
        dataset: List[dict],
        model_name: str,
        max_iters: int,
        language: str,
        log_path: str,
        verbose: bool,
        is_leetcode: bool = False,
        Codecontests: bool = False
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
    pass_problem_subset = []

    for idx, item in enumerate(dataset):
        tests_i = item["given_tests"]
        if Codecontests:
            item["entry_point"] = ""
        else:
            tests_i = [test for test in tests_i if item['entry_point'] in test and 'assert False' not in test]
        is_solved, is_weaker_solved, is_passing = False, False, False
        num_try = 0
        cur_func_impl = gen.func_impl(item["prompt"], model, "simple")
        is_passing, feedback, reward = gen_test_eval(exe, cur_func_impl, tests_i, prev=item["prev"])
        num_try += 1
        cur_feedback = feedback
        for i in range(max_iters-1):
            if is_passing: break
            reflection = gen.self_reflection(
                cur_func_impl, cur_feedback, model)
            cur_func_impl = gen.func_impl(
                func_sig=item["prompt"],
                model=model,
                strategy="reflexion",
                prev_func_impl=cur_func_impl,
                feedback=cur_feedback,
                self_reflection=reflection,
            )
            is_passing, cur_feedback, reward = gen_test_eval(exe, cur_func_impl, tests_i, prev=item["prev"])
            num_try += 1

        # Exit when passed public test cases.
        if is_passing:
            is_solved = exe.evaluate(
                item["entry_point"], cur_func_impl, item["test"], timeout=1, prev=item["prev"])  # early exit
            if "weaker_test" in item.keys():
                is_weaker_solved = exe.evaluate(
                    item["entry_point"], cur_func_impl, item["weaker_test"], timeout=1, prev=item["prev"])
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

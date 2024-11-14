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

from .model import ModelBase, Message, messages_to_str
import random
#from ..main import Codecontests
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import get_parsed_args
args = get_parsed_args()
Codecontests = False if args.function else True
from typing import Union, List, Optional, Callable
PY_STRATEGY = ("You are an AI assistant that provides strategy for Python programmers to code. You will be given a function signature and its docstring by the user. "
             "Your goal is to think of {py_strategy_k} "
             "strategies in English(Not Code) on how to approach this problem and solve it. Describe each strategy with a FEW sentences in a SINGLE Line. List and Number your strategies line by line using \"1. \"; \"2. \"; \"3. \" and so on.")
Prompt_flexible = "The number of alternatives(either one or multiple) should be determined given this specific case."
PY_IMPLEMENT = """You are an AI assistant who helps the user write code. The user will give you a function signature and its docstring and also suggest a strategy. You should instruct (in English) the user to implement their strategy, adding details not provided in the strategy. You must give {py_strategy_k} alternatives on how to implement the strategy exactly. Each alternative should be FEW sentences in a SINGLE line. List and number your {py_strategy_k} implementation alternatives using \"1. \", \"2. \"."""
PY_REFELCTION = ("You are an AI assistant who can reflect on problem-solving solution program. You will be given a task, an incorrect function implementation and feedbacks from executing the code. Your goal is to describe the existing issue(s) and suggest methods on how to improve the code. Rules:\n"
                 "1. From the algorithm and implementation level, there could be multiple methods to fix the error, you should provide {py_strategy_k} alternative reflections using various strategies. If the bug is clouded and ambigious, you can use alternatives as different interpretations, too.\n"
                 "2. Each reflection should briefly describe the issues and bugs, what kind of improvement is needed, then describe how to implement the correction. You are allowed to restate the bug for each reflection if needed. Each reflection should start be complete and self-contained. In other words, if there are more than one bugs, they should be presented in one reflection rather than separately.\n"
                 "3. Answer format: List and number your alternatives line by line, starting with \"1. \", \"2. \" and so on. Each reflection alternative is in a single line within a few sentences.\n")


system_stop_simplified = """The user will provide a programming task along with a solution that passes all visible test cases. Your task is to further review the solution before it is judged against hidden test cases. Determine whether the solution is robust and general enough to pass unseen, valid test cases. Guideline:
    1. Generalization Check: Verify that the solution uses general methods, avoiding hardcoding specific values or cases unless explicitly required. Confirm that the approach logically extends to unseen cases without special assumptions.
    2. Boundary Check: Ensure all boundaries are correctly handled, including list indexing, loop start and end points, if-else conditions, and recursion exits. Look for potential off-by-one errors or boundary misses that could cause functional errors.
	3. Edge Case Check: Confirm that the solution correctly handles valid edge/corner cases, such as zero, negative, empty, boundary values, or other special problem-specific situations. Note: All unseen test cases are guaranteed to follow stated data types, formats, conditions, and other constraints in the problem, no need to handle unallowed inputs. Do NOT apply redundant handling for cases that the current solution inherently manages, such as empty lists in sorting algorithms (`sorted([]) → []`), unless they explicitly fail (e.g., `max([]) → error`).
	4. Major Efficiency Check: Check if the solution is within polynomial time/space complexity, if NOT, fail this check.

**Response Format**:
Firstly, within several sentences, follow the guideline and briefly analyze.
On a new line, respond with “True” if the solution is ACCEPTABLE as-is, or “False” if NECESSARY modifications are required to handle unseen valid test cases.

The following is one example of how to review:
<EXAMPLE 1>:
```python
def find_first_unique(nums: list[int]) -> int:
    \"\"\"
    Find the first unique integer in a list of integers.
    Args: nums (list[int]): A list of integers to search through.
    Returns: int: The first unique integer in the list, or -1 if no unique integer is found.
    Examples:
        >>> find_first_unique([4, 5, 1, 2, 0, 4])  ==>  5
        >>> find_first_unique([7, 3]) ==> 7
    \"\"\"
    for i, num in enumerate(nums):
        if num not in nums[i:]: return num
    return -1
```
<EXAMPLE 1 Review>:
1. Generalization Check: `num not in nums[i:]` won’t handle cases where the number appears previous positions, `find_first_unique([7, 7]) ==> 7` instead of -1. Other checks are omitted for now since the solution logic is wrong.
False
"""
if Codecontests:
    PY_STRATEGY = PY_STRATEGY.replace("a function signature and its docstring", "a programming problem and its required input/output formats")
    PY_IMPLEMENT = PY_IMPLEMENT.replace("a function signature and its docstring", "a programming problem and its required input/output formats")
    PY_REFELCTION = PY_REFELCTION.replace("function implementation", "solution program to a problem")

def generic_generate_func_impl(
    func_sig: str,
    model: ModelBase,
    strategy: str,
    prev_func_impl,
    feedback,
    self_reflection,
    num_comps,
    temperature,
    reflection_chat_instruction: str,
    reflection_few_shot: str,
    simple_chat_instruction: str,
    reflection_completion_instruction: str,
    simple_completion_instruction: str,
    code_block_instruction: str,
    parse_code_block: Callable[[str], str],
    add_code_block: Callable[[str], str]
) -> Union[str, List[str]]:
    if strategy != "reflexion" and strategy != "simple" and strategy != "self-repair":
        raise ValueError(
            f"Invalid strategy: given `{strategy}` but expected one of `reflexion` or `simple`")
    if strategy == "reflexion" and (prev_func_impl is None or feedback is None or self_reflection is None):
        raise ValueError(
            f"Invalid arguments: given `strategy=reflexion` but `prev_func_impl`, `feedback`, or `self_reflection` is None")
    if model.is_chat:
        func_bodies = None
        if strategy == "reflexion":
            # message = f"{reflection_few_shot}\n[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[unit test results from previous impl]:\n{feedback}\n\n[reflection on previous impl]:\n{self_reflection}\n\n[improved impl]:\n{func_sig}"
            prompt = f"{reflection_chat_instruction}\n{code_block_instruction}"
            # func_bodies is a really bad name, as it can also be just 1 string
            #print_messages(prompt, message)
            messages = [
                Message(
                    role="system",
                    content=prompt,
                ),
                Message(
                    role="user", # TODO: check this
                    content=f"Here's the challenge for you:\n{func_sig}\n[implement]:\n",
                ),
                Message(
                    role="assistant",
                    content=f"{add_code_block(prev_func_impl)}"
                ),
                Message(
                    role="user",
                    content=f"[unit test results from previous implement]:\n{feedback}\n\n[reflection on previous implement]:\n",
                ),
                Message(
                    role="assistant",
                    content=self_reflection+"\n",
                ),
                Message(
                    role="user",
                    content=f"[improved implement]:\n",
                ),
            ]
            print(messages_to_str(messages))
            func_bodies = model.generate_chat(messages=messages, num_comps=num_comps, temperature=temperature, max_tokens=4096)
        else: # Simple
            messages = [
                Message(
                    role="system",
                    content=f"{simple_chat_instruction}\n{code_block_instruction}",
                ),
                Message(
                    role="user",
                    content=func_sig,
                ),
            ]
            func_bodies = model.generate_chat(messages=messages, num_comps=num_comps, temperature=temperature, max_tokens=4096)

    else:
        if strategy == "reflexion":
            prompt = f"{reflection_completion_instruction}\n{add_code_block(prev_func_impl)}\n\nunit tests:\n{feedback}\n\nhint:\n{self_reflection}\n\n# improved implementation\n{func_sig}\n{code_block_instruction}"
            func_bodies = model.generate(
                prompt, num_comps=num_comps, temperature=temperature)
        else:
            prompt = f"{simple_completion_instruction}\n{func_sig}\n{code_block_instruction}"
            func_bodies = model.generate(
                prompt, num_comps=num_comps, temperature=temperature)

    if num_comps == 1:
        assert isinstance(func_bodies, str)
        #print("model responses!", func_bodies)
        func_body_str = parse_code_block(func_bodies)
        print_generated_func_body(func_body_str)
        return func_body_str

    else:
        try:
            func_bodies = [parse_code_block(func_body) for func_body in func_bodies]
            print_generated_func_body("\n\n".join(func_bodies))
        except:
            print(func_bodies)

        return func_bodies


def generate_with_accumulated_context(
    func_sig: str,
    model: ModelBase,
    strategy: str,
    prev_func_impl,
    accumulated_feedback,
    accumulated_reflection,
    num_comps,
    temperature,
    reflection_chat_instruction: str,
    reflection_few_shot: str,
    simple_chat_instruction: str,
    reflection_completion_instruction: str,
    simple_completion_instruction: str,
    code_block_instruction: str,
    parse_code_block: Callable[[str], str],
    add_code_block: Callable[[str], str]
) -> Union[str, List[str]]:
    # Ensure that the strategy is valid
    if strategy != "reflexion" and strategy != "simple":
        raise ValueError(
            f"Invalid strategy: given `{strategy}` but expected one of `reflexion` or `simple`")
    if strategy == "reflexion" and (prev_func_impl is None or accumulated_feedback is None or accumulated_reflection is None):
        raise ValueError(
            f"Invalid arguments: given `strategy=reflexion` but `prev_func_impl`, `feedback`, or `self_reflection` is None")

    # Build the accumulated context from the provided feedback and reflections
    accumulated_context = "\n\n".join(
        [f"[previous impl {i+1}]:\n{add_code_block(impl)}\n[unit test results from previous impl {i+1}]:\n{feedback}\n[reflection on previous impl {i+1}]:\n{reflection}" 
         for i, (impl, feedback, reflection) in enumerate(zip(prev_func_impl, accumulated_feedback, accumulated_reflection))]
    )

    if model.is_chat:
        if strategy == "reflexion":
            # Constructing the message using a loop for accumulated context
            messages = [
                Message(role="system", content=f"{reflection_chat_instruction}\n{code_block_instruction}"),
                Message(role="user", content=reflection_few_shot)
            ]
            
            for impl, feedback, reflection in zip(prev_func_impl, accumulated_feedback, accumulated_reflection):
                messages.append(Message(role="assistant", content=add_code_block(impl)))
                messages.append(Message(role="user", content=f"[unit test results from previous impl]:\n{feedback}\n\n[reflection on previous impl]:\n{reflection}"))
            
            messages.append(Message(role="user", content=f"[improved impl]:\n{func_sig}"))
            prompt = "\n".join([message.content for message in messages])
            message = (f"{reflection_few_shot}\n{accumulated_context}\n\n[improved impl]:\n{func_sig}")
            print_messages(prompt, message)

            func_bodies = model.generate_chat(messages=messages, num_comps=num_comps, temperature=temperature)
        else:
            system_prompt = f"{simple_chat_instruction}\n{code_block_instruction}"
            print_messages(system_prompt, func_sig)
            messages = [
                Message(role="system", content=f"{simple_chat_instruction}\n{code_block_instruction}"),
                Message(role="user", content=func_sig)
            ]
            func_bodies = model.generate_chat(messages=messages, num_comps=num_comps, temperature=temperature)
    else:
        if strategy == "reflexion":
            prompt = f"{reflection_completion_instruction}\n{accumulated_context}\n\n# improved implementation\n{func_sig}\n{code_block_instruction}"
            func_bodies = model.generate(prompt, num_comps=num_comps, temperature=temperature)
            print_messages(prompt, "")  
        else:
            prompt = f"{simple_completion_instruction}\n{func_sig}\n{code_block_instruction}"
            func_bodies = model.generate(prompt, num_comps=num_comps, temperature=temperature)
            print_messages(prompt, "")

    if num_comps == 1:
        assert isinstance(func_bodies, str)
        func_body_str = parse_code_block(func_bodies)
        print_generated_func_body(func_body_str)
        return func_body_str

    else:
        func_bodies = [parse_code_block(func_body) for func_body in func_bodies]
        print_generated_func_body("\n\n".join(func_bodies))
        return func_bodies
    

def generic_generate_internal_tests(
        func_sig: str,
        model: ModelBase,
        max_num_tests: int,
        test_generation_few_shot: str,
        test_generation_chat_instruction: str,
        test_generation_completion_instruction: str,
        parse_tests: Callable[[str], List[str]],
        is_syntax_valid: Callable[[str], bool],
        is_react: bool = False
) -> List[str]:
    """Generates tests for a function."""
    if model.is_chat:
        if is_react:
            messages = [
                Message(
                    role="system",
                    content=test_generation_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f"{test_generation_few_shot}\n\n[func signature]:\n{func_sig}\n\n[think]:"
                )
            ]
            output = model.generate_chat(messages=messages, max_tokens=2048)
            print(f'React test generation output: {output}')
        else:
            messages = [
                Message(
                    role="system",
                    content=f"{test_generation_chat_instruction}\n\n{test_generation_few_shot}",
                ),
                Message(
                    role="user",
                    content=f"[func signature]:\n{func_sig}\n\n[unit tests]:",
                )
            ]
            output = model.generate_chat(messages=messages, max_tokens=2048)
    else:
        prompt = f'{test_generation_completion_instruction}\n\nfunc signature:\n{func_sig}\nunit tests:'
        output = model.generate(prompt, max_tokens=2048)
    all_tests = parse_tests(output)  # type: ignore
    valid_tests = [test for test in all_tests if is_syntax_valid(test)]

    return sample_n_random(valid_tests, max_num_tests)


def generic_generate_self_reflection(
        func: str,
        feedback: str,
        model: ModelBase,
        self_reflection_chat_instruction: str,
        self_reflection_completion_instruction: str,
        add_code_block: Callable[[str], str],
        self_reflection_few_shot: Optional[str] = None,
        task = "evaluation"
) -> str:
    if model.is_chat:
        if task == "evaluation":
            messages = [
                Message(
                    role="system",
                    content=self_reflection_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f'[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\nThis function passed visible tests, please further evaluate the code. Your options are "1. Correct implementation of desired function", "2. Mostly correct implementation, didn\'t consider edge/corner cases", "Only fits some situations, not the desired functionality."',
                )
            ]
        if self_reflection_few_shot is not None:
            messages = [
                Message(
                    role="system",
                    content=self_reflection_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f'{self_reflection_few_shot}\n\n[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:',
                )
            ]
            reflection = model.generate_chat(messages=messages)
            print(f'Self reflection output: {reflection}')
        else:
            messages = [
                Message(
                    role="system",
                    content=self_reflection_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f'[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:',
                )
            ]
            reflection = model.generate_chat(messages=messages)
    else:
        reflection = model.generate(
            f'{self_reflection_completion_instruction}\n{add_code_block(func)}\n\n{feedback}\n\nExplanation:')
    return reflection  # type: ignore
def generic_evaluate(func_sig: str,
             model: ModelBase,
             parse_response: Callable,
             task = "stop",
             code = "",
             temperature=0.0,
             lang="python",
             exe_feedback = "",
             code_impr=None
             ):
    """
    1. Which strategy to explore first
    2. When pass_public_test, whether the current solution is acceptable, or keep exploring
    3. Whether rollback to before-fix; keeping the summary of this_fix, mark as fail
    """
    if task == "eval":
        messages = [
            Message(role="system", content="Your task is to evaluate a strategy and corresponding implementation for solving a programming problem. You should score from 1 to 5 separately on the following aspects.\n"
                                           "Correctness: How well can the solution solve the task?\n"
                                           "Simpleness: How straightforward is the implementation given the difficulty of the problem?\n"
                                           "Generalizability: How well can this solution cover all cases, even ones not mentioned in examples?\n"
                                           "Insightfulness: Even when the solution is incorrect, how well does it point out a good direction to solve the problem?\n"
                                           "Your scores should use the follwing standards. 1: bad, 2: not too bad, 3: fair, 4: good, 5: excellent"),
            Message(role="user", content=f"Task Description:\n{func_sig}\n\nCode to Evaluate:\n```{lang}\n{code}\n```\nFeedback from executing the code on visible test cases:\n\n{exe_feedback}")
        ]
    elif task == "stop":
        messages = [
            Message(role="system", content=system_stop_simplified),
            Message(role="user", content=f"Task Description:\n{func_sig}\n\nCode to Evaluate:\n```{lang}\n{code}\n```\nFeedback from executing the code on visible test cases:\n```\n{exe_feedback}\n```")
        ]
        print(messages_to_str(messages))
    elif task == "tests":
        messages = [
            Message(role="system",
                    content="Your task is to evaluate the execution outputs of a code implementation. The statement and code is given by the user, and the output/expected output on a set of test cases."
                            "Your should analyze the expected outputs and execution outputs. From a 0 to 5 range, you should give a score on how far the execution outputs are from the expected ones. Standards are below:\n"
                            "\n0: Errors or time out when executing.\n"
                            "\n1: No pattern found when comparing pairs of <output, expected_output>, errors are hard to interpret.\n"
                            "\n2: Results abnormal for a part of cases(e.g., cannot handle negative elements; only half of it sorted).\n"
                            "\n3: Result pairs have clear patterns(e.g., all elements offset by 1; all elements + 1; corp by value; reverse all elements...)\n"
                            "\n4: Lack consideration of edge condition/corner cases(e.g., error only when elements are equal), otherwise correct.\n"
                            "\n5: Results matched.\n"
                            "\nGive your brief analysis first. Afterwards, start a new line with A SINGLE INTEGER NUMBER as your final score(0 to 5)."),
            Message(role="user",
                    content=f"Task Description:\n{func_sig}\n\nCode to Evaluate:\n```{lang}\n{code}\n```\nFeedback from executing the code on visible test cases:\n\n{exe_feedback}")
        ]
    # input: test; goal: score the test output vs. expected output; for CodeContests
    elif task == "compare":
        assert code_impr is not None
        messages = [
            Message(role="system",
                    content="Your task is to compare a pair of solutions. The SECOND solution is a bug-fixing attempt to the FIRST solution, which fails to fix the bug. You should evaluate the attempt on whether it should be rollbacked. You should first analyze, and answer 'Rollback.' or 'Keep.'as the last word of your response."),
            Message(role="user",
                    content=f"Task Description:\n{func_sig}\n\nCode to Evaluate:\n```{lang}\n{code}\n```\n\n```{lang}\n{code_impr}\n```\nFeedback from executing the code on visible test cases:\n\n{exe_feedback}")
        ]
    else: raise ValueError("task not in one of eval, tests, stop, compare")
    response = model.generate_chat(messages=messages, max_tokens=2048, temperature=temperature)
    result = parse_response(response)
    return result

def sample_n_random(items: List[str], n: int) -> List[str]:
    """Sample min(n, len(items)) random items from a list"""
    assert n >= 0
    if n >= len(items):
        return items
    return random.sample(items, n)
def generic_gen_strategy(
        func_sig: str,
        model: ModelBase,
        parse_strategy: Callable[[str], List[str]],
        code_combine: Callable,
        task = "strategy",
        given_strategy="",
        incorrect_code="",
        test_feedback="",
        temperature=0.0,
        lang="python",
        num_list = "3",
) -> List[str]:
    """Generates tests for a function."""
    if model.is_chat:
        if task == "strategy":
            system_prompt = PY_STRATEGY.replace("{py_strategy_k}", str(num_list))
            if "multiple" in str(num_list): system_prompt += Prompt_flexible
            # print_messages(system_prompt, func_sig)
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=func_sig)
            ]
        elif task == "implementation":
            system_prompt = PY_IMPLEMENT.replace("{py_strategy_k}", str(num_list)) #+ "\n"
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=f"```{lang}\n{func_sig}\n```\nHigh Level Strategy: {given_strategy}")
            ]
        elif task == "reflection":
            system_prompt = PY_REFELCTION.replace("{py_strategy_k}", str(num_list)) #+ "\n"
            if "multiple" in str(num_list): system_prompt += Prompt_flexible
            if given_strategy is None: given_strategy = ""
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content= f"[problem] {func_sig}\n\n[proposed strategy]{given_strategy}\n```{lang}\n{incorrect_code}\n```\n"),
                Message(role="user", content=f"[unit test results]:\n{test_feedback}")
            ]
        else: raise ValueError("Must be in one of strategy/reflection/implementation")
        print(messages_to_str(messages))
        func_bodies = model.generate_chat(messages=messages, max_tokens=2048, temperature=temperature)
        assert isinstance(func_bodies, str)
        func_body_str = parse_strategy(func_bodies) # how many strategies are given
        print_generated_func_body(func_body_str)
        return func_body_str
    else:
        raise ValueError("For chat models only.")
def print_messages(system_message_text: str, user_message_text: str) -> None:
    print(f"""----------------------- SYSTEM MESSAGE -----------------------)
{system_message_text}
----------------------------------------------
----------------------- USER MESSAGE -----------------------
{user_message_text}
----------------------------------------------
""", flush=True)

def print_generated_func_body(func_body_str: str) -> None:
    print(f"""--------------------- GENERATED FUNC BODY ---------------------
{func_body_str}
------------------------------------------""")

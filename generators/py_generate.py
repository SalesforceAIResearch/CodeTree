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

from .model import ModelBase, message_to_str
from .generator_types import Generator
from .generator_utils import generic_generate_func_impl, generic_generate_internal_tests, generic_generate_self_reflection, generate_with_accumulated_context, generic_gen_strategy, generic_evaluate
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from typing import Optional, List, Union
import ast
import re
from .parse import parse_code_block, add_code_block, parse_multiple_code_block, combine_function
from config import get_parsed_args
args = get_parsed_args()
Codecontests = False if args.function else True

PY_SIMPLE_COMPLETION_INSTRUCTION = "# Write the body of this function only."
PY_REFLEXION_COMPLETION_INSTRUCTION = "You are a Python writing assistant. You will be given your past function implementation, a series of unit tests, and a hint to change the implementation appropriately. Write your full implementation (restate the function signature).\n\n-----"
PY_SELF_REFLECTION_COMPLETION_INSTRUCTION = "You are a Python writing assistant. You will be given a function implementation and a series of unit tests. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation.\n\n-----"
USE_PYTHON_CODEBLOCK_INSTRUCTION = "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
PY_SIMPLE_CHAT_INSTRUCTION = "You are an AI that only responds with python code, NOT ENGLISH. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature). Don't include test cases or printing statements in the code block."
PY_SIMPLE_CHAT_INSTRUCTION_V2 = "You are an AI that only responds with only python code. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature)."
PY_REFLEXION_CHAT_INSTRUCTION = "You are an AI Python assistant. You will be given your past function implementation, a series of unit tests, and a hint to change the implementation appropriately. Write your full implementation (restate the function signature)."
PY_REFLEXION_CHAT_INSTRUCTION_V2 = "You are an AI Python assistant. You will be given your previous implementation of a function, a series of unit tests results, and your self-reflection on your previous implementation. Write your full implementation (restate the function signature)."
PY_SELF_REFLECTION_CHAT_INSTRUCTION = "You are a Python programming assistant. You will be given a function implementation and a series of unit tests. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation."
PY_SELF_REFLECTION_CHAT_INSTRUCTION_V2 = "You are a Python programming assistant. You will be given a function implementation and a series of unit test results. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as guidance when you try again later. Only provide the few sentence description in your answer, not the implementation. You will be given a few examples by the user."
PY_TEST_GENERATION_COMPLETION_INSTRUCTION = f"""You are an AI coding assistant that can write unique, diverse, and intuitive unit tests for functions given the signature and docstring."""
PY_TEST_GENERATION_CHAT_INSTRUCTION = """You are an AI coding assistant that can write unique, diverse, and intuitive unit tests for functions given the signature and docstring."""

if Codecontests:
    PY_REFLEXION_COMPLETION_INSTRUCTION = "You are a Python writing assistant. You will be given your past solution to a problem, a series of unit tests, and a hint to improve the solution appropriately. Write your full program(include read input/print output).\n\n-----"
    PY_SELF_REFLECTION_COMPLETION_INSTRUCTION = "You are a Python writing assistant. You will be given a solution to a problem and a series of unit tests. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation.\n\n-----"
    USE_PYTHON_CODEBLOCK_INSTRUCTION = "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
    PY_SIMPLE_CHAT_INSTRUCTION = "You are an AI that only responds with python code, NOT ENGLISH. You will be given a programming problem and its required input/output formats. Write your full implementation (include read input/print output; exclude test cases) in a code block."
    PY_REFLEXION_CHAT_INSTRUCTION = "You are an AI Python assistant. You will be given your past solution to a problem, a series of unit tests, and a hint to change the implementation appropriately. Write your full program(include read input/print output; exclude test cases).\n\n-----"


class PyGenerator(Generator):
    def self_reflection(self, func: str, feedback: str, model: ModelBase) -> str:
        return generic_generate_self_reflection(
            func=func,
            feedback=feedback,
            model=model,
            self_reflection_chat_instruction=PY_SELF_REFLECTION_CHAT_INSTRUCTION,
            self_reflection_completion_instruction=PY_SELF_REFLECTION_COMPLETION_INSTRUCTION,
            add_code_block=lambda x: add_code_block(x, "python"),
            self_reflection_few_shot=PY_SELF_REFLECTION_CHAT_INSTRUCTION
        )

    def func_impl(
        self,
        func_sig: str,
        model: ModelBase,
        strategy: str,
        prev_func_impl: Optional[str] = None,
        feedback: Optional[str] = None,
        self_reflection: Optional[str] = None,
        num_comps: int = 1,
        temperature: float = 0.8,
        acc_feedback: Optional[str] = None,
        acc_reflection: Optional[str] = None,
    ) -> Union[str, List[str]]:
        if strategy == "mcts":
            return generate_with_accumulated_context(
                func_sig=func_sig,
                model=model,
                strategy="reflexion",
                prev_func_impl=prev_func_impl,
                accumulated_feedback=acc_feedback,
                accumulated_reflection=acc_reflection,
                num_comps=num_comps,
                temperature=temperature,
                reflection_chat_instruction=PY_REFLEXION_CHAT_INSTRUCTION,
                reflection_few_shot=PY_REFLEXION_CHAT_INSTRUCTION,
                simple_chat_instruction=PY_SIMPLE_CHAT_INSTRUCTION,
                reflection_completion_instruction=PY_REFLEXION_COMPLETION_INSTRUCTION,
                simple_completion_instruction=PY_SIMPLE_COMPLETION_INSTRUCTION,
                code_block_instruction=USE_PYTHON_CODEBLOCK_INSTRUCTION,
                parse_code_block=lambda x: parse_code_block(x, "python"),
                add_code_block=lambda x: add_code_block(x, "python"),
            )
        else:
            return generic_generate_func_impl(
                func_sig=func_sig,
                model=model,
                strategy=strategy,
                prev_func_impl=prev_func_impl,
                feedback=feedback,
                self_reflection=self_reflection,
                num_comps=num_comps,
                temperature=temperature,
                reflection_chat_instruction=PY_REFLEXION_CHAT_INSTRUCTION,
                reflection_few_shot=PY_REFLEXION_CHAT_INSTRUCTION,
                simple_chat_instruction=PY_SIMPLE_CHAT_INSTRUCTION,
                reflection_completion_instruction=PY_REFLEXION_COMPLETION_INSTRUCTION,
                simple_completion_instruction=PY_SIMPLE_COMPLETION_INSTRUCTION,
                code_block_instruction=USE_PYTHON_CODEBLOCK_INSTRUCTION,
                parse_code_block=lambda x: parse_code_block(x, "python"),
                add_code_block=lambda x: add_code_block(x, "python"),
            )

    def internal_tests(self, func_sig: str, model: ModelBase, max_num_tests: int = 12) -> List[str]:
        def parse_tests(tests: str) -> List[str]:
            return [test.strip() for test in tests.splitlines() if "assert" in test]
        """
        Generates tests for a function.
        """
        return generic_generate_internal_tests(
            func_sig=func_sig,
            model=model,
            max_num_tests=max_num_tests,
            test_generation_few_shot=PY_TEST_GENERATION_CHAT_INSTRUCTION,
            test_generation_chat_instruction=PY_TEST_GENERATION_CHAT_INSTRUCTION,
            test_generation_completion_instruction=PY_TEST_GENERATION_COMPLETION_INSTRUCTION,
            parse_tests=parse_tests,
            is_syntax_valid=py_is_syntax_valid,
        )

    def strategy(self,
                 func_sig: str,
                 model: ModelBase,
                 num_strategy: int=3,
                 temperature: float = 0.0,
                 prev_func_impl: Optional[str] = None,
                 feedback: Optional[str] = None,
                 given_strategy: Optional[str] = None,
                 task: str="strategy") -> List[str]:
        def parse_strategy(strategies: str) -> List[str]:
            pattern = r"^\s*<\d+>(.*)"
            pattern2 = r"\d+\.(.*)"
            new_strategies = []
            lines = strategies.splitlines()
            lines = [ele for ele in lines if ele.strip() != '']
            for line in lines:
                if len(line) < 5: continue
                a = re.search(pattern, line.strip())
                if a: new_strategies.append(a.groups()[0])
                else:
                    a = re.search(pattern2, line.strip())
                    if a: new_strategies.append(a.groups()[0])
                    # else: new_strategies.append(line)
            return new_strategies
        return generic_gen_strategy(
            func_sig=func_sig,
            model=model,
            parse_strategy=parse_strategy,
            code_combine=combine_function,
            task=task,
            incorrect_code=prev_func_impl,
            test_feedback=feedback,
            given_strategy=given_strategy,
            num_list=num_strategy
        )

    def agent_eval(self,
                 func_sig: str,
                 model: ModelBase,
                 temperature: float = 0.0,
                 prev_func_impl: Optional[str] = None,
                 feedback: Optional[str] = None,
                 given_strategy: Optional[str] = None,
                 task: str="stop") -> List[str]:

        def binary_stop_parser(response):
            lines_of_response = response.splitlines()
            lines_of_response = [ele for ele in lines_of_response if ele.strip() != '']
            judge = True
            if "false" in lines_of_response[-1].lower(): judge=False
            elif "true" in lines_of_response[-1].lower(): judge=True
            else: 
                print("Sorry, this parse of judgement doesn't seem to work.")
                print("Response:", response)
            if len(lines_of_response) > 2: return judge, "\n".join(lines_of_response[:-1]) 
            return judge, lines_of_response[0]


        def test_eval_parser(response):
            response_last_line = response.splitlines()[-1]
            score = 0
            for ele in ["0","1","2", "3", "4", "5"]:
                if ele in response_last_line: score = int(ele)
            return score, "\n".join(response.splitlines()[:-1])

        return generic_evaluate(
            func_sig=func_sig,
            model=model,
            parse_response=binary_stop_parser if task=="stop" else test_eval_parser,
            task=task,
            code=prev_func_impl,
            exe_feedback=feedback,
            lang="python",
            code_impr=None
        )




DUMMY_FUNC_SIG = "def func():"
DUMMY_FUNC_CALL = "func()"


def handle_first_line_indent(func_body: str) -> str:
    if func_body.startswith("    "):
        return func_body
    split = func_body.splitlines()
    return f"    {split[0]}\n" + "\n".join(split[1:])


def handle_entire_body_indent(func_body: str) -> str:
    split = func_body.splitlines()
    res = "\n".join(["    " + line for line in split])
    return res


def fix_turbo_response(func_body: str) -> str:
    return fix_markdown(remove_unindented_signatures(func_body))


def fix_markdown(func_body: str) -> str:
    return re.sub("`{3}", "", func_body)


def remove_unindented_signatures(code: str) -> str:
    regex = r"^def\s+\w+\s*\("

    before_signature = []
    after_signature = []
    signature_found = False

    for line in code.split("\n"):
        if re.match(regex, line):
            signature_found = True
            continue

        if signature_found:
            after_signature.append(line)
        else:
            if not line.startswith("    ") and line.strip():
                line = "    " + line
            before_signature.append(line)

    return "\n".join(before_signature + after_signature)


def py_fix_indentation(func_body: str) -> str:
    func_body = fix_turbo_response(func_body)
    """
    3 cases:
        1. good syntax
        2. first line not good
        3. entire body not good
    """
    def parse_indent_rec(f_body: str, cur_state: int) -> str:
        f_body = fix_markdown(f_body)
        if cur_state > 1:
            return f_body
        code = f'{DUMMY_FUNC_SIG}\n{f_body}\n{DUMMY_FUNC_CALL}'
        try:
            exec(code)
            return f_body
        except (IndentationError, SyntaxError):
            p_func = handle_first_line_indent if cur_state == 0 else handle_entire_body_indent
            return parse_indent_rec(p_func(func_body), cur_state + 1)
        except Exception:
            return f_body
    return parse_indent_rec(func_body, 0)


def py_is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

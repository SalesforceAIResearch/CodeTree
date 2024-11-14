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

import ast
import signal
import astunparse
import subprocess
from .executor_utils import function_with_timeout
import sys
import random
import os
from typing import List
from .executor_types import ExecuteResult, Executor
def check_syntax_error(file_path):
    with open(file_path, 'r') as file:
        source_code = file.read()
        try:
            # Try to parse the source code.
            ast.parse(source_code)
            return False, None # without error
        except SyntaxError as e:
            # If a syntax error is found, report the error.
            print(f"SyntaxError in code: {e}")
            return True, e # with error
def check_syntax_error_source(source_code):
    try:
        # Try to parse the source code.
        ast.parse(source_code)
        return False, None # without error
    except SyntaxError as e:
        # If a syntax error is found, report the error.
        print(f"SyntaxError in file code: {e}")
        return True, e # with error

class PyExecutor(Executor):
    def execute(self, func: str, tests: List[str], timeout: int = 5, prev="") -> ExecuteResult:
        # Combine function code and assert statement
        error, syn = check_syntax_error_source(func)
        if error: return ExecuteResult(False, str(error), tuple([False]*len(tests)))
        imports = 'from typing import *\nfrom collections import *\nimport math\nimport sys\nimport random\nimport re\nimport itertools, functools, bisect, heapq, string\n'
        func_test_list = [f'{imports}\n{func}\n{test}\n' for test in tests]
        if prev: imports += f"{prev}\n"
        # Run the tests and collect the results
        success_tests = []
        failed_tests = []
        is_passing = True
        num_tests = len(func_test_list)
        for i in range(num_tests):
            try:
                pid = os.getpid()
                path = f"temp_files/temp_test{pid}.py"
                with open(path, "w") as temp_file:
                    temp_file.write(func_test_list[i])

                result = subprocess.run(
                    ['python', f'{path}'],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                if result.returncode == 0:
                    success_tests.append(tests[i])
                    continue
                else:
                    new_test_line = f'{imports}\n{func}\nprint({get_call_str(tests[i])})'

                    path = f"temp_files/temp_test_output{pid}.py"
                    with open(path, "w") as temp_file:
                        temp_file.write(new_test_line)

                    result = subprocess.run(
                        ['python', f'{path}'],
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )
                    output = result.stdout
                    error = result.stderr
                    if error: output = error + output

                    failed_tests += [f"{tests[i]} # incorrect program's output: {output}"]
                    is_passing = False
            except subprocess.TimeoutExpired:
                failed_tests.append(f"{tests[i]} # incorrect program's output: Program Timed Out after {timeout} seconds.")
                is_passing = False
            except Exception as e:

                failed_tests += [f"{tests[i]} # incorrect program's output: {e}"]
                is_passing = False

        state = []
        for test in tests:
            if test in success_tests:
                state += [True]
            else:
                state += [False]

        state = tuple(state)
        feedback = ""
        if success_tests: feedback += "Tested passed:"
        for test in success_tests:
            feedback += f"\n{test}"
        
        if failed_tests: feedback += "\nTests failed:"
        for test in failed_tests:
            feedback += f"\n{test}"
        if len(feedback) > 2500: feedback = feedback[:2500] # feedback length cut
        return ExecuteResult(is_passing, feedback, state)

    def evaluate(self, name: str, func: str, test: str, timeout: int = 5, prev="") -> bool:
        """
        Evaluates the implementation on Human-Eval Python.

        probably should be written in a dataset-agnostic way but not now
        """
        code = f"""from typing import *
from collections import *
import math, itertools, functools, bisect, heapq, string
import sys
import re
{prev}
{func}

{test}
"""
        try:
            pid = os.getpid()
            path = f"temp_files/temp_test{pid}.py"

            with open(path, "w") as temp_file:
                temp_file.write(code)
            result = subprocess.run(
                ['python', f'{path}'],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            if result.returncode == 0:
                return True

            return False
        except Exception as e:
            # print(code)
            # print("error during handling", e)
            return False

def get_call_str(assert_statement: str) -> str:
    ast_parsed = ast.parse(assert_statement)
    try:
        call_str = ast_parsed.body[0].test.left # type: ignore
    except:
        call_str = ast_parsed.body[0].test # type: ignore

    return astunparse.unparse(call_str).strip()

class PyExecutorPro():
    def execute(self, func: str, tests: List[dict], timeout: int = 5, prev="") -> ExecuteResult:
        num_tests = len(tests["input"])
        assert len(tests["output"]) == num_tests
        pid = os.getpid()
        path = f"temp_files/temp_test{pid}.py"
        with open(path, "w") as temp_file:
            temp_file.write(func)
        error, syn = check_syntax_error(path)
        if error: return ExecuteResult(False, str(error), tuple([False]*num_tests))
        success_tests = []
        failed_tests = []
        is_passing = True

        for i in range(num_tests):
            input_data = tests["input"][i]
            expected_output = tests["output"][i]
            try:
                result = subprocess.run([sys.executable, path], input=input_data, text=True, capture_output=True,
                                        check=True, timeout=timeout)
                code_output = result.stdout
                error_output = result.stderr
                if result.returncode == 0:
                    if code_output.strip() == expected_output.strip():
                        success_tests.append(f"Input:\n{input_data.strip()}\nOutput:\n{expected_output.strip()}")
                    else:
                        failed_tests.append(f"Input:\n{input_data.strip()}\nExpected Output:\n{expected_output.strip()}\nProgram's Output:\n{code_output.strip()}\n-----------------------------")
                        is_passing = False
                else:
                    is_passing = False
                    failed_tests.append(
                        f"Input:\n{input_data.strip()}\nExpected Output:\n{expected_output.strip()}\nProgram's Output:\n{code_output.strip()}\n-----------------------------")
            except subprocess.CalledProcessError as e:
                is_passing = False
                failed_tests.append(f"{input_data.strip()}\nProgram's Output: {e.stderr}")
            except subprocess.SubprocessError as e:
                failed_tests.append(f"{input_data.strip()}\nProgram's Output: {e}")
                is_passing = False
            except subprocess.TimeoutExpired:
                failed_tests.append(f"{input_data.strip()}\nProgram's Output: Program Timed Out after {timeout} seconds, could be read in format problem where program waiting on input.")
                is_passing = False
            except  Exception as e:
                failed_tests.append(f"{input_data.strip()}\nProgram's Output: {e}")
                is_passing = False
        state = []
        for test in tests:
            if test in success_tests:
                state += [True]
            else:
                state += [False]

        state = tuple(state)
        feedback = "Tests succeeded:"
        if success_tests:
            for test in success_tests:
                feedback += f"\n{test}"
        else: feedback += "\nNone"
        feedback += "\nTests failed:"
        if failed_tests:
            for test in failed_tests:
                feedback += f"\n{test}"
        else: feedback += "\nNone"
        if len(feedback) > 2500: feedback = feedback[:2500]
        return ExecuteResult(is_passing, feedback, state)
    
    def evaluate(self, name: str, func: str, test: List[dict], timeout: int = 5, prev="") -> bool:
        num_tests = len(test["input"])
        assert len(test["output"]) == num_tests
        pid = os.getpid()
        path = f"temp_files/temp_test{pid}.py"
        # path = "temp_files/temp_test.py"
        with open(path, "w") as temp_file:
            temp_file.write(func)
        error, syn = check_syntax_error(path)
        if error: return ExecuteResult(False, str(error), tuple([False]*num_tests))
        success_tests = []
        failed_tests = []
        is_passing = True

        for i in range(num_tests):
            input_data = test["input"][i]
            expected_output = test["output"][i]
            try:
                result = subprocess.run([sys.executable, path], input=input_data, text=True, capture_output=True,
                                        check=True, timeout=timeout)
                code_output = result.stdout
                if result.returncode == 0:
                    if code_output.strip() == expected_output.strip():
                        success_tests.append(f"Input:\n{input_data.strip()}\nOutput:\n{expected_output.strip()}")
                    else:
                        failed_tests.append(f"Input:\n{input_data.strip()}\nExpected Output:\n{expected_output.strip()}\nProgram's Output:\n{code_output.strip()}")
                        is_passing = False
                else: raise ValueError("Doesn't return output correctly")
            except Exception as e:
                failed_tests.append(f"{input_data.strip()}\nProgram's Output: {e}")
                is_passing = False
            except subprocess.TimeoutExpired:
                failed_tests.append(f"{input_data.strip()}\nProgram's Output: Program Timed Out after {timeout} seconds.")
                is_passing = False
            if not is_passing: break # already found failed case
        return is_passing



def get_output(func: str, assert_statement: str, timeout: int = 5) -> str:
    try:
        exec(f"from typing import *\n{func}", globals())
        func_call = get_call_str(assert_statement)
        output = function_with_timeout(eval, (func_call, globals()), timeout=timeout)
        return output
    except TimeoutError:
        return "TIMEOUT"
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    func1 = """
def add(a, b):
    return a + b
"""
    func = "def add(a, b):\n    while True:\n        x = 1\n    return a + b"
    tests = ["assert add(1, 2) == 3", "assert add(2, 2) == 3"]
    print(func)
    print(PyExecutor().execute(func, tests, timeout=1))

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

import sys


def combine_function(docstring: str, implementation: str) -> str:
    impl_lines = implementation.strip().split("\n")
    # Find the function definition line in the implementation
    func_def_line = None
    for i, line in enumerate(impl_lines):
        if line.strip().startswith("def "):
            func_def_line = i
            break
    if func_def_line is None:
        raise ValueError("Function definition not found in the implementation")
    impl_lines = docstring + "\n".join(impl_lines[func_def_line+1:])
    return impl_lines

if __name__ == "__main__":
    # Example usage
    docstring = '''
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
        """
    '''
    implementation = """def has_close_elements(numbers: List[float], threshold: float) -> bool:
    
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False
    """

    combined_function = combine_function(docstring, implementation)
    print(combined_function)
PY_IMPLEMENT = ("You are an AI assistant that help the user to write code. You will be given a function signature and its docstring by the user. The user would also suggest a strategy."
              "You should instruct the user how to implement it(in English not with Code), add details not provided in the strategy. You should give {py_strategy_k} distinct alternatives that are potentially correct on how to ground this idea."
              "e.g., if the idea was linear greedy search, it could be done through a forward or backward scan. Each alternative should be several sentences in one line. List and Number your implementation {py_strategy_k} alternatives line by line using \"1. \"; \"2. \" \"3. \" and so on.")
PY_REFELCTION = ("You are an AI assistant that provides reflection.  You will be given a function implementation and a series of unit tests. Your goal is to explain why the implementation is wrong as indicated by the tests, "
                "then point out a direction to fix the bug. You must provide 2 alternatives for the different possible bugs/fixes. List and number your alternatives line by line using \"1. \" and \"2. \". For each line, use a few sentences to analyze the issue from an angle, guess a possible bug, and suggest and describe how to fix it. Do not use new lines or list steps within each alternative.")
sys.stdout.write(PY_REFELCTION.replace("{py_strategy_k}","2"))

# USER:
# function signature:
# def say_hi() -> str:
#     """
#     Greet as a computer.
#     """
#
# strategy: output "hello world" is a good way to greet.
#
# MODEL:
# 1. use the bulit-in `print(string)` function in python
# 2. use `sys.stdout.write(string)` to directly write to standard output.
#
# But the model uses new lines and list step-by-step for a single implementation.




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
import textwrap
from copy import copy

def add_prev_funcs(prompts, cur_function, entry_point):
    all_functions = find_functions_with_implementation(prompts)
    if len(all_functions) == 1:
        return cur_function
    else:
        assert all_functions[-1]["func_name"] == entry_point  # last one is to be implemented
        return "\n".join(ele["func_code"] for ele in all_functions[:-1]) + f"\n{cur_function}"


def wrap_mbpp_data(dataset):
    list_of_data = []
    for i in range(len(dataset)):
        code = dataset[i]["code"]
        test_case_string = "\n".join(dataset[i]["test_list"])
        docstring = f"\"\"\"\n{dataset[i]['prompt']}\n\nExamples for reference:\n{test_case_string}\n\"\"\""
        lines = code.strip().splitlines()
        for line in reversed(lines):  # from last function definition
            if line[:3] == "def":
                header = line
                break
                # how_many_functions += 1

        comment = textwrap.indent(docstring, code.strip().splitlines()[-1].split("return")[0])
        new_point = dict(copy(dataset[i]))
        new_point["prompt"] = f"{header}\n{comment}"
        new_point["given_tests"] = copy(dataset[i]["test_list"])
        new_point["weaker_test"] = "\n".join(new_point["given_tests"])  # add lines to the end of program
        new_point["entry_point"] = header
        new_point["prev"] = ""
        list_of_data.append(new_point)
    print("mbpp question example:", list_of_data[0]["prompt"], sep="\n")
    return list_of_data

def gen_test_eval(exe, solution, test_cases, prev=""):
    is_passing, feedback, _ = exe.execute(solution, test_cases, timeout=1, prev=prev)
    if is_passing: reward = 1
    else:
        reward = _.count(True)/len(_)
    return is_passing, feedback, reward
def wrap_human_eval_data(dataset_loaded, dataset_evalplus):
    dataset_dict = {entry['task_id']: entry for entry in dataset_loaded}
    list_of_data = []
    for i in range(len(dataset_evalplus)):
        # print("processing")
        task_id = dataset_evalplus[i]['task_id']
        entry_point = dataset_evalplus[i]['entry_point']
        new_point = dict(copy(dataset_evalplus[i]))
        new_point["weaker_test"] = dataset_dict[task_id]['test'] + f"\ncheck({entry_point})\n"
        new_point["test"] = dataset_evalplus[i]["test"] + f"\ncheck({entry_point})\n"
        new_point["given_tests"] = copy(dataset_dict[task_id]['given_tests'])
        temp = extract_implemented_functions(
            dataset_evalplus[i]["prompt"])  # find_functions_with_implementation(dataset_evalplus[i]["prompt"])
        if temp:
            new_point["prev"] = f"\n{temp}\n"
            print("Multiple implementation!")
            print(temp)
        else:
            new_point["prev"] = ""
        list_of_data.append(new_point)

        # print(dataset_evalplus[i]["given_tests"])
    print(list_of_data[0]["weaker_test"])
    print(list_of_data[3]["prompt"])
    # print(list_of_data[0]["test"])
    return list_of_data  # dataset_evalplus
def has_docstring(func_node):
    if func_node.body and isinstance(func_node.body[0], ast.Expr):
        expr = func_node.body[0].value
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return True
    return False

def contains_return_with_value(func_node):
    for node in ast.walk(func_node):
        if isinstance(node, ast.Return):
            if node.value is not None:
                # Optionally, ensure the return value is not None
                if not (isinstance(node.value, ast.Constant) and node.value.value is None):
                    return True
    return False

def is_effective_function(func_node):
    if not func_node.body:
        return False

    # Exclude functions that only contain 'pass' or 'Ellipsis'
    for stmt in func_node.body:
        if isinstance(stmt, ast.Pass):
            return False
        if isinstance(stmt, ast.Expr):
            expr = stmt.value
            if isinstance(expr, ast.Constant) and expr.value == Ellipsis:
                return False

    # Check for return statements with values
    if contains_return_with_value(func_node):
        return True

    return False

def find_functions_with_implementation(source_code):
    node = ast.parse(source_code)
    functions_info = []

    for n in ast.walk(node):
        if isinstance(n, ast.FunctionDef):
            functions_info.append({
                "func_name": n.name,
                "func_code": ast.get_source_segment(source_code, n),
                "implemented": is_effective_function(n),
                "has_docstring": has_docstring(n)
            })

    return functions_info

def extract_implemented_functions(source_code):
    functions_info = find_functions_with_implementation(source_code)
    try:
        implemented_funcs = [func['func_code'] for func in functions_info if func['implemented']]
    except:
        print(functions_info)
    return "\n\n".join(implemented_funcs)

def cal_metrics(decisions):
    # Initialize counts for TP, TN, FP, and FN
    TP = TN = FP = FN = 0

    # Iterate through decisions and count each case
    for predict, label in decisions:
        if predict == 1 and label == 1:
            TP += 1
        elif predict == 0 and label == 0:
            TN += 1
        elif predict == 1 and label == 0:
            FP += 1
        elif predict == 0 and label == 1:
            FN += 1

    # Calculate accuracy
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0

    # Return the metrics as a dictionary
    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'accuracy': accuracy
    }


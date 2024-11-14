import re
from typing import Optional


def parse_code_block(string: str, lang: str) -> Optional[str]:
    code = None
    code_pattern = fr"```{lang}\n(.*?)\n```"
    match = re.search(code_pattern, string, re.DOTALL)
    if match:
        code = match.group(1)
    if code is None:
        generic_code_pattern = r"```\n(.*?)\n```"
        match = re.search(generic_code_pattern, string, re.DOTALL)
        if match:
            code = match.group(1)
    if code is None: return parse_first_func(string, lang)
    # Remove unit tests written by itself
    return code
    # def_appear = False
    # new_code_lines = []
    # for code_line in code.splitlines():
    #     if code_line.startswith("\t") or code_line.startswith(" ") or "import" in code_line:
    #         new_code_lines.append(code_line)
    #     elif "def" in code_line and ":" in code_line:
    #         new_code_lines.append(code_line)
    #         def_appear = True
    #     elif def_appear is False: # before fisrt function appears, you can write some constant definitions.
    #         new_code_lines.append(code_line)
    # return "\n".join(new_code_lines)



def combine_function(docstring: str, implementation: str) -> str:
    impl_lines = implementation.strip().split("\n")
    if docstring.count("def") < 1: return None
    if docstring.count("def") > 1 or implementation.count("def") > 1:
        print("Error, many functions found.")
        return None
    # Find the function definition line in the implementation
    func_def_line = None
    for i, line in enumerate(impl_lines):
        if line.strip().startswith("def "):
            func_def_line = i
            break
    if func_def_line is None:
        return None
    impl_lines = docstring + "\n".join(impl_lines[func_def_line+1:])
    return impl_lines

def parse_multiple_code_block(string: str, lang: str) -> Optional[str]:
    list_of_code = []
    for code_pattern in [fr"```{lang}\n(.*?)\n```", r"```\n(.*?)\n```"]:
        if re.search(code_pattern, string, re.DOTALL):
            matches = re.finditer(code_pattern, string, re.DOTALL)
            for match in matches:
                list_of_code.append(match.group(1))
            return list_of_code
    return parse_first_func(string, lang)


def parse_first_func(code: str, lang: str) -> Optional[str]:
    assert lang == "python", "Only python is supported for now. TODO: Rust"
    code_lines = code.split("\n")
    def_i = -1
    last_i = 0
    got_return = False
    for i, line in enumerate(code_lines):
        if line.startswith("def "):
            if def_i == -1:
                def_i = i
            else:
                break
        elif "return" in line and def_i != -1:
            got_return = True
        if line == "" and def_i != -1 and got_return:
            last_i = i
            break

    if last_i == 0:
        last_i = len(code_lines) - 1

    if def_i == -1:
        return ""

    return "\n".join(code_lines[def_i:last_i+1]).rstrip("[/PYTHON]")


def add_code_block(string: str, lang: str) -> str:
    return f"```{lang}\n{string}\n```"


def parse_functions_and_imports(code: str, lang: str) -> Optional[str]:
    assert lang == "python", "Only python is supported for now. TODO: Rust"
    code_lines = code.split("\n")
    filtered_lines = []
    inside_function_or_class = False
    inside_triple_quotes = False

    for line in code_lines:
        stripped_line = line.strip()
        if stripped_line.startswith("import ") or stripped_line.startswith("from "):
            filtered_lines.append(line)
        elif stripped_line.startswith("def ") or stripped_line.startswith("class "):
            inside_function_or_class = True
            filtered_lines.append(line)
        elif inside_function_or_class:
            filtered_lines.append(line)
            if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                inside_triple_quotes = not inside_triple_quotes
            if not inside_triple_quotes and not stripped_line.startswith("    "):
                inside_function_or_class = False
        elif inside_triple_quotes:
            filtered_lines.append(line)
            if stripped_line.endswith('"""') or stripped_line.endswith("'''"):
                inside_triple_quotes = False

    # Remove lines that should be excluded
    result = [line for line in filtered_lines if
              line.strip() and not line.strip().startswith("[") and not line.strip().startswith("my_wonderful_func()")]

    return "\n".join(result)


if __name__ == "__main__":
    CODE = """
import collections
a = 1
b = 2
sub_parser = parser.add_subparsers().add_parser("frf
a")

def my_wonderful_func():
    def useless_helper():
        return 1
    if 1:
        return 1
    else:
        return (
            1,
            2,
        )
    [1,2,3,4,5]

def bleh():
    return aaa
"""
    #print(parse_code_block(CODE, "python"))
    print(parse_functions_and_imports(CODE, "python"))
    CODE = """def total_match(lst1: List[str], lst2: List[str]) -> List[str]:
    \"\"\"
    Write a function that accepts two lists of strings and returns the list that has
    total number of chars in the all strings of the list less than the other list.
    
    if the two lists have the same number of chars, return the first list.
    
    Examples
    >>> total_match([], [])
    []
    >>> total_match(['hi', 'admin'], ['hI', 'Hi'])
    ['hI', 'Hi']
    >>> total_match(['hi', 'admin'], ['hi', 'hi', 'admin', 'project'])
    ['hi', 'admin']
    >>> total_match(['hi', 'admin'], ['hI', 'hi', 'hi'])
    ['hI', 'hi', 'hi']
    >>> total_match(['4'], ['1', '2', '3', '4', '5'])
    ['4']
    \"\"\"
    total_chars_lst1 = sum(len(word) for word in lst1)
    total_chars_lst2 = sum(len(word) for word in lst2)
    
    if total_chars_lst1 < total_chars_lst2:
        return lst1
    elif total_chars_lst1 > total_chars_lst2:
        return lst2
    else:
        return lst1
    """
    #print(parse_code_block(CODE, "python"))
    print(parse_functions_and_imports(CODE, "python"))

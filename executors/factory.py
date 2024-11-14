from .py_executor import PyExecutor, PyExecutorPro
from .executor_types import Executor

def executor_factory(lang: str, is_leet: bool = False) -> Executor:
    if lang == "code_contests": return PyExecutorPro()
    if lang == "py" or lang == "python":
            return PyExecutor()
    else:
        raise ValueError(f"Invalid language for executor: {lang}")

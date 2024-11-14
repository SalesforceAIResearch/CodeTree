#from ..main import Codecontests
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from .py_generate import PyGenerator # TODO: change source back
from .generator_types import Generator
from .model import myLLM, CodeLlama, ModelBase, GPT4, GPT35, StarChat, GPTDavinci, GPT4Omini, GPT4O


def generator_factory(lang: str) -> Generator:
    if lang == "py" or lang == "python":
        return PyGenerator()
    else:
        raise ValueError(f"Invalid language for generator: {lang}")


def model_factory(model_name: str) -> ModelBase:
    if model_name == "gpt-4":
        return GPT4()
    elif "gpt-3.5-turbo" in model_name:
        return GPT35()
    elif "gpt-4o-mini" in model_name:
        return GPT4Omini()
    elif "gpt-4o" in model_name: return GPT4O()
    elif "meta-llama" in model_name: return myLLM(model_name)
    elif model_name == "starchat":
        return StarChat()
    elif model_name.startswith("codellama"):
        # if it has `-` in the name, version was specified
        kwargs = {}
        if "-" in model_name:
            kwargs["version"] = model_name.split("-")[1]
        return CodeLlama(**kwargs)
    elif model_name.startswith("text-davinci"):
        return GPTDavinci(model_name)
    else:
        return myLLM(model_name)
        #raise ValueError(f"Invalid model name: {model_name}")

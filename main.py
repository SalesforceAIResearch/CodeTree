import os
import argparse
from reflexion import run_reflexion
from utils import read_jsonl, read_jsonl_gz, read_json
from bfs import run_bfs
from dfs_real import run_dfs as my_dfs
from strategy import strategy_guide
from llm_agent_guide import agent_guide
from resample_baseline import resample
from common import wrap_mbpp_data, wrap_human_eval_data
from datasets import load_dataset
from config import get_parsed_args

def strategy_factory(strategy: str):
    def kwargs_wrapper_gen(func, delete_keys=[]):
        def kwargs_wrapper(**kwargs):
            for key in delete_keys:
                del kwargs[key]
            return func(**kwargs)
        return kwargs_wrapper

    if strategy == "reflexion":
        return kwargs_wrapper_gen(run_reflexion)
    elif strategy == "dfs":
        return kwargs_wrapper_gen(my_dfs)
    elif strategy == "bfs":
        return kwargs_wrapper_gen(run_bfs)
    elif strategy == "strategy":
        return kwargs_wrapper_gen(strategy_guide)
    elif strategy == "agent":
        return kwargs_wrapper_gen(agent_guide)
    elif strategy == "resample":
        return kwargs_wrapper_gen(resample)
    else:
        raise ValueError(f"Strategy `{strategy}` is not supported")


def main(args):
    # check if the root dir exists and create it if not
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)
    dataset_name = os.path.basename(args.dataset_path).replace("jsonl", "")
    log_dir = os.path.join(args.root_dir, args.run_name)
    log_path = os.path.join(
        log_dir, f"{dataset_name}_{args.strategy}_{args.max_iters}_{args.model}_pass_at_k_{args.pass_at_k}_{args.language}.jsonl")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # check if the strategy is valid
    run_strategy = strategy_factory(args.strategy)

    # print starting message
    if args.verbose:
        print(f"""
Starting run with the following parameters:
strategy: {args.strategy}
pass@k: {args.pass_at_k}
""")
    else:
        print(f"Logs will be saved in `{log_dir}`")

    # load the dataset
    print(f'Loading the dataset...')
    if args.dataset_path.endswith(".json"):
        dataset = read_json(args.dataset_path)
    elif args.dataset_path.endswith(".jsonl"):
        dataset = read_jsonl(args.dataset_path)
    elif args.dataset_path.endswith(".jsonl.gz"):
        dataset = read_jsonl_gz(args.dataset_path)
    else:
        raise ValueError(
            f"Dataset path `{args.dataset_path}` is not supported")
    print(f"Loaded {len(dataset)} examples")
    if "mbpp" in args.dataset_path:
       dataset = load_dataset("evalplus/mbppplus")
       dataset = wrap_mbpp_data(dataset["test"]) # half-half
    if "humaneval" in args.dataset_path:
       new_dataset = load_dataset("evalplus/humanevalplus")
       dataset = wrap_human_eval_data(dataset, new_dataset["test"])
    Codecontests = False if args.function else True
    run_strategy(
        dataset=dataset,
        model_name=args.model,
        language=args.language,
        max_iters=args.max_iters,
        log_path=log_path,
        verbose=args.verbose,
        Codecontests=Codecontests
    )

    print(f"Done! Check out the logs in `{log_path}`")

if __name__ == "__main__":
    args = get_parsed_args()
    main(args)




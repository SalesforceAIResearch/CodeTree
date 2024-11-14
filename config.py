import argparse
_args=None
def get_parsed_args():
    global _args
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, help="The name of the run")
    parser.add_argument("--root_dir", type=str,
                        help="The root logging directory", default="root")
    parser.add_argument("--dataset_path", type=str,
                        help="The path to the benchmark dataset", default="root")
    parser.add_argument("--language", type=str, help=" `py` only")
    parser.add_argument("--model", type=str, help="GPT models, and LLaMA 3.1 models")
    parser.add_argument("--pass_at_k", type=int,
                        help="Pass@k metric, only implemented pass@1", default=1)
    parser.add_argument("--max_iters", type=int,
                        help="The maximum number of total tries in code implementation(budget)", default=10)
    parser.add_argument("--max_depth", type=int)
    parser.add_argument("--strategy", type=str, help="run methods, [reflexion, dfs, bfs, agent, resample, strategy]")
    parser.add_argument("--search_width", type=int)
    parser.add_argument("--verbose", action='store_true', help="To print live logs")
    parser.add_argument("--function", action='store_true',
                        help="if it's function implementation task or a program implementation task, codecontests=False, mbpp/humaneval=True")
    _args = parser.parse_args()
    return _args
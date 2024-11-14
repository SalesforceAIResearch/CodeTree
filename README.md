# Repo of CodeTree: Agent-guided Tree Search for Code Generation with Large Language Models

This is the repo of the paper: [CodeTree: Agent-guided Tree Search for Code Generation with Large Language Models](https://arxiv.org/abs/2411.04329)

## Run Scripts

To run the repo, you first need to install requirements.

```
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` environment variable to your OpenAI API key if you want to use openai methods:

```
export OPENAI_API_KEY=<your key>
```

Then run the scripts for the full method: 

```
bash `agent_humaneval.sh`
```

## Details

We currently support the following options as the `strategy` argument ( corresponding to the paper):

* `agent`: Full CodeTree method
* `bfs`: CodeTree-BFS
* `dfs`: CodeTree-DFS
* `reflexion`: Reflexion
* `resample`: Resample	

We currently support the following models as the `model` argument ( corresponding to the paper):

* `GPT-4o-mini`: gpt-4o-mini-2024-07-18 
* `GPT-4o`: gpt-4o-2024-08-06
* `GPT-3.5-turbo`: GPT-3.5-turbo (outdated and not recommended)
* `GPT-4`: GPT-4 (outdated and not recommended)
* `Llama-3.1-8B-Instruct`: meta-llama/Llama-3.1-8B-Instruct

Datasets are in `CodeTree/data/`


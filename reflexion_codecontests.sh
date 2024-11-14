# export OPENAI_API_KEY="" # input your openai key if not already
python main.py \
  --run_name "code_4o-mini-reflexion" \
  --root_dir "root" \
  --dataset_path data/code_contests_test.json \
  --strategy "reflexion" \
  --language "py" \
  --model "gpt-4o-mini" \
  --pass_at_k "1" \
  --max_iters 20 \
  --verbose
## Installation
```bash
pip install -r requirements.txt
```

## Run HF Server

```bash
uvicorn hf_server:app --host 0.0.0.0 --port 3000
```

## Run vLLM server

```bash
python -m vllm.entrypoints.openai.api_server     --model meta-llama/Llama-3.1-8B-Instruct   --host 0.0.0.0     --port 3000     --tensor-parallel-size 1   --max-model-len 49152   --max-num-batched-tokens 49152     --max-num-seqs 16 --dtype float16
```
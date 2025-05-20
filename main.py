from llama_cpp import Llama

import torch
from transformers import AutoTokenizer, AutoModel
from agent import Agent

llm = Llama(
    model_path="/home/krasniuk-ai/qwen-agent/Qwen_Qwen3-8B-Q4_K_M.gguf",
    n_ctx=1024,
    # 2500MB, mb can do even more,
    n_gpu_layers=14,
    offload_kqv=True,
)

path = 'Alibaba-NLP/gte-base-en-v1.5'
device = torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModel.from_pretrained(
    path,
    trust_remote_code=True,
    unpad_inputs=True,
    use_memory_efficient_attention=True,
    torch_dtype=torch.float16
).to(device)


SYSTEM_PROMPT = """Generate ONLY this JSON structure ONLY when query is related for news searches, otherwise dont use it:
{
  "name": "news_api_search",
  "arguments": {
    "query": "optimized_search_query",
    "language": "en",
    "max_results": 3
  }
}"""


ags = Agent(llm=llm, max_steps=2)

query = "give me the latest insights from crypto's and stock's world"

ags.call(query)
# ==============================================================================
#  Windows Performance Benchmark — Phi-3-mini-4k-instruct
#  Compares F16 vs Q4_K_M on 100 prompts
# ==============================================================================

import time
import os
import pandas as pd
from llama_cpp import Llama

# ================= CONFIG =================

QUANTIZATION   = "Q4_K_M"   # Change to "Q4_K_M" for second run
MAX_NEW_TOKENS = 200
N_GPU_LAYERS   = 0       # Keep 0 for Windows CPU
N_CTX          = 4096

MODEL_PATHS = {
    "F16":     "./Phi-3-mini-4k-instruct-fp16.gguf",
    "Q4_K_M":  "./Phi-3-mini-4k-instruct-q4.gguf",
}

PROMPTS_CSV = "./laptop_100_prompts.csv"
OUTPUT_FILE = f"phi3_{QUANTIZATION}_windows_100prompts.csv"

# ===========================================

print("=" * 60)
print(f"Phi-3 Windows Benchmark")
print(f"Quantization : {QUANTIZATION}")
print("=" * 60)

# Load model
model_path = MODEL_PATHS[QUANTIZATION]
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

print(f"\nLoading model: {model_path}")
llm = Llama(
    model_path=model_path,
    n_ctx=N_CTX,
    n_gpu_layers=N_GPU_LAYERS,
    verbose=False,
)
print("Model loaded.\n")

# Load prompts
prompts_df = pd.read_csv(PROMPTS_CSV)
assert len(prompts_df) == 100, "Expected 100 prompts"

def format_prompt(text):
    return f"<|user|>\n{text}<|end|>\n<|assistant|>\n"

results = []

print(f"{'ID':>4} {'Category':<16} {'Latency':>8} {'Tok/s':>8}")
print("-" * 45)

for _, row in prompts_df.iterrows():
    task_id  = int(row["ID"])
    category = row["CATEGORY"]
    prompt   = row["PROMPT"]

    formatted = format_prompt(prompt)

    t0 = time.time()
    output = llm(
        formatted,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        echo=False,
    )
    latency = time.time() - t0

    response_text  = output["choices"][0]["text"].strip()
    tokens_out     = output["usage"]["completion_tokens"]
    tokens_in      = output["usage"]["prompt_tokens"]
    tokens_per_sec = tokens_out / latency if latency > 0 else 0

    results.append({
        "ID": task_id,
        "Precision": QUANTIZATION,
        "Category": category,
        "Prompt": prompt,
        "Response": response_text,
        "Input_Tokens": tokens_in,
        "Output_Tokens": tokens_out,
        "Latency_s": round(latency, 4),
        "Tokens_per_sec": round(tokens_per_sec, 2),
        "Platform": "Windows_IrisXe_CPU",
    })

    print(f"{task_id:>4} {category:<16} {latency:>7.2f}s {tokens_per_sec:>7.1f}")

df = pd.DataFrame(results)

print("\n" + "=" * 60)
print(f"RESULTS — {QUANTIZATION}")
print("=" * 60)
print(f"Avg Latency    : {df.Latency_s.mean():.2f}s")
print(f"Avg Tokens/sec : {df.Tokens_per_sec.mean():.2f}")
print("=" * 60)

print("\nPer-category breakdown:")
summary = df.groupby("Category").agg(
    avg_latency_s    = ("Latency_s", "mean"),
    avg_tokens_per_s = ("Tokens_per_sec", "mean"),
).round(2)

print(summary.to_string())

df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved: {OUTPUT_FILE}")
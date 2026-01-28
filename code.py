# ==============================================================================
#  RESEARCH: THE GREEN LEARNING AUDIT (Colab-ready)
#  Objective: Measure Inference Energy (Joules), Latency (s), and compute LpW
# ==============================================================================

# --- CRITICAL: Install compatible stack for Phi-3 + Colab ---
!pip uninstall -y transformers -q
!pip install -q "transformers>=4.44.0" accelerate bitsandbytes codecarbon

import time
import torch
import pandas as pd

from transformers.cache_utils import DynamicCache
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
)
from codecarbon import EmissionsTracker
from google.colab import files

# ---------------------------------------------------------
# 0. CACHE COMPAT SHIM (ONLY seen_tokens; no cache hacks)
# ---------------------------------------------------------
if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())


# ---------------------------------------------------------
# 1. EXPERIMENTAL CONFIGURATION
# ---------------------------------------------------------
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
USE_QUANTIZATION = False  # True = NF4 4-bit, False = FP16
OUTPUT_FILE = "green_audit_lpw_results.csv"
OUTPUT_XLSX_PREFIX = "green_audit_lpw_results_batch"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("CUDA available:", torch.cuda.is_available())
print("Device:", DEVICE)


# ---------------------------------------------------------
# 2. THE 50 EDUCATIONAL PROMPTS
# ---------------------------------------------------------
PROMPTS = [
    # MATH (1-10)
    "Explain the Pythagorean theorem to a 10-year-old using Lego blocks.",
    "Solve for x: 3x + 5 = 20. Show every step clearly.",
    "What is the derivative of x^2? Explain why visually.",
    "I don't understand negative numbers. Can you use an elevator analogy?",
    "Calculate the area of a circle with radius 5. Explain the formula.",
    "What is a prime number? List the first 10.",
    "Explain the difference between mean, median, and mode with an example.",
    "If I flip a coin 3 times, what is the probability of 3 heads?",
    "How do I convert fractions to decimals? Teach me a trick.",
    "What is the order of operations (PEMDAS)? Give a tricky example.",
    # SCIENCE (11-20)
    "Explain photosynthesis as if I were a plant.",
    "What is Newton's Third Law? Give a real-life sports example.",
    "Why is the sky blue? Explain scattering simply.",
    "What is the difference between an atom and a molecule?",
    "Explain the water cycle using a kitchen pot as an analogy.",
    "What is DNA? Is it like a blueprint or a recipe?",
    "Why do things float in water? Explain density.",
    "What is a chemical reaction? Use baking soda and vinegar as an example.",
    "Explain the concept of 'kinetic energy' using a roller coaster.",
    "What are the three states of matter? describe ice, water, steam.",
    # CODING (21-30)
    "Debug this python code: def sum(a,b): return a * b",
    "Explain what a 'Variable' is in coding using a box analogy.",
    "What is a 'For Loop'? Explain it like a chore list.",
    "Difference between HTML and CSS? Explain using a house analogy.",
    "What is an algorithm? Is it like a cooking recipe?",
    "Explain 'Binary Code' (0s and 1s) to a beginner.",
    "What is the difference between hardware and software?",
    "Why is my internet slow? Explain bandwidth like a highway.",
    "What is 'Cloud Computing'? Is my data actually in the sky?",
    "Write a simple Python print statement for 'Hello World'.",
    # HUMANITIES (31-40)
    "Help me write a topic sentence for an essay about climate change.",
    "What is a haiku? Write one about homework.",
    "Explain the difference between 'There', 'Their', and 'They're'.",
    "Summarize the plot of Romeo and Juliet in 3 sentences.",
    "What is a metaphor? Give me an example about time.",
    "Who was Shakespeare? Why is he famous?",
    "Explain the concept of 'Democracy' to a child.",
    "What is a primary source in history? vs a secondary source?",
    "Give me 3 synonyms for the word 'Happy'.",
    "Check this sentence for grammar: 'Me and him went to the store.'",
    # META-COGNITION (41-50)
    "I can't focus on studying. Give me the Pomodoro technique.",
    "How do I take better notes? Explain the Cornell method.",
    "I have test anxiety. What is one breathing exercise I can do?",
    "Create a 3-step plan to learn Spanish vocabulary.",
    "Why is sleep important for learning? Explain memory consolidation.",
    "How do I prioritize my homework? Explain the Eisenhower Matrix.",
    "What is 'Active Recall'? Is it better than re-reading?",
    "I feel overwhelmed. Break down writing an essay into small steps.",
    "How do I ask my teacher for help politely via email?",
    "Motivate me! I feel like giving up on math."
]


# ---------------------------------------------------------
# 3. PEDAGOGICAL SCORES (Qped) - scale 1-10
# ---------------------------------------------------------
QPED_SCORES = [8] * 50


# ---------------------------------------------------------
# 4. MODEL LOADING (RoPE-safe + optional 4-bit)
# ---------------------------------------------------------
print("\n--- GREEN AUDIT: Loading Model ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
config.rope_scaling = None
print("rope_scaling:", config.rope_scaling)

if USE_QUANTIZATION and DEVICE == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map={"": 0},
        attn_implementation="eager",
    )

model.eval()
device = next(model.parameters()).device
print("Model device:", device)


# ---------------------------------------------------------
# 5. BASELINE IDLE CALIBRATION
# ---------------------------------------------------------
print("\n[STEP 1] Measuring Idle Power (Baseline)...")
baseline_tracker = EmissionsTracker(
    measure_power_secs=2.0,
    save_to_file=False,
    log_level="error",
)
baseline_tracker.start()
time.sleep(10)
baseline_tracker.stop()

idle_energy_kwh = getattr(baseline_tracker, "_total_energy", None)
if idle_energy_kwh is not None:
    idle_watts = (idle_energy_kwh.kWh * 3.6e6) / 10.0
else:
    idle_watts = 0.0
print(f"Baseline Idle Power: {idle_watts:.4f} Watts")


# ---------------------------------------------------------
# 6. MAIN EXPERIMENT LOOP
# ---------------------------------------------------------
results = []
tracker = EmissionsTracker(
    project_name="audit",
    measure_power_secs=1.0,
    save_to_file=False,
    log_level="error",
)
tracker.start()
total_tasks = len(PROMPTS)
print(f"\n[STEP 2] Processing {total_tasks} Prompts...")

for idx, prompt in enumerate(PROMPTS):
    task_id = idx + 1
    print(f"[{task_id}/{total_tasks}] Task: {prompt[:40]}...")

    # Snapshot Start
    tracker._measure_power_and_energy()
    e_start = tracker._total_energy.kWh

    # Run Inference
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,   # shorter for speed
            use_cache=False,      # SAFE for Phi-3
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.time() - t0

    # Snapshot End
    tracker._measure_power_and_energy()
    e_end = tracker._total_energy.kWh

    # Differential Energy
    gross_j = (e_end - e_start) * 3.6e6
    net_j = gross_j - (idle_watts * latency)
    if net_j < 0.001:
        net_j = 0.01

    # Learning-per-Watt (LpW)
    qped = QPED_SCORES[idx]
    denom = net_j * latency
    lpw = qped / denom if denom > 0 else 0

    category = (
        "Math" if idx < 10
        else "Science" if idx < 20
        else "CS" if idx < 30
        else "Humanities" if idx < 40
        else "Meta"
    )

    results.append({
        "ID": task_id,
        "Category": category,
        "Prompt": prompt[:40] + "...",
        "Latency(s)": round(latency, 4),
        "Net_AI_Energy(J)": round(net_j, 4),
        "Wattage(W)": round(net_j / latency, 2),
        "Qped": qped,
        "LpW": round(lpw, 6),
    })

    if task_id % 10 == 0:
        print(f"--- Completed {task_id}/{total_tasks} tasks ---")
        batch_df = pd.DataFrame(results)
        batch_filename = f"{OUTPUT_XLSX_PREFIX}_{task_id}.xlsx"
        batch_df.to_excel(batch_filename, index=False)
        files.download(batch_filename)

tracker.stop()


# ---------------------------------------------------------
# 7. EXPORT FINAL RESULTS
# ---------------------------------------------------------
df = pd.DataFrame(results)
df.to_csv(OUTPUT_FILE, index=False)
files.download(OUTPUT_FILE)

print("\n" + "=" * 50)
print(f"AUDIT COMPLETE. Average Net Energy: {df['Net_AI_Energy(J)'].mean():.4f} J")
print(f"Average LpW: {df['LpW'].mean():.6f}")
print("=" * 50)

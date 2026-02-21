# ==============================================================================
#  KV-CACHE ON EXPERIMENT — Appendix C (50 PROMPTS, KV-CACHE=TRUE)
#  Phi-3-mini-4k-instruct (CLEAN + CORRECT)
# ==============================================================================

!pip uninstall -y transformers -q
!pip install -q "transformers>=4.44.0" accelerate bitsandbytes codecarbon

import time
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from codecarbon import EmissionsTracker
from google.colab import files

# -------------------------
# CONFIG
# -------------------------
MODEL_ID       = "microsoft/Phi-3-mini-4k-instruct"
MAX_NEW_TOKENS = 200
OUTPUT_FILE    = "kvcache_50prompts_true.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

# -------------------------
# PROMPTS (50 total)
# -------------------------
PROMPTS = [
    # Mathematics (10)
    "Explain how to solve a quadratic equation using the quadratic formula, and walk through the steps for solving 2x² + 5x - 3 = 0.",
    "What is the difference between a function and a relation? Give an example of each.",
    "Explain what the slope of a line represents and how to calculate it from two points.",
    "How do you factor a trinomial of the form ax² + bx + c? Show the process with an example.",
    "What are exponent rules? Explain the product rule, quotient rule, and power rule with examples.",
    "Explain what a logarithm is and how it relates to exponentiation. What is log₂(8)?",
    "What is the difference between linear and exponential growth? Give a real-world example of each.",
    "How do you solve a system of two linear equations using substitution? Walk me through an example.",
    "Explain what absolute value means geometrically on a number line.",
    "What is the binomial theorem and how would you expand (x + y)³?",

    # Science (10)
    "Explain what photosynthesis is, what inputs it needs, and what it produces.",
    "What is cellular respiration and how does it differ from breathing?",
    "Explain the structure of DNA and what base pairing rules apply.",
    "What is mitosis and why do cells need to divide?",
    "Explain what meiosis is and how it differs from mitosis.",
    "What is natural selection and how does it lead to evolution over time?",
    "Explain the difference between an ecosystem and a biome.",
    "What is the role of the cell membrane and what does 'selectively permeable' mean?",
    "Explain the difference between prokaryotic and eukaryotic cells.",
    "What is the function of the mitochondria and why is it called the powerhouse of the cell?",

    # Programming-CS (10)
    "What is a variable in programming? Explain it like I'm 10 years old using a box analogy.",
    "What is a 'For Loop'? Explain it like a chore list.",
    "Explain what an 'if statement' is in programming using a traffic light analogy.",
    "What is a function in programming and why do we use them?",
    "Explain the difference between a while loop and a for loop.",
    "What is a list (or array) in programming and how do you access elements in it?",
    "Explain what a bug is in programming and give an example of a common one.",
    "What does it mean to 'compile' a program?",
    "Explain what recursion is with the example of calculating a factorial.",
    "What is the difference between a syntax error and a runtime error?",

    # Humanities (10)
    "Explain the main causes of World War I using the MAIN acronym.",
    "What was the significance of the French Revolution and its key outcomes?",
    "Explain what the Industrial Revolution was and how it changed society.",
    "What was the Cold War and the main tensions between the USA and USSR?",
    "Explain the causes and consequences of the Atlantic Slave Trade.",
    "What was the significance of the Magna Carta?",
    "Explain what colonialism is and how it affected Africa and Asia.",
    "What were the main causes of World War II?",
    "Explain the importance of the Renaissance period.",
    "What was the significance of the Civil Rights Movement in the USA?",

    # Meta-cognition (10)
    "Explain what metacognition means and why it helps learning.",
    "What is spaced repetition and why is it effective?",
    "Explain the Pomodoro technique.",
    "What is active recall?",
    "Explain interleaving in studying.",
    "Difference between surface and deep learning?",
    "Explain growth vs fixed mindset.",
    "What is elaborative interrogation?",
    "Explain the Feynman technique.",
    "Difference between recognition and recall?"
]

# -------------------------
# LOAD MODEL
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
config.use_cache = True   # ← IMPORTANT (GLOBAL ENABLE)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"
)

model.eval()
precision_label = "FP16"
print("Model loaded.")

# -------------------------
# IDLE POWER
# -------------------------
print("\n[1] Measuring idle power...")
idle_tracker = EmissionsTracker(measure_power_secs=2, save_to_file=False, log_level="error")
idle_tracker.start()
torch.cuda.synchronize()
time.sleep(10)
idle_tracker.stop()

idle_energy = idle_tracker._total_energy
idle_watts = (idle_energy.kWh * 3.6e6) / 10 if idle_energy else 0.0
print(f"Idle power: {idle_watts:.3f} W")

# -------------------------
# RUN EXPERIMENT
# -------------------------
results = []
tracker = EmissionsTracker(project_name="kvcache_true", measure_power_secs=1,
                          save_to_file=False, log_level="error")
tracker.start()

print("\n[2] Running 50 prompts (KV-CACHE=TRUE)")

for idx, prompt in enumerate(PROMPTS):
    task_id = idx + 1
    category = (
        "Mathematics" if idx < 10 else
        "Science" if idx < 20 else
        "Programming-CS" if idx < 30 else
        "Humanities" if idx < 40 else
        "Meta-cognition"
    )

    tracker._measure_power_and_energy()
    e_start = tracker._total_energy.kWh

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,        # ← ACTUAL KV-CACHE
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    latency = time.time() - t0

    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    tracker._measure_power_and_energy()
    e_end = tracker._total_energy.kWh

    gross_j = (e_end - e_start) * 3.6e6
    net_j   = max(gross_j - idle_watts * latency, 0.01)
    power_w = net_j / latency

    results.append({
        "ID": task_id,
        "Precision": precision_label,
        "Category": category,
        "Prompt": prompt,
        "Response": response,
        "Input_Tokens": input_len,
        "Latency_s": round(latency, 4),
        "Net_Energy_J": round(net_j, 4),
        "Power_W": round(power_w, 2),
        "use_cache": True
    })

    print(f"[{task_id:02d}] {category:<14} | {latency:.2f}s | {net_j:.1f}J")

tracker.stop()

# -------------------------
# SAVE CSV (EXCEL READY)
# -------------------------
df = pd.DataFrame(results)
df.to_csv(OUTPUT_FILE, index=False)
files.download(OUTPUT_FILE)

print("\n" + "="*60)
print("KV-CACHE=TRUE | FP16 | n=50")
print(f"Avg Latency:    {df.Latency_s.mean():.2f}s")
print(f"Avg Net Energy: {df.Net_Energy_J.mean():.1f}J")
print(f"Avg Power:      {df.Power_W.mean():.1f}W")
print("="*60)

print("\nPer-category means:")
print(df.groupby("Category")[["Latency_s","Net_Energy_J","Power_W"]].mean().round(2))
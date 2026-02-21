import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from codecarbon import EmissionsTracker

# -----------------------
# USER CONFIG
# -----------------------
MODEL_ID = "microsoft/Phi-4-mini-instruct"
MAX_NEW_TOKENS = 200
USE_QUANTIZATION = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

precision_label = "INT4-NF4" if USE_QUANTIZATION else "FP16"

print("Device:", DEVICE)
print("USE_QUANTIZATION:", USE_QUANTIZATION)
print("Precision:", precision_label)

# -----------------------
# TOKENIZER
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------
# NF4 QUANTIZATION
# -----------------------
bnb_config = None
if USE_QUANTIZATION:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

# -----------------------
# MODEL
# -----------------------
print("\nLoading model (this may take a minute)...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

model.eval()
print("Model loaded successfully.")

# -----------------------
# IDLE POWER (CodeCarbon)
# -----------------------
idle_tracker = EmissionsTracker(
    measure_power_secs=2.0,
    save_to_file=False,
    log_level="error"
)
idle_tracker.start()
if DEVICE == "cuda":
    torch.cuda.synchronize()
time.sleep(10)
idle_tracker.stop()

idle_energy = getattr(idle_tracker, "_total_energy", None)
idle_watts = (idle_energy.kWh * 3.6e6) / 10.0 if idle_energy else 0.0
print(f"Idle power: {idle_watts:.2f} W")

# -----------------------
# 50 CUSTOM PROMPTS
# -----------------------
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

    # Programming / CS (10)
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
    "Difference between recognition and recall?",
]

CATEGORIES = (
    ["Mathematics"] * 10 +
    ["Science"] * 10 +
    ["Programming_CS"] * 10 +
    ["Humanities"] * 10 +
    ["Meta_cognition"] * 10
)

# -----------------------
# INFERENCE LOOP
# -----------------------
results = []

for idx, (prompt, category) in enumerate(zip(PROMPTS, CATEGORIES)):
    task_id = idx + 1
    print(f"\n[{task_id}/50] Category: {category}")
    print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    tracker = EmissionsTracker(
        project_name=f"phi4_prompt_{task_id}",
        measure_power_secs=1.0,
        save_to_file=False,
        log_level="error"
    )
    tracker.start()

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,              # KV CACHE ON
            pad_token_id=tokenizer.eos_token_id,
        )

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    latency = time.time() - t0

    tracker._measure_power_and_energy()
    tracker.stop()

    seq = outputs if not hasattr(outputs, "sequences") else outputs.sequences
    output_ids = seq[0][input_len:]
    response_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    tokens_generated = len(output_ids)
    tokens_per_sec = tokens_generated / latency if latency > 0 else 0.0

    # Energy calculations
    energy_obj = getattr(tracker, "_total_energy", None)
    energy_kwh = energy_obj.kWh if energy_obj else 0.0
    energy_joules = energy_kwh * 3.6e6
    avg_power_w = energy_joules / latency if latency > 0 else 0.0

    # Net energy: subtract idle baseline
    idle_energy_for_duration = idle_watts * latency          # J consumed at idle
    net_j = max(energy_joules - idle_energy_for_duration, 0.0)
    power_w = max(avg_power_w - idle_watts, 0.0)

    co2_g = getattr(tracker, "final_emissions", 0.0) * 1000  # kg -> g

    print(f"  Latency: {latency:.2f}s | Tokens: {tokens_generated} | "
          f"Tok/s: {tokens_per_sec:.1f} | Energy: {energy_joules:.2f}J | "
          f"Net Energy: {net_j:.2f}J | Net Power: {power_w:.2f}W | CO₂: {co2_g:.4f}g")

    results.append({
        "ID": task_id,
        "Precision": precision_label,
        "Category": category,
        "Prompt": prompt,
        "Response": response_text,
        "Input_Tokens": int(input_len),
        "Output_Tokens": tokens_generated,
        "Latency_s": round(latency, 4),
        "Tokens_per_sec": round(tokens_per_sec, 2),
        "Energy_J": round(energy_joules, 4),
        "Net_Energy_J": round(net_j, 4),
        "Power_W": round(power_w, 2),
        "CO2_g": round(co2_g, 6),
        "use_cache": True,
    })

# -----------------------
# RESULTS DATAFRAME
# -----------------------
df = pd.DataFrame(results)

print("\n\n====== SUMMARY BY CATEGORY ======")
summary = df.groupby("Category").agg(
    avg_latency_s=("Latency_s", "mean"),
    avg_tokens_per_sec=("Tokens_per_sec", "mean"),
    avg_output_tokens=("Output_Tokens", "mean"),
    total_energy_J=("Energy_J", "sum"),
    total_net_energy_J=("Net_Energy_J", "sum"),
    avg_power_w=("Power_W", "mean"),
    total_co2_g=("CO2_g", "sum"),
).round(4)
print(summary.to_string())

print("\n\n====== OVERALL STATS ======")
print(f"Total prompts       : {len(df)}")
print(f"Precision           : {precision_label}")
print(f"Total latency       : {df['Latency_s'].sum():.2f}s")
print(f"Avg latency/prompt  : {df['Latency_s'].mean():.2f}s")
print(f"Avg tokens/sec      : {df['Tokens_per_sec'].mean():.2f}")
print(f"Total energy        : {df['Energy_J'].sum():.4f} J")
print(f"Total net energy    : {df['Net_Energy_J'].sum():.4f} J")
print(f"Total CO₂           : {df['CO2_g'].sum():.4f} g")

# Save results
df.to_csv("phi4_50prompt_results.csv", index=False)
print("\nResults saved to phi4_50prompt_results.csv")
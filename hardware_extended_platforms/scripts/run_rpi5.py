#!/usr/bin/env python3
# ==============================================================================
#  GREEN LEARNING AUDIT — Raspberry Pi 5 (4GB / 8GB)
#  Backend: llama.cpp (Python bindings via llama-cpp-python)
#  Model:   Phi-3 Mini 4k Instruct — Q4_K_M GGUF only
#
#  SETUP (run once in terminal on the Pi):
#    sudo apt update && sudo apt install python3-pip python3-venv -y
#    python3 -m venv lpw_env && source lpw_env/bin/activate
#    pip install llama-cpp-python codecarbon pandas
#    pip install huggingface-hub
#    huggingface-cli download bartowski/Phi-3-mini-4k-instruct-GGUF \
#        Phi-3-mini-4k-instruct-Q4_K_M.gguf \
#        --local-dir ./models
#
#  NOTE ON F16: Phi-3 Mini F16 GGUF requires ~7.6 GB RAM.
#    Pi 5 4GB  → F16 NOT FEASIBLE (OOM)
#    Pi 5 8GB  → F16 technically fits but swap thrashing makes results invalid
#    This script runs Q4_K_M only.
#
#  THERMAL NOTE: Pi 5 will thermal throttle under sustained load.
#    Use active cooling (official Pi 5 cooler or heatsink + fan).
#    Script logs CPU frequency per-prompt so throttling is detectable.
#
#  n=100 prompts (not 500): at ~130s/prompt, 500 = ~18 hours.
#    100 prompts = ~3.6 hours, matches Appendix D methodology.
# ==============================================================================

import time
import os
import platform
import subprocess
import pandas as pd
from llama_cpp import Llama
from codecarbon import EmissionsTracker

# ==============================================================================
# ── CONFIGURATION ──────────────────────────────────────────────────────────────
# ==============================================================================

MODEL_PATH  = "./models/Phi-3-mini-4k-instruct-Q4_K_M.gguf"
N_THREADS   = 4          # Pi 5 has 4 Cortex-A76 cores — use all
N_CTX       = 1024       # Smaller context saves RAM
MAX_TOKENS  = 200
OUTPUT_DIR  = "green_audit_output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "rpi5_Q4_K_M.csv")
N_PROMPTS   = 100        # Matches Appendix D CPU baseline

# ==============================================================================
# ── DO NOT EDIT BELOW ──────────────────────────────────────────────────────────
# ==============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print(f"  Green Learning Audit — Raspberry Pi 5")
print(f"  Platform : {platform.machine()} / {platform.processor()}")
print(f"  Precision: Q4_K_M (F16 not feasible — see header note)")
print(f"  n_prompts: {N_PROMPTS}")
print("=" * 60)

# ── CPU frequency helper (detects thermal throttling) ─────────────────────────
def get_cpu_freq_mhz():
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_clock", "arm"],
            capture_output=True, text=True, timeout=2
        )
        freq_hz = int(result.stdout.split("=")[1].strip())
        return round(freq_hz / 1e6, 0)
    except Exception:
        try:
            with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
                return round(int(f.read().strip()) / 1e3, 0)
        except Exception:
            return None

# ── Temperature helper ─────────────────────────────────────────────────────────
def get_cpu_temp():
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_temp"],
            capture_output=True, text=True, timeout=2
        )
        temp = float(result.stdout.split("=")[1].replace("'C", "").strip())
        return temp
    except Exception:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return round(int(f.read().strip()) / 1e3, 1)
        except Exception:
            return None

# ==============================================================================
# ── 100 PROMPTS — same first-100 as Appendix D sampling ──────────────────────
# (20 per category, balanced)
# ==============================================================================

PROMPTS = [
    # Mathematics (20)
    "What is a limit in calculus and how do you evaluate lim(x→2) of (x² - 4)/(x - 2)?",
    "Explain how to solve a quadratic equation using the quadratic formula, and walk through 2x² + 5x - 3 = 0.",
    "What is the difference between a function and a relation? Give an example of each.",
    "Explain what the slope of a line represents and how to calculate it from two points.",
    "What are exponent rules? Explain the product rule, quotient rule, and power rule with examples.",
    "Explain what a logarithm is and how it relates to exponentiation. What is log₂(8)?",
    "What is the chain rule in calculus and when do you use it?",
    "Explain what a derivative represents geometrically and how to find the derivative of x³.",
    "What is integration and how does it relate to area under a curve?",
    "Explain the difference between mean, median, and mode.",
    "What is standard deviation and what does it tell you about a dataset?",
    "Explain what a normal distribution is and what the 68-95-99.7 rule means.",
    "What is probability and how do you calculate the probability of two independent events both occurring?",
    "What is the Pythagorean theorem and how do you apply it to find a missing side?",
    "What is modular arithmetic and how does it work?",
    "Explain what a vector is and how to add two vectors.",
    "What is Euler's formula and why is it considered beautiful?",
    "Explain the concept of infinity in mathematics — is infinity a number?",
    "What is a hypothesis test and what does p-value mean?",
    "Explain the difference between correlation and causation.",

    # Science (20)
    "Explain what photosynthesis is, what inputs it needs, and what it produces.",
    "What is cellular respiration and how does it differ from breathing?",
    "Explain the structure of DNA and what base pairing rules apply.",
    "What is mitosis and why do cells need to divide?",
    "What is natural selection and how does it lead to evolution over time?",
    "Explain what an enzyme is and how it works.",
    "What is osmosis and how does it differ from diffusion?",
    "What is the difference between aerobic and anaerobic respiration?",
    "What is Newton's first law of motion?",
    "Explain Newton's second law: F = ma.",
    "What is Newton's third law and give an example.",
    "Explain what momentum is and state the law of conservation of momentum.",
    "What is gravitational potential energy and how is it calculated?",
    "Explain what kinetic energy is.",
    "What is the principle of conservation of energy?",
    "Explain what a wave is and the difference between transverse and longitudinal waves.",
    "What is the electromagnetic spectrum?",
    "What is the difference between conductors and insulators?",
    "Explain Ohm's law.",
    "What is the difference between series and parallel circuits?",

    # Programming-CS (20)
    "What is a binary tree and explain what root, leaf, and node mean.",
    "What is a variable in programming? Explain it like I am 10 years old using a box analogy.",
    "What is a for loop? Explain it like a chore list.",
    "What is a function in programming and why do we use them?",
    "Explain what recursion is with the example of calculating a factorial.",
    "What is the difference between a syntax error and a runtime error?",
    "Explain what a stack data structure is and give an example of its use.",
    "What is a sorting algorithm? Explain bubble sort.",
    "What is Big O notation and what does O(n) mean?",
    "What is object-oriented programming?",
    "Explain what a class and an object are in OOP.",
    "What is inheritance in OOP? Give an example.",
    "Explain what an operating system does.",
    "What is virtual memory?",
    "Explain what a database is and the difference between SQL and NoSQL.",
    "Explain what an API is.",
    "What is the difference between HTTP and HTTPS?",
    "Explain what encryption is and the difference between symmetric and asymmetric encryption.",
    "What is machine learning in simple terms?",
    "Explain what binary search is and why it is faster than linear search.",

    # Humanities (20)
    "What is the bystander effect and what causes it?",
    "Explain the main causes of World War I using the MAIN acronym.",
    "What was the significance of the French Revolution and its key outcomes?",
    "Explain what the Industrial Revolution was and how it changed society.",
    "What was the Cold War and the main tensions between the USA and USSR?",
    "What was the significance of the Magna Carta?",
    "What were the main causes of World War II?",
    "Explain the importance of the Renaissance period.",
    "What was the significance of the Civil Rights Movement in the USA?",
    "What is globalisation and what are its effects?",
    "Explain what the United Nations does.",
    "What is democracy and how does it differ from authoritarianism?",
    "What is propaganda and how is it used?",
    "Explain what a GDP and what does it measure?",
    "Explain the difference between developed and developing countries.",
    "What is the separation of powers?",
    "What is a constitution?",
    "What was the significance of the printing press?",
    "Explain what the Silk Road was.",
    "What was the significance of the Nuremberg Trials?",

    # Meta-cognition (20)
    "Explain how to manage your time effectively during exam revision.",
    "Explain what metacognition means and why it helps learning.",
    "What is spaced repetition and why is it effective?",
    "Explain the Pomodoro technique.",
    "What is active recall?",
    "Explain interleaving in studying.",
    "What is elaborative interrogation?",
    "Explain the Feynman technique.",
    "What is the difference between recognition and recall?",
    "Explain what retrieval practice is and why it works.",
    "What is working memory and how does it affect learning?",
    "Explain what long-term memory is and how information gets stored there.",
    "What is encoding in memory?",
    "What is deliberate practice?",
    "What is transfer of learning?",
    "Explain what schema theory says about how we learn.",
    "What is scaffolding in education?",
    "What is the difference between intrinsic and extrinsic motivation?",
    "Explain what self-efficacy is.",
    "What is academic procrastination and how do you overcome it?",
]

CATEGORIES = (
    ["Mathematics"]    * 20 +
    ["Science"]        * 20 +
    ["Programming-CS"] * 20 +
    ["Humanities"]     * 20 +
    ["Meta-cognition"] * 20
)

assert len(PROMPTS) == 100

# ==============================================================================
# ── MODEL LOAD ─────────────────────────────────────────────────────────────────
# ==============================================================================

assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}\nRun setup commands above."

print(f"\nLoading Q4_K_M model from {MODEL_PATH}...")
print("(This may take 30–60 seconds on Pi 5)")
llm = Llama(
    model_path=MODEL_PATH,
    n_threads=N_THREADS,
    n_ctx=N_CTX,
    n_gpu_layers=0,
    verbose=False,
)
print("Model loaded.")

# ==============================================================================
# ── IDLE BASELINE ──────────────────────────────────────────────────────────────
# ==============================================================================

print("\nMeasuring idle power baseline (10 seconds)...")
idle_tracker = EmissionsTracker(measure_power_secs=2, save_to_file=False, log_level="error")
idle_tracker.start()
time.sleep(10)
idle_tracker.stop()

idle_energy = getattr(idle_tracker, "_total_energy", None)
idle_watts = (idle_energy.kWh * 3.6e6) / 10.0 if idle_energy else 0.0
print(f"Idle power: {idle_watts:.2f} W")

print("\nWARNING: Ensure active cooling is attached. Thermal throttling")
print("         invalidates timing measurements. Throttle events are")
print("         logged per-prompt via CPU frequency column.\n")

# ==============================================================================
# ── INFERENCE LOOP ─────────────────────────────────────────────────────────────
# ==============================================================================

results = []

main_tracker = EmissionsTracker(
    project_name="rpi5_Q4_K_M",
    measure_power_secs=1,
    save_to_file=False,
    log_level="error",
)
main_tracker.start()

print(f"Running {N_PROMPTS} prompts — Q4_K_M | llama.cpp | Pi 5\n")
print(f"{'ID':>4} {'Category':<16} {'Latency':>8} {'Net_J':>7} {'Tok/s':>6} {'Temp°C':>7} {'MHz':>6}")
print("-" * 56)

for idx, (prompt, category) in enumerate(zip(PROMPTS, CATEGORIES)):
    task_id = idx + 1

    cpu_freq_before = get_cpu_freq_mhz()
    temp_before = get_cpu_temp()

    main_tracker._measure_power_and_energy()
    e_before = main_tracker._total_energy.kWh

    t0 = time.time()
    output = llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        echo=False,
    )
    latency = time.time() - t0

    main_tracker._measure_power_and_energy()
    e_after = main_tracker._total_energy.kWh

    cpu_freq_after = get_cpu_freq_mhz()
    temp_after = get_cpu_temp()

    response_text  = output["choices"][0]["text"]
    tokens_out     = output["usage"]["completion_tokens"]
    tokens_per_sec = tokens_out / latency if latency > 0 else 0

    gross_j = (e_after - e_before) * 3.6e6
    net_j   = max(gross_j - idle_watts * latency, 0.01)
    power_w = net_j / latency if latency > 0 else 0

    # Detect throttling: freq drop > 10% from max (2400 MHz on Pi 5)
    throttled = (cpu_freq_after is not None and cpu_freq_after < 2100)

    results.append({
        "ID":              task_id,
        "HW_Platform":     "Raspberry Pi 5 4GB (BCM2712 Cortex-A76)",
        "Backend":         "llama.cpp",
        "Precision":       "Q4_K_M",
        "Category":        category,
        "Prompt":          prompt,
        "Response":        response_text,
        "Output_Tokens":   tokens_out,
        "Latency_s":       round(latency, 4),
        "Tokens_per_sec":  round(tokens_per_sec, 3),
        "Gross_Energy_J":  round(gross_j, 4),
        "Net_Energy_J":    round(net_j, 4),
        "Power_W":         round(power_w, 3),
        "CPU_Freq_MHz":    cpu_freq_after,
        "CPU_Temp_C":      temp_after,
        "Throttled":       throttled,
        "Q_ped":           "",
        "LpW":             "",
    })

    freq_str = f"{cpu_freq_after:.0f}" if cpu_freq_after else "N/A"
    temp_str = f"{temp_after:.1f}" if temp_after else "N/A"
    throttle_flag = " ⚠THROTTLE" if throttled else ""
    print(
        f"{task_id:>4} {category:<16} {latency:>7.1f}s {net_j:>7.1f}J "
        f"{tokens_per_sec:>5.2f} {temp_str:>7} {freq_str:>6}{throttle_flag}"
    )

    if task_id % 20 == 0:
        pd.DataFrame(results).to_csv(
            os.path.join(OUTPUT_DIR, f"checkpoint_rpi5_{task_id}.csv"), index=False
        )
        print(f"  >>> Checkpoint saved at {task_id}")

main_tracker.stop()

# ==============================================================================
# ── SAVE & SUMMARY ─────────────────────────────────────────────────────────────
# ==============================================================================

df = pd.DataFrame(results)
df.to_csv(OUTPUT_FILE, index=False)

throttled_count = df["Throttled"].sum() if "Throttled" in df else 0

print("\n" + "=" * 60)
print(f"  RESULTS — Q4_K_M | Raspberry Pi 5 | n={N_PROMPTS}")
print("=" * 60)
print(f"  Avg Latency     : {df.Latency_s.mean():.1f}s")
print(f"  Avg Net Energy  : {df.Net_Energy_J.mean():.1f} J")
print(f"  Avg Power       : {df.Power_W.mean():.2f} W")
print(f"  Avg Tokens/sec  : {df.Tokens_per_sec.mean():.2f}")
print(f"  Throttled runs  : {throttled_count} / {N_PROMPTS}")
print(f"  Saved           : {OUTPUT_FILE}")
print("=" * 60)

if throttled_count > 10:
    print("\n  ⚠ WARNING: High throttle count — consider excluding")
    print("    throttled rows from analysis or re-run with active cooling.")

print("\nPer-category:")
print(df.groupby("Category").agg(
    avg_lat     =("Latency_s",    "mean"),
    avg_energy  =("Net_Energy_J", "mean"),
    avg_power   =("Power_W",      "mean"),
    avg_tok_s   =("Tokens_per_sec","mean"),
).round(2).to_string())

# ==============================================================================
# After scoring: compute LpW
# df = pd.read_csv("green_audit_output/rpi5_Q4_K_M_scored.csv")
# df["LpW"] = df["Q_ped"] / (df["Net_Energy_J"] * df["Latency_s"])
# ==============================================================================

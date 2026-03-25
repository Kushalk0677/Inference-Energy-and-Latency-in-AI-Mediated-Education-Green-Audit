# ==============================================================================
#  EXPERIMENT — Phi-3-mini-4k-instruct
#  Run this script TWICE:
#    Run 1: SET USE_QUANTIZATION = False   → produces FP16 results
#    Run 2: SET USE_QUANTIZATION = True    → produces NF4 results
#  Each run saves a CSV. Compare the two CSVs for your corrected analysis.
# ==============================================================================

# ── INSTALL (Colab cell 1) ────────────────────────────────────────────────────
# Run this first, then restart the runtime before running the rest of the script
# !pip install -q "transformers==4.44.2" accelerate bitsandbytes codecarbon
import time
import os
import glob
import importlib
import sys
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
)
from codecarbon import EmissionsTracker
# from google.colab import files  # uncomment if running in Colab

# ==============================================================================
# ── CONFIGURATION — only edit this section ────────────────────────────────────
# ==============================================================================

MODEL_ID       = "microsoft/Phi-3-mini-4k-instruct"
MAX_NEW_TOKENS = 200
USE_QUANTIZATION = False   # False = FP16 (Run 1) | True = NF4 (Run 2)

# ==============================================================================
# ── DO NOT EDIT BELOW THIS LINE ───────────────────────────────────────────────
# ==============================================================================

PRECISION_LABEL = "NF4" if USE_QUANTIZATION else "FP16"
OUTPUT_FILE     = f"phi3_{PRECISION_LABEL}_corrected_500prompts.csv"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print(f"  Phi-3 Mini Corrected Experiment")
print(f"  Precision : {PRECISION_LABEL}")
print(f"  Device    : {DEVICE}")
print(f"  use_cache : True (KV-cache enabled — standard inference)")
print("=" * 60)

# ==============================================================================
# ── 500 PROMPTS (100 per category, same as original study) ────────────────────
# ==============================================================================

PROMPTS_MATHEMATICS = [
    "What is a limit in calculus and how do you evaluate lim(x→2) of (x² - 4)/(x - 2)?",
    "Explain how to solve a quadratic equation using the quadratic formula, and walk through 2x² + 5x - 3 = 0.",
    "What is the difference between a function and a relation? Give an example of each.",
    "Explain what the slope of a line represents and how to calculate it from two points.",
    "How do you factor a trinomial of the form ax² + bx + c? Show the process with an example.",
    "What are exponent rules? Explain the product rule, quotient rule, and power rule with examples.",
    "Explain what a logarithm is and how it relates to exponentiation. What is log₂(8)?",
    "What is the difference between linear and exponential growth? Give a real-world example of each.",
    "How do you solve a system of two linear equations using substitution?",
    "Explain what absolute value means geometrically on a number line.",
    "What is the binomial theorem and how would you expand (x + y)³?",
    "Explain the difference between a permutation and a combination.",
    "What is the chain rule in calculus and when do you use it?",
    "Explain what a derivative represents geometrically and how to find the derivative of x³.",
    "What is integration and how does it relate to area under a curve?",
    "Explain what a matrix is and how to multiply two 2x2 matrices.",
    "What is the Pythagorean theorem and how do you apply it to find a missing side?",
    "Explain what a prime number is and why 1 is not considered prime.",
    "What is modular arithmetic and how does it work?",
    "Explain the difference between mean, median, and mode.",
    "What is standard deviation and what does it tell you about a dataset?",
    "Explain what a normal distribution is and what the 68-95-99.7 rule means.",
    "What is probability and how do you calculate the probability of two independent events both occurring?",
    "Explain what a geometric series is and how to find its sum.",
    "What is the difference between a rational and an irrational number?",
    "Explain what a vector is and how to add two vectors.",
    "What is the dot product of two vectors and what does it represent geometrically?",
    "Explain what a complex number is and how to multiply two complex numbers.",
    "What is Euler's formula and why is it considered beautiful?",
    "Explain the concept of infinity in mathematics — is infinity a number?",
    "What is a proof by contradiction? Give a simple example.",
    "Explain what a function's domain and range are.",
    "What is the difference between a local and a global maximum?",
    "Explain what L'Hôpital's rule is and when you use it.",
    "What is a Taylor series and what is it used for?",
    "Explain what the determinant of a matrix represents.",
    "What is an eigenvalue and why does it matter?",
    "Explain what Bayes' theorem is with a simple real-world example.",
    "What is the difference between theoretical and experimental probability?",
    "Explain what a confidence interval is in statistics.",
    "What is a hypothesis test and what does p-value mean?",
    "Explain the difference between correlation and causation.",
    "What is linear regression and what does the regression line represent?",
    "Explain what a fractal is and give an example.",
    "What is the Fibonacci sequence and where does it appear in nature?",
    "Explain what a set is in mathematics and what the union and intersection of two sets mean.",
    "What is a bijection and why does it matter for comparing the size of infinite sets?",
    "Explain Cantor's diagonal argument in simple terms.",
    "What is a graph in discrete mathematics (vertices and edges)?",
    "Explain what the travelling salesman problem is and why it is hard to solve.",
    "What is the four-colour theorem?",
    "Explain what a recursive sequence is and give an example.",
    "What is Pascal's triangle and how does it relate to binomial coefficients?",
    "Explain what a Z-score is and how to calculate it.",
    "What is the difference between a parameter and a statistic?",
    "Explain what a type I and type II error are in hypothesis testing.",
    "What is the law of large numbers?",
    "Explain what the central limit theorem says.",
    "What is a Poisson distribution and when is it used?",
    "Explain what a binomial distribution is.",
    "What is the difference between discrete and continuous probability distributions?",
    "Explain what a random variable is.",
    "What is expected value and how do you calculate it?",
    "Explain what variance is and how it differs from standard deviation.",
    "What is a box plot and how do you read one?",
    "Explain what a scatter plot shows and how to interpret it.",
    "What is the interquartile range and why is it useful?",
    "Explain what an outlier is and how it affects the mean.",
    "What is a two-way table and how do you read one?",
    "Explain what a cumulative frequency graph shows.",
    "What is a histogram and how does it differ from a bar chart?",
    "Explain what the area model for multiplication is.",
    "What is long division and how does it work?",
    "Explain what significant figures are and why they matter.",
    "What is scientific notation and how do you convert a number to it?",
    "Explain what a percentage increase and decrease are.",
    "What is compound interest and how does it differ from simple interest?",
    "Explain what a ratio is and how to simplify one.",
    "What is direct proportion and how do you identify it from a graph?",
    "Explain what inverse proportion means.",
    "What is a sequence and what is the difference between arithmetic and geometric sequences?",
    "Explain how to find the nth term of an arithmetic sequence.",
    "What is the sum of the first n natural numbers?",
    "Explain what a venn diagram shows.",
    "What is a tree diagram in probability?",
    "Explain what a sample space is.",
    "What is complementary probability?",
    "Explain what mutually exclusive events are.",
    "What is conditional probability?",
    "Explain the addition rule of probability.",
    "What is the multiplication rule of probability?",
    "Explain what a quartile is.",
    "What is the difference between a population and a sample in statistics?",
    "Explain what sampling bias is.",
    "What is a stratified sample?",
    "Explain what a frequency table is and how to construct one.",
    "What is a stem-and-leaf plot?",
    "Explain what back-to-back stem-and-leaf plots are used for.",
    "What is the difference between primary and secondary data?",
    "Explain what a misleading graph is and how to spot one.",
]

PROMPTS_SCIENCE = [
    "What are the main causes of deforestation and its effects on the environment?",
    "Explain what photosynthesis is, what inputs it needs, and what it produces.",
    "What is cellular respiration and how does it differ from breathing?",
    "Explain the structure of DNA and what base pairing rules apply.",
    "What is mitosis and why do cells need to divide?",
    "Explain what meiosis is and how it differs from mitosis.",
    "What is natural selection and how does it lead to evolution over time?",
    "Explain the difference between an ecosystem and a biome.",
    "What is the role of the cell membrane and what does selectively permeable mean?",
    "Explain the difference between prokaryotic and eukaryotic cells.",
    "What is the function of the mitochondria and why is it called the powerhouse of the cell?",
    "Explain what an enzyme is and how it works.",
    "What is osmosis and how does it differ from diffusion?",
    "Explain the structure and function of the nucleus.",
    "What is the difference between aerobic and anaerobic respiration?",
    "Explain what a food web is and how energy flows through it.",
    "What is the nitrogen cycle and why is it important?",
    "Explain what a chromosome is and how it relates to genes and DNA.",
    "What is a gene and what does it mean for a gene to be expressed?",
    "Explain what a mutation is and give an example of how it can affect an organism.",
    "What is the difference between dominant and recessive alleles?",
    "Explain what a Punnett square is and how to use it.",
    "What is codominance? Give an example.",
    "Explain what sex-linked inheritance is.",
    "What is the difference between genotype and phenotype?",
    "Explain what homeostasis is and give two examples from the human body.",
    "What is the role of insulin and glucagon in blood sugar regulation?",
    "Explain what the nervous system does and how a nerve impulse travels.",
    "What is a reflex arc and why are reflexes important?",
    "Explain the structure of the heart and how blood flows through it.",
    "What is the difference between arteries and veins?",
    "Explain what gas exchange is and where it happens in humans.",
    "What is the role of haemoglobin?",
    "Explain what the immune system does and the difference between antigens and antibodies.",
    "What is a vaccine and how does it work?",
    "Explain what active and passive immunity are.",
    "What is the difference between bacteria and viruses?",
    "Explain how antibiotics work and why antibiotic resistance is a problem.",
    "What is the role of the kidney in the body?",
    "Explain what dialysis is and why some patients need it.",
    "What is Newton's first law of motion?",
    "Explain Newton's second law: F = ma.",
    "What is Newton's third law and give an example.",
    "Explain what momentum is and state the law of conservation of momentum.",
    "What is the difference between speed and velocity?",
    "Explain what acceleration is and how to calculate it.",
    "What is gravitational potential energy and how is it calculated?",
    "Explain what kinetic energy is.",
    "What is the principle of conservation of energy?",
    "Explain what work is in physics and how it is calculated.",
    "What is power in physics and how does it differ from energy?",
    "Explain what pressure is and give the formula.",
    "What is Hooke's law?",
    "Explain what a wave is and the difference between transverse and longitudinal waves.",
    "What is the difference between reflection and refraction?",
    "Explain what total internal reflection is.",
    "What is the electromagnetic spectrum?",
    "Explain what ionising radiation is and name three types.",
    "What is half-life?",
    "Explain what nuclear fission and fusion are.",
    "What is the difference between conductors and insulators?",
    "Explain Ohm's law.",
    "What is the difference between series and parallel circuits?",
    "Explain what voltage, current, and resistance are.",
    "What is electric power and how is it calculated?",
    "Explain what a magnetic field is.",
    "What is electromagnetic induction?",
    "Explain what a transformer does and how it works.",
    "What is the difference between an element, compound, and mixture?",
    "Explain what an atom is and describe its structure.",
    "What is the periodic table and how is it organised?",
    "Explain what isotopes are.",
    "What is ionic bonding?",
    "Explain what covalent bonding is.",
    "What is the difference between ionic and covalent compounds?",
    "Explain what an acid and a base are.",
    "What is pH and what does it measure?",
    "Explain what a neutralisation reaction is.",
    "What is oxidation and reduction?",
    "Explain what electrolysis is.",
    "What is the difference between exothermic and endothermic reactions?",
    "Explain what a catalyst does.",
    "What is the rate of reaction and what factors affect it?",
    "Explain what collision theory says.",
    "What is a reversible reaction and what is meant by equilibrium?",
    "Explain Le Chatelier's principle.",
    "What is a polymer and give an example of how one is made?",
    "Explain what crude oil is and how fractional distillation works.",
    "What are the products of combustion of a hydrocarbon?",
    "Explain what the greenhouse effect is.",
    "What is global warming and how does CO₂ contribute to it?",
    "Explain what acid rain is and what causes it.",
    "What is eutrophication?",
    "Explain what biodiversity is and why it matters.",
    "What is a keystone species?",
    "Explain what carbon footprint means.",
    "What is renewable versus non-renewable energy?",
    "Explain how solar panels work.",
    "What is the difference between a food chain and a food web?",
    "Explain what a limiting factor is in population ecology.",
]

PROMPTS_PROGRAMMING_CS = [
    "What is a binary tree and explain what root, leaf, and node mean.",
    "What is a variable in programming? Explain it like I am 10 years old using a box analogy.",
    "What is a for loop? Explain it like a chore list.",
    "Explain what an if statement is in programming using a traffic light analogy.",
    "What is a function in programming and why do we use them?",
    "Explain the difference between a while loop and a for loop.",
    "What is a list (or array) in programming and how do you access elements in it?",
    "Explain what a bug is in programming and give an example of a common one.",
    "What does it mean to compile a program?",
    "Explain what recursion is with the example of calculating a factorial.",
    "What is the difference between a syntax error and a runtime error?",
    "Explain what a stack data structure is and give an example of its use.",
    "What is a queue data structure and how does it differ from a stack?",
    "Explain what a linked list is.",
    "What is a hash table and how does it work?",
    "Explain what a graph data structure is and give a real-world example.",
    "What is a sorting algorithm? Explain bubble sort.",
    "Explain how merge sort works.",
    "What is quicksort and how does it work?",
    "Explain what binary search is and why it is faster than linear search.",
    "What is Big O notation and what does O(n) mean?",
    "Explain the difference between O(n) and O(n²) with an example.",
    "What is object-oriented programming?",
    "Explain what a class and an object are in OOP.",
    "What is inheritance in OOP? Give an example.",
    "Explain what polymorphism means in programming.",
    "What is encapsulation in OOP?",
    "Explain what abstraction means in programming.",
    "What is the difference between a compiler and an interpreter?",
    "Explain what an operating system does.",
    "What is virtual memory?",
    "Explain what a process and a thread are.",
    "What is deadlock in computing?",
    "Explain what a database is and the difference between SQL and NoSQL.",
    "What is a primary key in a database?",
    "Explain what a JOIN is in SQL.",
    "What is normalisation in databases?",
    "Explain what an API is.",
    "What is the difference between HTTP and HTTPS?",
    "Explain what a DNS server does.",
    "What is an IP address?",
    "Explain the difference between TCP and UDP.",
    "What is a firewall and what does it do?",
    "Explain what encryption is and the difference between symmetric and asymmetric encryption.",
    "What is a public key and a private key?",
    "Explain what a man-in-the-middle attack is.",
    "What is SQL injection?",
    "Explain what a denial-of-service attack is.",
    "What is two-factor authentication?",
    "Explain what a VPN is and how it works.",
    "What is machine learning in simple terms?",
    "Explain the difference between supervised and unsupervised learning.",
    "What is a neural network?",
    "Explain what overfitting means in machine learning.",
    "What is a training set and a test set?",
    "Explain what a decision tree classifier is.",
    "What is the difference between classification and regression?",
    "Explain what natural language processing is.",
    "What is a large language model?",
    "Explain what tokenisation is in NLP.",
    "What is the difference between RAM and ROM?",
    "Explain what the CPU does.",
    "What is the fetch-decode-execute cycle?",
    "Explain what cache memory is and why it matters.",
    "What is the difference between 32-bit and 64-bit systems?",
    "Explain what binary is and how to convert 13 to binary.",
    "What is hexadecimal and why is it used in computing?",
    "Explain what a logic gate is and describe AND, OR, and NOT gates.",
    "What is a truth table?",
    "Explain what a flip-flop circuit is used for.",
    "What is the difference between lossless and lossy compression?",
    "Explain how run-length encoding works.",
    "What is the difference between a text file and a binary file?",
    "Explain what a pixel is and how colour is represented digitally.",
    "What is sampling in audio digitisation?",
    "Explain what bit depth means in audio.",
    "What is a file format and why do different formats exist?",
    "Explain what version control is and why developers use it.",
    "What is Git and what is a commit?",
    "Explain what open source software means.",
    "What is agile development?",
    "Explain what a software requirement specification is.",
    "What is the difference between white-box and black-box testing?",
    "Explain what unit testing is.",
    "What is a stack overflow error?",
    "Explain what an infinite loop is and how to avoid it.",
    "What is the difference between pass by value and pass by reference?",
    "Explain what a pointer is in C.",
    "What is dynamic memory allocation?",
    "Explain what a memory leak is.",
    "What is functional programming?",
    "Explain what a lambda function is.",
    "What is the difference between mutable and immutable data?",
    "Explain what a dictionary (key-value store) is in Python.",
    "What is list comprehension in Python?",
    "Explain what a try-except block does.",
    "What is the difference between == and is in Python?",
    "Explain what a module is in Python.",
    "What is the difference between a shallow copy and a deep copy?",
    "Explain what an abstract class is in OOP.",
]

PROMPTS_HUMANITIES = [
    "What is the bystander effect and what causes it?",
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
    "Explain what the Holocaust was and how it happened.",
    "What was the significance of the Berlin Wall?",
    "Explain what apartheid was in South Africa.",
    "What was the significance of the Russian Revolution?",
    "Explain what the Great Depression was and what caused it.",
    "What was the New Deal and what was it designed to do?",
    "Explain what imperialism is.",
    "What was the significance of the Enlightenment?",
    "Explain what the Reformation was and why it happened.",
    "What were the causes of the American Revolution?",
    "Explain what the Boston Tea Party was.",
    "What is the significance of the Declaration of Independence?",
    "Explain what the US Constitution is and why it was written.",
    "What is the Bill of Rights?",
    "Explain what the Indian independence movement was.",
    "What was the significance of Mahatma Gandhi?",
    "Explain what partition of India was and why it happened.",
    "What was the significance of Nelson Mandela?",
    "Explain what the Rwandan genocide was.",
    "What is globalisation and what are its effects?",
    "Explain what the United Nations does.",
    "What is NATO and why was it formed?",
    "Explain what the European Union is.",
    "What is democracy and how does it differ from authoritarianism?",
    "Explain what a dictatorship is and give a historical example.",
    "What is propaganda and how is it used?",
    "Explain what censorship is.",
    "What is human rights? Give examples of basic human rights.",
    "Explain what the Universal Declaration of Human Rights is.",
    "What is the difference between a refugee and an economic migrant?",
    "Explain what pushes and pull factors are in migration.",
    "What is urbanisation and what causes it?",
    "Explain what gentrification is.",
    "What is a GDP and what does it measure?",
    "Explain the difference between developed and developing countries.",
    "What is the Human Development Index?",
    "Explain what fair trade is.",
    "What is foreign aid and is it effective?",
    "Explain what the debt crisis in developing nations refers to.",
    "What is a supply chain?",
    "Explain what inflation is and what causes it.",
    "What is unemployment and what are its effects on society?",
    "Explain what a trade deficit is.",
    "What is free trade and what are its advantages and disadvantages?",
    "Explain what a tariff is.",
    "What is the World Trade Organization?",
    "Explain what the International Monetary Fund does.",
    "What is the World Bank?",
    "Explain what structural adjustment programmes are.",
    "What is the difference between a federal and a unitary state?",
    "Explain what devolution means in politics.",
    "What is the separation of powers?",
    "Explain what checks and balances are in government.",
    "What is judicial review?",
    "Explain the difference between civil law and criminal law.",
    "What is a constitution?",
    "Explain what a referendum is.",
    "What is proportional representation in voting systems?",
    "Explain what first-past-the-post voting is.",
    "What is political ideology? Explain left-wing and right-wing.",
    "Explain what liberalism is.",
    "What is conservatism?",
    "Explain what socialism is.",
    "What is nationalism?",
    "Explain what populism is.",
    "What is the difference between a primary and secondary source in history?",
    "Explain what historiography means.",
    "What is bias in historical sources and how do you identify it?",
    "Explain what oral history is.",
    "What is archaeology and what can it tell us about the past?",
    "Explain what the Silk Road was.",
    "What was the significance of the printing press?",
    "Explain what the Black Death was and its impact on Europe.",
    "What were the Crusades?",
    "Explain what the Roman Empire was and why it fell.",
    "What was the significance of ancient Greek democracy?",
    "Explain what the Ottoman Empire was.",
    "What was the Mughal Empire?",
    "Explain what the Ming Dynasty was known for.",
    "What was the significance of the Meiji Restoration in Japan?",
    "Explain what the Korean War was.",
    "What was the Vietnam War about?",
    "Explain what the Gulf War was.",
    "What is the Israeli-Palestinian conflict about?",
    "Explain what the Arab Spring was.",
    "What is terrorism and how do governments respond to it?",
    "Explain what nuclear deterrence is.",
    "What was the significance of the Nuremberg Trials?",
    "Explain what the Marshall Plan was and why it mattered.",
]

PROMPTS_METACOGNITION = [
    "Explain how to manage your time effectively during exam revision.",
    "Explain what metacognition means and why it helps learning.",
    "What is spaced repetition and why is it effective?",
    "Explain the Pomodoro technique.",
    "What is active recall?",
    "Explain interleaving in studying.",
    "What is the difference between surface and deep learning?",
    "Explain the growth versus fixed mindset.",
    "What is elaborative interrogation?",
    "Explain the Feynman technique.",
    "What is the difference between recognition and recall?",
    "Explain what retrieval practice is and why it works.",
    "What is the testing effect?",
    "Explain what desirable difficulties are in learning.",
    "What is the difference between massed and distributed practice?",
    "Explain what a learning objective is and how to write one.",
    "What is self-regulated learning?",
    "Explain what a study schedule is and how to build one.",
    "What is cognitive load and how can you reduce it while studying?",
    "Explain what the forgetting curve is.",
    "What is the spacing effect?",
    "Explain what mind mapping is and how it helps learning.",
    "What is the difference between summarising and paraphrasing?",
    "Explain what Cornell note-taking is.",
    "What is the SQ3R reading method?",
    "Explain what elaboration means in the context of studying.",
    "What is concrete examples as a learning strategy?",
    "Explain what dual coding is.",
    "What is the generation effect in memory?",
    "Explain what the primacy and recency effect are.",
    "What is working memory and how does it affect learning?",
    "Explain what long-term memory is and how information gets stored there.",
    "What is encoding in memory?",
    "Explain what retrieval cues are.",
    "What is context-dependent memory?",
    "Explain what state-dependent memory is.",
    "What is the difference between declarative and procedural memory?",
    "Explain what semantic memory is.",
    "What is episodic memory?",
    "Explain what automaticity means in skill learning.",
    "What is deliberate practice?",
    "Explain what a mental model is.",
    "What is transfer of learning?",
    "Explain what near and far transfer are.",
    "What is analogical reasoning?",
    "Explain what schema theory says about how we learn.",
    "What is constructivism in education?",
    "Explain what zone of proximal development means.",
    "What is scaffolding in education?",
    "Explain what formative and summative assessment are.",
    "What is the difference between intrinsic and extrinsic motivation?",
    "Explain what self-efficacy is.",
    "What is attribution theory in education?",
    "Explain what learned helplessness is.",
    "What is academic procrastination and how do you overcome it?",
    "Explain what test anxiety is and how to manage it.",
    "What is the difference between deep and shallow processing of information?",
    "Explain what chunking is in memory.",
    "What is a mnemonic device? Give an example.",
    "Explain what the method of loci is.",
    "What is a flashcard and how do you use it effectively?",
    "Explain what overlearning is and when it is useful.",
    "What is the difference between blocked and interleaved practice?",
    "Explain what feedback is in learning and why it matters.",
    "What is the difference between formative feedback and grades?",
    "Explain what peer learning is.",
    "What is collaborative learning?",
    "Explain what the jigsaw method of cooperative learning is.",
    "What is problem-based learning?",
    "Explain what project-based learning is.",
    "What is inquiry-based learning?",
    "Explain what flipped classroom means.",
    "What is personalised learning?",
    "Explain what differentiated instruction is.",
    "What is universal design for learning?",
    "Explain what a learning disability is and give an example.",
    "What is dyslexia?",
    "Explain what ADHD is and how it affects learning.",
    "What are learning styles and why is the evidence for them weak?",
    "Explain what neuromyths in education are.",
    "What is brain-based learning?",
    "Explain what sleep does for memory consolidation.",
    "What is the effect of exercise on learning and memory?",
    "Explain what stress does to memory and learning.",
    "What is mindfulness and how might it help students?",
    "Explain what grit is and how it relates to academic success.",
    "What is academic resilience?",
    "Explain what a growth mindset intervention looks like in school.",
    "What is the difference between effort and ability praise?",
    "Explain what mastery goals are versus performance goals.",
    "What is self-monitoring in studying?",
    "Explain what self-explanation is as a learning strategy.",
    "What is the PQRST study method?",
    "Explain what revision cards are and how to use them.",
    "What is the difference between skimming and scanning when reading?",
    "Explain what annotation is as a reading strategy.",
    "What is a graphic organiser?",
    "Explain what a KWL chart is.",
    "What is think-aloud as a learning strategy?",
    "Explain what metacognitive monitoring means.",
]

ALL_PROMPTS  = (
    PROMPTS_MATHEMATICS +
    PROMPTS_SCIENCE +
    PROMPTS_PROGRAMMING_CS +
    PROMPTS_HUMANITIES +
    PROMPTS_METACOGNITION
)

CATEGORIES = (
    ["Mathematics"]    * 100 +
    ["Science"]        * 100 +
    ["Programming-CS"] * 100 +
    ["Humanities"]     * 100 +
    ["Meta-cognition"] * 100
)

assert len(ALL_PROMPTS) == 500, f"Expected 500 prompts, got {len(ALL_PROMPTS)}"
assert len(CATEGORIES)  == 500

# ==============================================================================
# ── TOKENIZER ─────────────────────────────────────────────────────────────────
# ==============================================================================

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==============================================================================
# ── MODEL ─────────────────────────────────────────────────────────────────────
# ==============================================================================

print("Loading model...")

# ── Load model — rope_scaling fix ────────────────────────────────────────────
# The config for Phi-3-mini uses rope_scaling with key "rope_type" but the
# downloaded modeling_phi3.py expects "type". Nulling rope_scaling tells the
# model to use standard unscaled RoPE, which is correct for the 4k context
# version of this model.

from transformers import AutoConfig as _AC
_cfg = _AC.from_pretrained(MODEL_ID, trust_remote_code=True)
_cfg.rope_scaling = None   # disable — not needed for 4k native context

if USE_QUANTIZATION:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=_cfg,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=_cfg,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

model.eval()
print(f"Model loaded — {PRECISION_LABEL}")

# ==============================================================================
# ── IDLE POWER BASELINE ───────────────────────────────────────────────────────
# ==============================================================================

print("\nMeasuring idle power baseline (10 seconds)...")
idle_tracker = EmissionsTracker(
    measure_power_secs=2,
    save_to_file=False,
    log_level="error"
)
idle_tracker.start()
if DEVICE == "cuda":
    torch.cuda.synchronize()
time.sleep(10)
idle_tracker.stop()

idle_energy_obj = idle_tracker._total_energy
idle_watts = (idle_energy_obj.kWh * 3.6e6) / 10.0 if idle_energy_obj else 0.0
print(f"Idle power: {idle_watts:.2f} W")

# ==============================================================================
# ── SINGLE SHARED TRACKER — avoids per-prompt tracker drift ───────────────────
# ==============================================================================
# Using one tracker with manual energy snapshots per prompt.
# This matches the approach in kvcache_50prompts_true.csv and avoids
# the measurement noise that came from spinning up a new tracker each time.

main_tracker = EmissionsTracker(
    project_name=f"phi3_{PRECISION_LABEL}_500prompts",
    measure_power_secs=1,
    save_to_file=False,
    log_level="error"
)
main_tracker.start()

# ==============================================================================
# ── INFERENCE LOOP ────────────────────────────────────────────────────────────
# ==============================================================================

results = []

print(f"\nRunning 500 prompts — {PRECISION_LABEL} | use_cache=True\n")
print(f"{'ID':>4} {'Category':<16} {'Latency':>8} {'Net_J':>8} {'Tok/s':>7}")
print("-" * 50)

for idx, (prompt, category) in enumerate(zip(ALL_PROMPTS, CATEGORIES)):
    task_id = idx + 1

    # Energy snapshot before
    main_tracker._measure_power_and_energy()
    e_before_kwh = main_tracker._total_energy.kWh

    inputs   = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,            # ← KV-cache ON — standard inference
            do_sample=False,           # deterministic — same as original study
            temperature=None,          # must be None when do_sample=False
            top_p=None,                # must be None when do_sample=False
            pad_token_id=tokenizer.eos_token_id,
        )

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    latency = time.time() - t0

    # Energy snapshot after
    main_tracker._measure_power_and_energy()
    e_after_kwh = main_tracker._total_energy.kWh

    # Decode response
    output_ids     = outputs[0][input_len:]
    response_text  = tokenizer.decode(output_ids, skip_special_tokens=True)
    tokens_out     = len(output_ids)
    tokens_per_sec = tokens_out / latency if latency > 0 else 0.0

    # Energy accounting
    gross_j = (e_after_kwh - e_before_kwh) * 3.6e6
    net_j   = max(gross_j - idle_watts * latency, 0.01)
    power_w = net_j / latency if latency > 0 else 0.0

    results.append({
        "ID":           task_id,
        "Precision":    PRECISION_LABEL,
        "Category":     category,
        "Prompt":       prompt,
        "Response":     response_text,
        "Input_Tokens": int(input_len),
        "Output_Tokens": int(tokens_out),
        "Latency_s":    round(latency, 4),
        "Tokens_per_sec": round(tokens_per_sec, 2),
        "Gross_Energy_J": round(gross_j, 4),
        "Net_Energy_J": round(net_j, 4),
        "Power_W":      round(power_w, 2),
        "use_cache":    True,
    })

    print(f"{task_id:>4} {category:<16} {latency:>7.2f}s {net_j:>8.1f}J {tokens_per_sec:>6.1f}")

main_tracker.stop()

# ==============================================================================
# ── RESULTS ───────────────────────────────────────────────────────────────────
# ==============================================================================

df = pd.DataFrame(results)

print("\n" + "=" * 60)
print(f"  RESULTS — {PRECISION_LABEL} | use_cache=True | n=500")
print("=" * 60)
print(f"  Avg Latency     : {df.Latency_s.mean():.2f}s")
print(f"  Avg Net Energy  : {df.Net_Energy_J.mean():.1f} J")
print(f"  Avg Power       : {df.Power_W.mean():.1f} W")
print(f"  Avg Tokens/sec  : {df.Tokens_per_sec.mean():.1f}")
print(f"  Total prompts   : {len(df)}")
print("=" * 60)

print("\nPer-category breakdown:")
summary = df.groupby("Category").agg(
    avg_latency_s   = ("Latency_s",    "mean"),
    avg_net_energy_J= ("Net_Energy_J", "mean"),
    avg_power_W     = ("Power_W",      "mean"),
    avg_tokens_per_s= ("Tokens_per_sec","mean"),
).round(2)
print(summary.to_string())

# Save
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved: {OUTPUT_FILE}")

#from google.colab import files
#import pandas as pd

#df = pd.DataFrame(results)
#df.to_csv("phi3_FP16_corrected_500prompts.csv", index=False, encoding='utf-8-sig')
#files.download("phi3_FP16_corrected_500prompts.csv")
#print(f"Downloaded {len(df)} rows")

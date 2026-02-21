# ==============================================================================
#  RESEARCH: THE GREEN LEARNING AUDIT 
# ==============================================================================

!pip uninstall -y transformers -q
!pip install -q "transformers>=4.44.0" accelerate bitsandbytes codecarbon anthropic

import time
import torch
import pandas as pd
import anthropic

from transformers.cache_utils import DynamicCache
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
)
from codecarbon import EmissionsTracker
from google.colab import files

# Cache compat shim
if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
MODEL_ID        = "microsoft/Phi-3-mini-4k-instruct"
USE_QUANTIZATION = False          # Toggle for FP16 vs NF4 runs
MAX_NEW_TOKENS  = 200             # Sufficient for scaffolded explanation
OUTPUT_FILE     = "green_audit_results.csv"
ANTHROPIC_API_KEY = "YOUR_KEY_HERE"  # For auto Qped scoring
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# ---------------------------------------------------------
# PROMPTS — paste your full 500 here, or use a subset
# ---------------------------------------------------------
PROMPTS = [
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
    "How do you simplify rational expressions? Walk through simplifying (x² - 4)/(x - 2).",
    "Explain arithmetic sequences: what they are, their formula, and an example.",
    "What is a geometric sequence and how do you find the sum of the first n terms?",
    "Explain the concept of domain and range of a function using a simple example.",
    "How do you solve an inequality and represent the solution on a number line?",
    "What is completing the square, and when would you use it?",
    "Explain the relationship between zeros of a polynomial and its graph.",
    "What is Pascal's triangle and how does it connect to binomial expansion?",
    "How do you graph a parabola? What does the vertex form of a quadratic tell you?",
    "Explain what a polynomial is and how to perform polynomial long division.",
    "What is the Pythagorean theorem and how do you use it to find the length of a missing side?",
    "Explain the difference between similar and congruent triangles with examples.",
    "What are the properties of a parallelogram? How is a rectangle a special case?",
    "How do you calculate the area of a circle and explain why the formula is πr²?",
    "Explain what a tangent line to a circle is and the relationship between the radius and the tangent.",
    "What is the difference between perimeter and area? Why do they have different units?",
    "How do you find the volume and surface area of a rectangular prism?",
    "Explain the triangle inequality theorem and why it matters.",
    "What is the angle sum property of a triangle, and how does it extend to polygons?",
    "Explain what congruent triangles are and the criteria for proving congruence (SSS, SAS, ASA).",
    "What is the midpoint formula and how do you find the midpoint of a line segment?",
    "Explain the concept of transformations: translation, rotation, reflection, and dilation.",
    "How do you find the distance between two points using the distance formula?",
    "What is a regular polygon, and how do you calculate its interior angles?",
    "Explain what the circumference of a circle is and how it relates to diameter and π.",
    "What is the Pythagorean triple and give three examples?",
    "How do you calculate the area of a triangle using different methods?",
    "Explain what a transversal is and the angle relationships it creates with parallel lines.",
    "What is coordinate geometry, and how do you prove a quadrilateral is a rectangle?",
    "Explain what a geometric proof is and why it is important in mathematics.",
    "Explain the three basic trigonometric ratios: sine, cosine, and tangent, in terms of a right triangle.",
    "What is the unit circle and why is it important in trigonometry?",
    "Explain what it means for two angles to be complementary and how their trig functions relate.",
    "How do you use the law of sines to find a missing side or angle in a triangle?",
    "What is the law of cosines and when would you use it instead of the law of sines?",
    "Explain radian measure and how to convert between degrees and radians.",
    "What are the graphs of sin and cos? Describe their amplitude, period, and key features.",
    "Explain the Pythagorean identity sin²θ + cos²θ = 1 and how to derive it.",
    "What is an inverse trig function and what does arcsin(0.5) mean?",
    "Explain how trigonometry is used to calculate the height of a tall building using angle of elevation.",
    "What is the difference between mean, median, and mode? When is each most useful?",
    "Explain what standard deviation measures and how a large vs small standard deviation affects a dataset.",
    "What is the difference between a population and a sample in statistics?",
    "Explain the concept of probability: what it is and how you calculate the probability of rolling a 4 on a die.",
    "What is a normal distribution and what does the 68-95-99.7 rule mean?",
    "Explain the difference between independent and dependent events in probability.",
    "What is a scatter plot and how do you interpret correlation from one?",
    "Explain what a box plot shows and how to read quartiles from it.",
    "What is the difference between permutations and combinations? Give an example of each.",
    "How do you calculate conditional probability and explain Bayes' theorem simply?",
    "What is a limit in calculus and how do you evaluate lim(x→2) of (x² - 4)/(x - 2)?",
    "Explain what a derivative represents geometrically and physically.",
    "What is the power rule for differentiation and how do you use it?",
    "Explain what an integral represents geometrically using a graph.",
    "What is the fundamental theorem of calculus in simple terms?",
    "How do you find the maximum or minimum of a function using derivatives?",
    "What is a rate of change and how does it connect to everyday life?",
    "Explain what continuity of a function means and give an example of a discontinuous function.",
    "What is the chain rule in differentiation and when do you need it?",
    "How does integration relate to the area under a curve?",
    "What is a prime number and how do you determine if a number is prime?",
    "Explain the difference between rational and irrational numbers with examples.",
    "What is the greatest common factor (GCF) and how do you find it using prime factorization?",
    "Explain what modular arithmetic is with a real-world clock analogy.",
    "What is a complex number and what does the imaginary unit i represent?",
    "Explain what a vector is and how it differs from a scalar quantity.",
    "What is a matrix and how do you multiply two 2×2 matrices together?",
    "Explain what the Fibonacci sequence is and where it appears in nature.",
    "What is mathematical induction and how would you use it to prove a formula?",
    "What is the difference between a proof by contradiction and a direct proof?",
    "Explain what an asymptote is and why some functions never reach certain values.",
    "What is a piecewise function and how do you graph one?",
    "Explain the concept of infinity in mathematics: are all infinities equal?",
    "What is the difference between discrete and continuous data?",
    "How do you convert between fractions, decimals, and percentages?",
    "Explain what significant figures are and why they matter in calculations.",
    "What is scientific notation and how do you multiply two numbers in scientific notation?",
    "Explain what a recursive formula is with a simple example.",
    "What is the relationship between a function and its inverse? How do you find an inverse?",
    "Explain the concept of proportional reasoning and give a real-world example.",
    "What is dimensional analysis and how is it used to convert units?",
    "How do you solve a word problem systematically? Walk through a worked example.",
    "What is the difference between an equation and an expression?",
    "Explain what error analysis is and why it is important in mathematical modeling.",
    "What is a Venn diagram and how does it represent set operations like union and intersection?",
    "Explain the order of operations (PEMDAS/BODMAS) with a tricky example.",
    "What is the difference between deductive and inductive reasoning in mathematics?",
    "How do you approach an unfamiliar math problem? What strategies do you use?",
    "Explain what a slope-intercept form of a line tells you and how to graph it.",
    "What is the quadratic formula and how is it derived by completing the square?",

    # SCIENCE
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
    "Explain what enzymes are and how temperature affects their function.",
    "What is the difference between DNA and RNA?",
    "Explain what a gene is and how it codes for a protein.",
    "What is Mendelian genetics and explain the difference between dominant and recessive traits.",
    "Explain what osmosis is and give a real-life example.",
    "What is the difference between an autotroph and a heterotroph?",
    "Explain the carbon cycle and the role of decomposers.",
    "What is homeostasis and give an example of how the body maintains it.",
    "Explain the structure and function of the human nervous system.",
    "What is the immune system's response to a pathogen like a virus?",
    "Explain what an atom is and describe its basic structure.",
    "What is the periodic table and how are elements organized in it?",
    "Explain what a chemical bond is and the difference between ionic and covalent bonds.",
    "What is the difference between an element, a compound, and a mixture?",
    "Explain what happens during a chemical reaction and what the law of conservation of mass means.",
    "What is an acid and a base, and how does the pH scale measure them?",
    "Explain what oxidation and reduction mean and give an everyday example.",
    "What is the mole concept and why is it useful in chemistry?",
    "Explain what electronegativity is and how it affects bond polarity.",
    "What is stoichiometry and how do you use it to calculate reactant quantities?",
    "Explain what intermolecular forces are and how they affect boiling points.",
    "What is a solution, a solute, and a solvent? Give an example.",
    "Explain what entropy means in thermodynamics in simple terms.",
    "What is Le Chatelier's principle and how does it apply to chemical equilibrium?",
    "Explain what isotopes are and why carbon-14 is useful for dating ancient objects.",
    "What is nuclear fission and how does it differ from nuclear fusion?",
    "Explain what a catalyst does in a chemical reaction.",
    "What are hydrocarbons and give examples of their uses?",
    "Explain what the ideal gas law states and what each variable represents.",
    "What is the difference between exothermic and endothermic reactions?",
    "Explain Newton's three laws of motion with everyday examples for each.",
    "What is the difference between speed and velocity?",
    "Explain what momentum is and the law of conservation of momentum.",
    "What is kinetic energy and potential energy, and how do they convert between each other?",
    "Explain what work and power mean in physics with formulas.",
    "What is gravity and how does the gravitational force between two objects depend on their masses?",
    "Explain what a wave is and the difference between transverse and longitudinal waves.",
    "What is the electromagnetic spectrum and give examples of each type of radiation?",
    "Explain what electric current, voltage, and resistance are, and how Ohm's law relates them.",
    "What is the difference between series and parallel circuits?",
    "Explain what friction is and the difference between static and kinetic friction.",
    "What is the Doppler effect and give a real-world example?",
    "Explain what refraction is and why a straw appears bent in a glass of water.",
    "What is the difference between reflection and refraction of light?",
    "Explain what sound is and why it cannot travel through a vacuum.",
    "What is thermal energy and how does it transfer by conduction, convection, and radiation?",
    "Explain what acceleration is and how to calculate it.",
    "What is projectile motion and what two components of motion act independently?",
    "Explain what pressure is and how depth affects water pressure.",
    "What is centripetal acceleration and what causes it in circular motion?",
    "Explain what plate tectonics is and how it explains earthquakes and volcanoes.",
    "What is the rock cycle and how do igneous, sedimentary, and metamorphic rocks form?",
    "Explain what the greenhouse effect is and how it warms the Earth.",
    "What causes the seasons, and why is it not because Earth is closer to the Sun?",
    "Explain what a star is and describe the life cycle of a star like our Sun.",
    "What is the difference between a solar eclipse and a lunar eclipse?",
    "Explain what the water cycle is and describe each stage.",
    "What causes ocean tides and how does the Moon influence them?",
    "Explain what a black hole is and how one forms.",
    "What is the Big Bang theory and what evidence supports it?",
    "Explain what biodiversity is and why it is important for ecosystems.",
    "What is the difference between renewable and non-renewable energy sources?",
    "Explain what climate change means and the difference between climate and weather.",
    "What is the ozone layer and why is it important for life on Earth?",
    "Explain what eutrophication is and how excess fertilizer causes it.",
    "What are the main causes of deforestation and its effects on the environment?",
    "Explain what a food chain and food web are, using an ecosystem example.",
    "What is the nitrogen cycle and why is nitrogen important for life?",
    "Explain what ocean acidification is and how it harms marine ecosystems.",
    "What is the difference between adaptation and acclimatization?",
    "Explain the scientific method step by step with an example experiment.",
    "What is the difference between a hypothesis and a theory in science?",
    "Explain what a control group and an experimental group are in an experiment.",
    "What is peer review and why is it important in science?",
    "Explain the difference between correlation and causation with an example.",
    "What is experimental error and how do scientists minimize it?",
    "Explain what a variable is in an experiment: independent, dependent, and controlled.",
    "What does it mean for scientific results to be reproducible?",
    "Explain what a model is in science and give an example of a scientific model.",
    "What is the difference between qualitative and quantitative data in an experiment?",
    "Explain what a hypothesis is and how it is different from a prediction.",
    "What is the difference between a physical and chemical change? Give an example of each.",
    "Explain what potential energy and kinetic energy are, using a rollercoaster as an example.",
    "What is the difference between speed and velocity? Give an example.",
    "Explain what a food web is and how removing one species can affect the whole web.",
    "What is the difference between aerobic and anaerobic respiration?",
    "Explain what a mutation is and how it can be neutral, harmful, or beneficial.",
    "What is the role of ribosomes in protein synthesis?",
    "Explain how vaccines work to protect against disease.",
    "What is the difference between a conductor and an insulator? Give examples.",

    # PROGRAMMING / CS
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
    "What is a data type and give examples: integer, string, float, boolean.",
    "Explain what a conditional statement is and how it controls program flow.",
    "What is a dictionary (or hash map) in programming and when would you use one?",
    "Explain what pseudocode is and why programmers write it before coding.",
    "What is debugging and what strategies do programmers use to find bugs?",
    "Explain what a string is and how to concatenate two strings.",
    "What is the difference between passing by value and passing by reference?",
    "Explain what scope means in programming: local vs global variables.",
    "What is an algorithm and can you explain bubble sort in simple steps?",
    "Explain what comments in code are for and why they matter.",
    "What is object-oriented programming (OOP)? Explain using a car as an analogy.",
    "Explain what a class and an object are in OOP.",
    "What is inheritance in OOP and give an example using animals.",
    "Explain what encapsulation means in programming.",
    "What is polymorphism and give a real-world analogy for it?",
    "Explain what a constructor is in a class.",
    "What is the difference between a method and a function?",
    "Explain what abstraction means in OOP with an example.",
    "What is an interface in programming and how does it differ from a class?",
    "Explain the concept of overloading and overriding in OOP.",
    "What is a stack data structure and explain LIFO with a stack of plates analogy.",
    "What is a queue and how does it differ from a stack? Use a real-world example.",
    "Explain what a linked list is and how it differs from an array.",
    "What is a binary tree and explain what root, leaf, and node mean.",
    "Explain what a graph data structure is and give a real-world example.",
    "What is a hash table and why is it useful for fast lookups?",
    "Explain the difference between a tree and a graph.",
    "What is a heap data structure and when would you use one?",
    "Explain what a binary search tree is and how you insert a value into one.",
    "What is the difference between depth-first and breadth-first search?",
    "Explain what Big O notation means and why O(n²) is slower than O(n log n).",
    "Walk me through binary search: how it works and why it is faster than linear search.",
    "Explain merge sort step by step using a simple example.",
    "What is a greedy algorithm and give an example of when it works well?",
    "Explain what dynamic programming is with the example of the Fibonacci sequence.",
    "What is the difference between a divide and conquer approach and brute force?",
    "Explain what a sorting algorithm does and compare insertion sort and selection sort.",
    "What is time complexity and space complexity?",
    "Explain the traveling salesman problem in simple terms.",
    "What is a heuristic in computer science and when do we use heuristics?",
    "Explain what the internet is and how data travels from one computer to another.",
    "What is the difference between the internet and the World Wide Web?",
    "Explain what HTML, CSS, and JavaScript each do in a webpage.",
    "What is a URL and explain what each part of https://www.example.com/page means.",
    "Explain what HTTP and HTTPS are and why HTTPS is more secure.",
    "What is a server and how does it respond to a browser's request?",
    "Explain what a database is and the difference between SQL and NoSQL databases.",
    "What is an API and explain it using the analogy of a restaurant menu.",
    "Explain what cookies are in a web browser and why websites use them.",
    "What is the difference between front-end and back-end web development?",
    "What is encryption and explain it using a simple substitution cipher example.",
    "Explain what a phishing attack is and how to spot one.",
    "What is a firewall and what does it protect a network from?",
    "Explain what two-factor authentication (2FA) is and why it improves security.",
    "What is malware and explain the difference between a virus and a Trojan horse?",
    "Explain what a strong password is and why 'password123' is a bad one.",
    "What is open-source software and how does it differ from proprietary software?",
    "Explain what digital privacy means and why it matters.",
    "What is cyberbullying and what are the digital ethics around online behaviour?",
    "Explain what the digital divide is and its consequences for global education.",
    "Explain what a CPU does and use the analogy of a brain.",
    "What is RAM and how does it differ from storage (hard drive/SSD)?",
    "Explain what an operating system is and give three examples.",
    "What is the difference between hardware and software?",
    "Explain what binary is and convert the decimal number 13 to binary.",
    "What is Moore's Law and is it still relevant today?",
    "Explain what a GPU is and why it is useful for AI and gaming.",
    "What is the difference between 32-bit and 64-bit operating systems?",
    "Explain what cloud computing is and give examples of cloud services.",
    "What is an input device and an output device? Give two examples of each.",
    "Explain what machine learning is in simple terms using an email spam filter example.",
    "What is the difference between supervised and unsupervised learning?",
    "Explain what a neural network is using the analogy of how the brain works.",
    "What is artificial intelligence and what are its potential benefits and risks?",
    "Explain what a large language model (LLM) is and how it generates text.",
    "What is the difference between AI and automation?",
    "Explain what data privacy means in the context of AI training data.",
    "What is robotics and how do sensors help robots interact with the world?",
    "Explain what augmented reality (AR) and virtual reality (VR) are.",
    "What is blockchain technology and how does it work at a basic level?",
    "Explain what version control is and why developers use tools like Git.",
    "What is the difference between a compiled language and an interpreted language?",
    "Explain what an IDE (Integrated Development Environment) is and its key features.",
    "What is the difference between a local variable and a global variable?",
    "Explain what exception handling is and why it is important in programs.",
    "What is the difference between a class method and an instance method?",
    "Explain what refactoring means in software development.",
    "What is a software library and how does it differ from a framework?",
    "Explain what unit testing is and why developers write tests.",
    "What is the difference between a shallow copy and a deep copy of an object?",

    # HUMANITIES
    "Explain the main causes of World War I using the MAIN acronym (Militarism, Alliances, Imperialism, Nationalism).",
    "What was the significance of the French Revolution and what were its key outcomes?",
    "Explain what the Industrial Revolution was and how it changed society.",
    "What was the Cold War and what were the main tensions between the USA and USSR?",
    "Explain the causes and consequences of the Atlantic Slave Trade.",
    "What was the significance of the Magna Carta in the development of democracy?",
    "Explain what colonialism is and how it affected Africa and Asia.",
    "What were the main causes of World War II?",
    "Explain the importance of the Renaissance period in European history.",
    "What was the significance of the Civil Rights Movement in the USA?",
    "Explain what the Holocaust was and why it is studied in history education.",
    "What was the Enlightenment and how did it influence modern government?",
    "Explain the causes of the American Revolution and what principles it was founded on.",
    "What was the significance of the printing press in spreading knowledge?",
    "Explain what imperialism is and how it shaped the modern world.",
    "What was apartheid and how did it end in South Africa?",
    "Explain the significance of the Silk Road in world history.",
    "What were the main events and significance of the Russian Revolution of 1917?",
    "Explain what the United Nations does and why it was created.",
    "What is the significance of the fall of the Berlin Wall in 1989?",
    "Explain what human geography and physical geography study and how they differ.",
    "What are the main factors that influence population distribution across the world?",
    "Explain what globalisation is and give examples of how it affects daily life.",
    "What is the difference between a developed and a developing country?",
    "Explain what urbanisation is and the challenges it creates in cities.",
    "What is the difference between a refugee and an economic migrant?",
    "Explain what the Human Development Index (HDI) measures.",
    "What causes famine and what are the human and natural factors involved?",
    "Explain what a trade balance is and what happens when a country runs a deficit.",
    "What is sustainable development and why is it important for the future?",
    "Explain the law of supply and demand using the example of concert tickets.",
    "What is inflation and how does it affect the purchasing power of money?",
    "Explain what GDP (Gross Domestic Product) measures and its limitations.",
    "What is the difference between a free market economy and a planned economy?",
    "Explain what unemployment is and the difference between structural and cyclical unemployment.",
    "What is a budget deficit and why might a government run one?",
    "Explain what opportunity cost means using a real-life example.",
    "What is the difference between fiscal policy and monetary policy?",
    "Explain what a tariff is and how it affects international trade.",
    "What is compound interest and why is it powerful over long time periods?",
    "Explain what a metaphor is and give three examples from everyday language.",
    "What is the difference between a simile and a metaphor?",
    "Explain what theme means in literature and identify the theme of a story you know.",
    "What is dramatic irony and give an example from a play or film?",
    "Explain the difference between first-person and third-person narrative perspective.",
    "What is imagery in literature and how does it affect the reader?",
    "Explain what satire is and give a well-known example of satirical writing.",
    "What is a sonnet and what are its structural rules?",
    "Explain what foreshadowing is and why authors use it.",
    "What is the difference between denotation and connotation in language?",
    "Explain what alliteration is and give a memorable example.",
    "What is an unreliable narrator and why do authors use this technique?",
    "Explain what the hero's journey narrative structure is.",
    "What is the difference between prose and poetry?",
    "Explain what characterization is and the difference between direct and indirect characterization.",
    "Explain the difference between ethics and morality.",
    "What is utilitarianism and what is the main criticism of this ethical theory?",
    "Explain what Kant's categorical imperative means in simple terms.",
    "What is the trolley problem and what does it reveal about moral decision-making?",
    "Explain what epistemology is: the study of knowledge and how we know what we know.",
    "What is the difference between deductive and inductive reasoning?",
    "Explain what Plato's allegory of the cave means.",
    "What is the social contract theory and which philosophers developed it?",
    "Explain what confirmation bias is and give an example.",
    "What is the difference between an argument and an opinion in philosophy?",
    "Explain what cognitive dissonance is with a real-life example.",
    "What is Maslow's hierarchy of needs and how does it explain motivation?",
    "Explain what the nature vs nurture debate is about.",
    "What is the difference between classical and operant conditioning?",
    "Explain what social conformity is and describe the Asch conformity experiment.",
    "What is the bystander effect and what causes it?",
    "Explain what a stereotype is and how it differs from a prejudice.",
    "What is social stratification and what factors determine class?",
    "Explain what culture is and give examples of cultural differences.",
    "What is critical thinking and why is it an important skill?",
    "Explain what media literacy means and why it is important today.",
    "What is propaganda and how do you identify it in media?",
    "Explain the difference between primary and secondary sources.",
    "What is bias in reporting and give an example of how the same event can be framed differently?",
    "Explain what fake news is and give three strategies to fact-check information.",
    "What is the role of a free press in a democratic society?",
    "Explain what an argument structure is: claim, evidence, and warrant.",
    "What is the difference between persuasion and manipulation?",
    "Explain what digital citizenship means and its responsibilities.",
    "What is the difference between copyright and plagiarism?",
    "Explain what the Universal Declaration of Human Rights is and why it was created.",
    "What is the difference between a democracy and an autocracy?",
    "Explain what the Reformation was and how it changed Europe.",
    "What is the significance of the Gutenberg printing press in history?",
    "Explain what globalisation means for culture and give an example of cultural diffusion.",
    "What is the difference between immigration and emigration?",
    "Explain what a census is and why governments conduct them.",
    "What is the role of the United Nations Security Council?",
    "Explain what the term 'checks and balances' means in government.",
    "What is the difference between a civil law and a criminal law?",
    "Explain what the term 'sovereignty' means in international relations.",
    "What is the difference between a political party and a pressure group?",
    "Explain what the Enlightenment thinkers meant by 'natural rights'.",
    "What is the difference between a tariff and a quota in trade policy?",
    "Explain what a monarchy is and the difference between absolute and constitutional monarchies.",

    # META-COGNITION
    "Explain what metacognition means and why thinking about your own thinking helps you learn better.",
    "What is spaced repetition and why is it more effective than cramming?",
    "Explain the Pomodoro technique for studying and why it works.",
    "What is active recall and how does it differ from passive re-reading?",
    "Explain what interleaving is in studying and why mixing topics helps retention.",
    "What is the difference between surface learning and deep learning?",
    "Explain what a growth mindset is and how it differs from a fixed mindset.",
    "What is elaborative interrogation as a study technique?",
    "Explain the Feynman technique for learning: how does teaching a concept help you understand it?",
    "What is the difference between recognition and recall in memory?",
    "Explain what working memory is and how it affects how much information you can hold at once.",
    "What is long-term potentiation and how does sleep help consolidate learning?",
    "Explain what chunking is in memory and give an example of how you use it.",
    "What is a mind map and how can it help you organize information?",
    "Explain what retrieval practice is and why tests can improve learning.",
    "What is the spacing effect and how should you plan a revision schedule using it?",
    "Explain the concept of cognitive load and how it affects learning new material.",
    "What is deliberate practice and how does it differ from just repeating a task?",
    "Explain what prior knowledge is and how it helps you learn new information faster.",
    "What is the generation effect in learning?",
    "Explain what the scientific method has in common with everyday problem solving.",
    "What are heuristics and biases and give an example of each.",
    "Explain what systems thinking is and give an example of a complex system.",
    "What is a mental model and how do models help you reason about the world?",
    "Explain what first-principles thinking is with an example.",
    "What is the difference between convergent and divergent thinking?",
    "Explain what brainstorming is and what rules make it effective.",
    "What is a cognitive bias and explain the availability heuristic.",
    "Explain what Occam's razor means and give an example of applying it.",
    "What is the difference between a problem and a dilemma?",
    "Explain what lateral thinking is and give a classic lateral thinking puzzle.",
    "What is decision-making under uncertainty and what strategies help?",
    "Explain what root cause analysis is and how it helps solve problems.",
    "What is the difference between analysis and synthesis?",
    "Explain what creative thinking is and how constraints can actually boost creativity.",
    "What is analogical reasoning and give an example of how it helps learning?",
    "Explain what transfer of learning is and why it is the ultimate goal of education.",
    "What is metacognitive monitoring and how do you check whether you actually understand something?",
    "Explain what the Socratic method is and how questioning deepens understanding.",
    "What is counterfactual thinking and how does 'what if' reasoning help learning?",
    "Explain how to write a strong thesis statement for an essay.",
    "What is the difference between summarizing and paraphrasing?",
    "Explain how to read a scientific article efficiently using SQ3R.",
    "What is annotating a text and why does it improve comprehension?",
    "Explain what note-taking systems like Cornell Notes are and how they help.",
    "What is a study group and what makes one effective vs ineffective?",
    "Explain how to manage your time effectively during exam revision.",
    "What is academic procrastination and what strategies help overcome it?",
    "Explain the importance of sleep for academic performance.",
    "What is exam anxiety and give three evidence-based strategies to manage it?",
    "Explain the difference between skimming and scanning a text.",
    "What makes a good essay structure? Explain introduction, body, and conclusion.",
    "Explain what citing sources means and why academic integrity matters.",
    "What is the difference between primary and secondary research?",
    "Explain what a research question is and what makes one good or weak.",
    "What is plagiarism and why is it taken seriously in academic work?",
    "Explain how to give constructive feedback on a peer's work.",
    "What is self-regulated learning and how do successful students use it?",
    "Explain what reflective learning is and how keeping a journal can improve it.",
    "What is the difference between effort and strategy in academic success?",
    "Explain the difference between intrinsic and extrinsic motivation.",
    "What is self-efficacy and how does believing in your ability affect performance?",
    "Explain what goal-setting theory says about effective goals (SMART goals).",
    "What is learned helplessness and how can you overcome it?",
    "Explain what emotional regulation is and why it matters for learning.",
    "What is the difference between stress and burnout and how do you avoid burnout?",
    "Explain what mindfulness is and how it can improve focus during study.",
    "What is the role of curiosity in learning and how do you cultivate it?",
    "Explain what failure tolerance means and why making mistakes is important for growth.",
    "What is the imposter syndrome and how do students manage it?",
    "Explain what positive self-talk is and how it affects confidence.",
    "What is the value of asking for help and what prevents students from doing so?",
    "Explain what social learning theory says about learning from others.",
    "What is the difference between exams as evaluation and exams as learning tools?",
    "Explain what academic resilience is and how to build it.",
    "Explain how AI tutors like chatbots differ from human teachers.",
    "What are the benefits and risks of using AI to help you learn?",
    "Explain what critical AI literacy means: how do you evaluate AI-generated explanations?",
    "What is the difference between using AI as a crutch and using it as a scaffold?",
    "Explain how to use feedback from an AI tutor effectively.",
    "What does it mean to verify information from an AI and why is it important?",
    "Explain what adaptive learning systems are and how they personalize education.",
    "What is the difference between memorizing facts and developing understanding?",
    "Explain why learning how to learn is the most important skill for the future.",
    "What does lifelong learning mean and why is it important in a fast-changing world?",
    "Explain what digital literacy means and why it is a core 21st-century skill.",
    "What is the difference between consuming information and creating knowledge?",
    "Explain how to set learning goals effectively and track your own progress.",
    "What is the role of reflection in deep learning?",
    "Explain how collaboration and peer learning differ from individual study.",
    "Explain what the zone of proximal development (ZPD) is and how a tutor can use it.",
    "What is the difference between knowing how to do something and knowing why it works?",
    "Explain what attention is and why it is essential for learning.",
    "What does it mean to be an independent learner?",
    "Explain what formative and summative assessment are and how they differ.",
    "What is the difference between rote memorisation and meaningful learning?",
    "Explain what a learning objective is and why it helps focus study sessions.",
    "What is the Dunning-Kruger effect and what does it tell us about self-assessment?",
    "Explain what academic vocabulary is and why building it matters across subjects.",
    "What does it mean to 'unpack' a question before answering it in an exam?",

    
]

# ---------------------------------------------------------
# AUTO QPED SCORER
# ---------------------------------------------------------
SCORING_SYSTEM_PROMPT = """You are an expert educational evaluator assessing AI tutor responses.

Score the response on a scale of 1-10 using this rubric:
- 9-10: Correct, clear, age-appropriate, includes analogy or example, well scaffolded
- 7-8:  Mostly correct, clear, minor omissions or slightly unclear
- 5-6:  Partially correct, some confusion or missing key ideas
- 3-4:  Mostly incorrect or unclear, but some relevant content present
- 1-2:  Incorrect, irrelevant, or incomprehensible

Respond with ONLY a JSON object in this exact format:
{"score": 8, "reason": "one sentence explanation"}"""

def score_response(prompt: str, response: str, client: anthropic.Anthropic) -> tuple[int, str]:
    """Auto-score a response using Claude. Returns (score, reason)."""
    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",  # Fast and cheap for scoring
            max_tokens=100,
            system=SCORING_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"PROMPT: {prompt}\n\nAI RESPONSE: {response}"
            }]
        )
        import json
        text = message.content[0].text.strip()
        parsed = json.loads(text)
        return int(parsed["score"]), parsed.get("reason", "")
    except Exception as e:
        print(f"  Scoring error: {e}")
        return 7, "scoring_failed"  # Conservative fallback

# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------
print("\n--- Loading Model ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
config.rope_scaling = None

if USE_QUANTIZATION and DEVICE == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, config=config, quantization_config=bnb_config,
        trust_remote_code=True, device_map={"": 0},
        low_cpu_mem_usage=True, attn_implementation="eager",
    )
    precision_label = "NF4"
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, config=config, torch_dtype=torch.float16,
        trust_remote_code=True, device_map={"": 0},
        attn_implementation="eager",
    )
    precision_label = "FP16"

model.eval()
device = next(model.parameters()).device
print(f"Model loaded in {precision_label} on {device}")

# ---------------------------------------------------------
# IDLE BASELINE
# ---------------------------------------------------------
print("\n[1] Measuring idle power...")
baseline_tracker = EmissionsTracker(
    measure_power_secs=2.0, save_to_file=False, log_level="error"
)
baseline_tracker.start()
time.sleep(10)
baseline_tracker.stop()

idle_energy = getattr(baseline_tracker, "_total_energy", None)
idle_watts = (idle_energy.kWh * 3.6e6) / 10.0 if idle_energy else 0.0
print(f"Idle power: {idle_watts:.4f} W")

# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
results = []

tracker = EmissionsTracker(
    project_name="green_audit",
    measure_power_secs=1.0,
    save_to_file=False,
    log_level="error",
)
tracker.start()

print(f"\n[2] Running {len(PROMPTS)} prompts ({precision_label})...")

for idx, prompt in enumerate(PROMPTS):
    task_id = idx + 1
    category = (
        "Mathematics"      if idx < 100 else
        "Science"          if idx < 200 else
        "Programming-CS"   if idx < 300 else
        "Humanities"       if idx < 400 else
        "Meta-cognition"
    )

    # Energy snapshot: start
    tracker._measure_power_and_energy()
    e_start = tracker._total_energy.kWh

    # Tokenize and infer
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=False,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.time() - t0

    # Decode response (strip the prompt tokens)
    input_len = inputs["input_ids"].shape[1]
    response_text = tokenizer.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    )

    # Energy snapshot: end
    tracker._measure_power_and_energy()
    e_end = tracker._total_energy.kWh

    gross_j = (e_end - e_start) * 3.6e6
    net_j   = max(gross_j - (idle_watts * latency), 0.01)

    # AUTO QPED SCORING
    qped, score_reason = score_response(prompt, response_text, anthropic_client)

    # LpW
    denom = net_j * latency
    lpw   = qped / denom if denom > 0 else 0.0

    results.append({
        "ID":             task_id,
        "Precision":      precision_label,
        "Category":       category,
        "Prompt":         prompt,
        "Response":       response_text,          # Keep full response
        "Latency_s":      round(latency, 4),
        "Net_Energy_J":   round(net_j, 4),
        "Power_W":        round(net_j / latency, 2),
        "Qped":           qped,
        "Score_Reason":   score_reason,
        "LpW":            round(lpw, 8),
    })

    print(f"  [{task_id:>3}] {category:<18} | "
          f"Lat: {latency:.1f}s | "
          f"Energy: {net_j:.1f}J | "
          f"Qped: {qped} | "
          f"LpW: {lpw:.5f}")

    # Checkpoint every 50 prompts
    if task_id % 50 == 0:
        df_checkpoint = pd.DataFrame(results)
        checkpoint_file = f"checkpoint_{precision_label}_{task_id}.csv"
        df_checkpoint.to_csv(checkpoint_file, index=False)
        files.download(checkpoint_file)
        print(f"  >>> Checkpoint saved at {task_id} prompts")

tracker.stop()

# ---------------------------------------------------------
# FINAL EXPORT
# ---------------------------------------------------------
df = pd.DataFrame(results)
df.to_csv(OUTPUT_FILE, index=False)
files.download(OUTPUT_FILE)

# Summary statistics
print("\n" + "="*60)
print(f"PRECISION: {precision_label}")
print(f"Prompts completed: {len(df)}")
print(f"Avg Latency:    {df['Latency_s'].mean():.2f}s  "
      f"(range: {df['Latency_s'].min():.1f}–{df['Latency_s'].max():.1f}s)")
print(f"Avg Net Energy: {df['Net_Energy_J'].mean():.1f}J  "
      f"(range: {df['Net_Energy_J'].min():.1f}–{df['Net_Energy_J'].max():.1f}J)")
print(f"Avg Qped:       {df['Qped'].mean():.2f}  "
      f"(range: {df['Qped'].min()}–{df['Qped'].max()})")
print(f"Avg LpW:        {df['LpW'].mean():.6f}")
print(f"Median LpW:     {df['LpW'].median():.6f}")
print("="*60)

# Per-category breakdown
print("\nPer-category summary:")
print(df.groupby("Category")[["Latency_s","Net_Energy_J","Qped","LpW"]].mean().round(4))
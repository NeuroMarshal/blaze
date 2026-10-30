# The needle_test.py filler and needles as a raw chat prompt file for
# `blaze generate --prompt-file`; --report asks for a long answer instead of the codes (decode at long context).
#   python3 scripts/long_prompt.py TOKENS OUT [--report]
import random, sys
sys.argv = [sys.argv[0]] + sys.argv[1:]
n, out = int(sys.argv[1]), sys.argv[2]
CITIES = ["Lisbon", "Osaka", "Tbilisi", "Quito", "Bergen", "Hanoi", "Tallinn", "Accra", "Perth", "Cusco"]
NOUNS = ["turbine", "reservoir", "bridge", "relay", "pipeline", "furnace", "antenna", "rail line", "dam", "depot"]
ADJS = ["northern", "aging", "upgraded", "temporary", "primary", "backup", "coastal", "inland", "eastern", "remote"]
VERBS = ["inspected", "repaired", "measured", "recalibrated", "audited", "photographed", "cleaned", "tested"]
PLACES = ["Glasgow", "Mombasa", "Recife", "Krakow", "Busan", "Nagpur", "Tromso", "Leon", "Izmir", "Darwin"]
REPORT = "\n\nWrite a long, detailed maintenance report (at least 1500 words) summarizing the inspections above by place and equipment type."
rng = random.Random(11)
words, sentences = 0, []
while words < int(n * 0.62):
    s = (f"Report {rng.randint(1000, 9999)}: the {rng.choice(ADJS)} {rng.choice(NOUNS)} near {rng.choice(PLACES)} was "
         f"{rng.choice(VERBS)} on day {rng.randint(1, 365)} and showed {rng.randint(10, 990)} units of load, "
         f"{rng.randint(1, 99)} percent above the previous check.")
    sentences.append(s); words += len(s.split())
q = "\n\nList the secret code of every city mentioned in the IMPORTANT lines above, one per line, as 'City: code'. Answer with the list only."
q = REPORT if '--report' in sys.argv else q
open(out, "w").write("<|im_start|>user\n" + " ".join(sentences) + q + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

"""Build the rank table from the OFFICIAL evaluator so the rules cannot drift.

tab[triple_index, pair_index] = dense rank (1 = worst) of the five-card hand
made of that board triple plus that hole pair, or 0 if the cards overlap.
"""
import os, sys
from itertools import combinations
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluator import get_table

OUT = sys.argv[1] if len(sys.argv) > 1 else "work"
os.makedirs(OUT, exist_ok=True)
table = get_table()
ranks = {s: i + 1 for i, s in enumerate(sorted(set(table.values())))}
print("distinct hand ranks", len(ranks))
pairs = list(combinations(range(36), 2))
triples = list(combinations(range(36), 3))
pm = [(1 << a) | (1 << b) for a, b in pairs]
tab = np.zeros((len(triples), len(pairs)), dtype=np.uint16)
for ti, (a, b, c) in enumerate(triples):
    tm = (1 << a) | (1 << b) | (1 << c)
    row = tab[ti]
    for pi, m in enumerate(pm):
        if tm & m == 0:
            row[pi] = ranks[table[tm | m]]
tab.tofile(os.path.join(OUT, "tab.bin"))
print("tab", tab.shape, tab.max())

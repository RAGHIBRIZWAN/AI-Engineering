from collections import Counter
from cube import SOLVED, MOVES, apply, DIST, scramble, DESC

def run(s, seq):
    for m in seq.split():
        s = apply(s, m)
    return s

def inv(m):
    return m if m.endswith("2") else (m[0] if m.endswith("'") else m + "'")

ok = True
def check(name, cond):
    global ok
    ok &= bool(cond)
    print(("PASS" if cond else "FAIL"), "-", name)

check("18 moves defined", len(MOVES) == 18 and len(DESC) == 18)
check("solved state has 54 chars", len(SOLVED) == 54)
check("every move changes the cube", all(apply(SOLVED, m) != SOLVED for m in MOVES))
check("each move has 9 of every colour", all(sorted(Counter(apply(SOLVED, m)).values()) == [9]*6 for m in MOVES))
check("move then its inverse = solved", all(apply(apply(SOLVED, m), inv(m)) == SOLVED for m in MOVES))
check("every quarter turn repeated 4 times = solved", all(run(SOLVED, " ".join([m]*4)) == SOLVED for m in MOVES if len(m) == 1))
check("U matches the known facelet string",
      apply(SOLVED, "U") == "UUUUUUUUU" + "BBBRRRRRR" + "RRRFFFFFF" + "DDDDDDDDD" + "FFFLLLLLL" + "LLLBBBBBB")
check("(R U R' U') x 6 = solved (real cube property)", run(SOLVED, "R U R' U' " * 6) == SOLVED)
check("(R U R' U') x 2 is NOT solved", run(SOLVED, "R U R' U' " * 2) != SOLVED)
print("BFS states per depth:", dict(sorted(Counter(DIST.values()).items())), "(expected 1, 18, 243, 3240)")
check("BFS counts correct", dict(Counter(DIST.values())) == {0: 1, 1: 18, 2: 243, 3: 3240})
for seed in (1, 2, 3):
    s = scramble(seed)
    check(f"seed {seed}: scramble is exactly 3 moves from solved", DIST[s] == 3 and len(s) == 54)
check("same seed gives same scramble", scramble(5) == scramble(5))
print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")

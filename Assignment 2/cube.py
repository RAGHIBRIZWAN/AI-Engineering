"""Deterministic cube engine: 54-char Kociemba facelet string (faces U,R,F,D,L,B; 9 each, row-major)."""
import random
from collections import deque

FACES = "URFDLB"
SOLVED = "".join(f * 9 for f in FACES)
NORMAL = {"U": (0, 1, 0), "R": (1, 0, 0), "F": (0, 0, 1), "D": (0, -1, 0), "L": (-1, 0, 0), "B": (0, 0, -1)}

def _pos(f, i):
    r, c = divmod(i, 3)
    return {"U": (c-1, 1, r-1), "R": (1, 1-r, 1-c), "F": (c-1, 1-r, 1),
            "D": (c-1, -1, 1-r), "L": (-1, 1-r, c-1), "B": (1-c, 1-r, -1)}[f]

SLOTS = [(_pos(f, i), NORMAL[f]) for f in FACES for i in range(9)]
IDX = {s: k for k, s in enumerate(SLOTS)}

def _cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def _dot(a, b): return sum(x*y for x, y in zip(a, b))
def _rot(v, n):  # 90 degrees clockwise when looking at the face from outside
    c, d = _cross(n, v), _dot(n, v)
    return tuple(-c[k] + n[k]*d for k in range(3))

def _perm(f):
    n, p = NORMAL[f], list(range(54))
    for k, (pos, nor) in enumerate(SLOTS):
        if _dot(pos, n) == 1:
            p[IDX[(_rot(pos, n), _rot(nor, n))]] = k
    return p

PERM, MOVES = {}, []
for _f in FACES:
    _p1 = _perm(_f); _p2 = [_p1[i] for i in _p1]; _p3 = [_p1[i] for i in _p2]
    for _name, _p in ((_f, _p1), (_f + "2", _p2), (_f + "'", _p3)):
        MOVES.append(_name); PERM[_name] = _p

_FACE = dict(U="upper", R="right", F="front", D="down", L="left", B="back")
_TURN = {"": "clockwise by 90 degrees", "'": "counterclockwise by 90 degrees", "2": "by 180 degrees"}
DESC = {m: f"Turn the {_FACE[m[0]]} face {_TURN[m[1:]]}" for m in MOVES}

def apply(state, move):
    return "".join(state[i] for i in PERM[move])

def depth_map(d=3):
    """BFS from solved: exact shortest distance for every state within d moves."""
    dist, q = {SOLVED: 0}, deque([SOLVED])
    while q:
        s = q.popleft()
        if dist[s] == d:
            continue
        for m in MOVES:
            t = apply(s, m)
            if t not in dist:
                dist[t] = dist[s] + 1; q.append(t)
    return dist

DIST = depth_map(3)

def scramble(seed, d=3):
    """A state whose exact optimal distance is d (BFS-verified), chosen reproducibly by seed."""
    pool = sorted(s for s, k in DIST.items() if k == d)
    return random.Random(seed).choice(pool)

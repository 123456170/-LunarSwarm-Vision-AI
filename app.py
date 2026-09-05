# ============================================================
# LunarSwarm Vision AI — single-file Streamlit application
# (smooth-render edition)
# ------------------------------------------------------------
#   sim/            -> synthetic lunar terrain + simulation kernel
#   perception/     -> noisy onboard sensing + hazard/target detector
#   slam/           -> per-robot vote-based local map + frontier exploration
#   mapping_fusion/ -> pairwise map merge + persistent object IDs
#   coordination/   -> auction-based task allocation + fault recovery
# ============================================================
from __future__ import annotations

import base64
import io
import json
import math
import random
import time
from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image as PILImage
from scipy.ndimage import gaussian_filter

st.set_page_config(page_title="LunarSwarm Vision AI", page_icon="🌙", layout="wide")

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
GRID = 100
SAFE, CRATER, BOULDER, SCIENCE = 0, 1, 2, 3
SENSE_R = 2
SENSE_OFFS = [(dr, dc, math.hypot(dr, dc))
              for dr in range(-SENSE_R, SENSE_R + 1)
              for dc in range(-SENSE_R, SENSE_R + 1)
              if math.hypot(dr, dc) <= SENSE_R + 0.3]
DIRS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc)]
MOVE_SPEED = 0.5            # cells per tick (continuous motion -> smooth)

ROBOT_COLORS = ["#33d9ff", "#ffe066", "#7bffb0", "#ff8fd8", "#c3a6ff"]
SHARED_RGB = (255, 250, 235)
FAULT_RGB = (125, 128, 134)

CHART_CONFIG = dict(displayModeBar=False, doubleClick=False, scrollZoom=False,
                    showTips=False, responsive=False)


def hex2rgb(h: str):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def png_b64(arr_uint8) -> str:
    buf = io.BytesIO()
    PILImage.fromarray(arr_uint8, "RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------
# sim/ : terrain generator
# ------------------------------------------------------------
def generate_world(seed: int, n_robots: int) -> dict:
    rng = np.random.default_rng(seed)
    H = W = GRID
    Y, X = np.mgrid[0:H, 0:W]

    base = gaussian_filter(rng.random((H, W)), sigma=8.0)
    fine = gaussian_filter(rng.random((H, W)), sigma=2.0)
    h = 0.7 * base + 0.3 * fine
    h = (h - h.min()) / (h.max() - h.min() + 1e-9)

    labels = np.zeros((H, W), np.int8)
    craters = []

    for _ in range(int(rng.integers(9, 13))):
        r = c = 0
        rad = 4.0
        for _try in range(25):
            r = int(rng.integers(8, H - 8))
            c = int(rng.integers(8, W - 8))
            rad = float(rng.uniform(3.0, 6.5))
            if all(math.hypot(r - r2, c - c2) > rad + rad2 + 3 for r2, c2, rad2 in craters):
                break
        craters.append((r, c, rad))
        D = np.sqrt((Y - r) ** 2 + (X - c) ** 2)
        labels[D <= rad * 0.82] = CRATER
        h -= 0.50 * np.clip(1.0 - D / rad, 0.0, 1.0) ** 2
        h += 0.22 * np.exp(-((D - rad) ** 2) / (0.4 * rad + 1e-6))

    for _ in range(int(rng.integers(40, 55))):
        r = int(rng.integers(2, H - 2))
        c = int(rng.integers(2, W - 2))
        if labels[r, c] == SAFE:
            labels[r, c] = BOULDER
            if rng.random() < 0.3 and labels[r, c + 1] == SAFE:
                labels[r, c + 1] = BOULDER

    patch_seeds = []
    for _ in range(6):
        r = c = 0
        for _try in range(25):
            r = int(rng.integers(10, H - 10))
            c = int(rng.integers(10, W - 10))
            if labels[r, c] == SAFE and all(math.hypot(r - pr, c - pc) > 16 for pr, pc in patch_seeds):
                break
        patch_seeds.append((r, c))
        cells = [(r, c)]
        labels[r, c] = SCIENCE
        for _k in range(int(rng.integers(9, 20))):
            rr, cc = cells[int(rng.integers(0, len(cells)))]
            dr, dc = DIRS[int(rng.integers(0, 8))]
            nr, nc = rr + dr, cc + dc
            if 1 <= nr < H - 1 and 1 <= nc < W - 1 and labels[nr, nc] == SAFE:
                labels[nr, nc] = SCIENCE
                cells.append((nr, nc))

    spawns = []
    for _ in range(n_robots):
        placed = False
        for _try in range(300):
            r = int(rng.integers(6, H - 6))
            c = int(rng.integers(6, W - 6))
            if labels[r, c] == SAFE and all(math.hypot(r - rr, c - cc) > 20 for rr, cc in spawns):
                spawns.append((r, c))
                placed = True
                break
        if not placed:
            spawns.append((int(rng.integers(6, H - 6)), int(rng.integers(6, W - 6))))

    h = (h - h.min()) / (h.max() - h.min() + 1e-9)
    return dict(h=h, labels=labels, craters=craters, spawns=spawns)


# ------------------------------------------------------------
# Models
# ------------------------------------------------------------
@dataclass
class Robot:
    rid: int
    pos: list
    color: str
    team: int
    active: bool = True
    goal: tuple | None = None
    wp: tuple | None = None              # next waypoint (continuous motion)
    trail: list = field(default_factory=list)
    votes: np.ndarray | None = None
    known: np.ndarray | None = None


@dataclass
class SciPatch:
    pid: str
    cells: list
    seen_by: set
    status: str = "discovered"
    assigned_to: int | None = None

    @property
    def centroid(self):
        r = sum(a for a, _ in self.cells) / len(self.cells)
        c = sum(b for _, b in self.cells) / len(self.cells)
        return r, c


@dataclass
class Hazard:
    hid: str
    kind: str
    cells: list
    seen_by: set


# ------------------------------------------------------------
# Event log + state
# ------------------------------------------------------------
def add_event(kind: str, msg: str):
    s = st.session_state
    s.events.append({"t": s.tick, "type": kind, "msg": msg})
    if len(s.events) > 400:
        s.events = s.events[-300:]


def init_state(seed: int, n_robots: int):
    world = generate_world(seed, n_robots)
    robots = []
    for i in range(n_robots):
        rb = Robot(rid=i, pos=list(world["spawns"][i]), color=ROBOT_COLORS[i], team=i)
        rb.votes = np.zeros((GRID, GRID, 4), np.int16)
        rb.known = np.zeros((GRID, GRID), np.int8)
        robots.append(rb)
    st.session_state.update(
        world=world, robots=robots, rnd=random.Random(seed), seed=seed,
        tick=0, done=False, events=[],
        science=[], hazards=[],
        global_known=np.zeros((GRID, GRID), bool),
        shared_mask=np.zeros((GRID, GRID), bool),
        fused=np.full((GRID, GRID), -1, np.int8),
        teams=[{i} for i in range(n_robots)],
        pair_cool={}, merge_count=0, last_merge=None,
        next_auction=0, reassign_center=None, reassign_until=-1,
        cov_milestone=0, paused=False, boulder_log_tick=-99,
    )
    add_event("mission", f"🚀 LunarSwarm deployed: {n_robots} rovers · frontier exploration armed · seed {seed}")


def coverage(s) -> float:
    return 100.0 * float(s.global_known.sum()) / (GRID * GRID)


# ------------------------------------------------------------
# perception/
# ------------------------------------------------------------
def confuse(truth: int, rnd) -> int:
    if truth == SAFE:
        return BOULDER if rnd.random() < 0.55 else CRATER
    if truth == BOULDER:
        return SAFE if rnd.random() < 0.8 else CRATER
    if truth == CRATER:
        return SAFE if rnd.random() < 0.7 else BOULDER
    return SAFE if rnd.random() < 0.65 else BOULDER


def sense(s, rb: Robot, rnd):
    truth = s.world["labels"]
    r0, c0 = int(rb.pos[0]), int(rb.pos[1])
    cells = []
    for dr, dc, d in SENSE_OFFS:
        r, c = r0 + dr, c0 + dc
        if not (0 <= r < GRID and 0 <= c < GRID):
            continue
        p_ok = max(0.35, 0.94 - 0.13 * d)
        obs = truth[r, c] if rnd.random() < p_ok else confuse(truth[r, c], rnd)
        rb.votes[r, c, obs] += 1
        cells.append((r, c))
    return cells


def register_object(s, rb, r, c, lab):
    if lab == SCIENCE:
        for p in s.science:
            cr, cc = p.centroid
            if math.hypot(cr - r, cc - c) <= 5.0:
                p.cells.append((r, c))
                p.seen_by.add(rb.rid)
                return
        pid = f"SCI-{len(s.science) + 1:02d}"
        s.science.append(SciPatch(pid=pid, cells=[(r, c)], seen_by={rb.rid}))
        add_event("target", f"🔬 SCIENCE TARGET {pid} discovered by R{rb.rid + 1} — entering auction pool")
    elif lab == CRATER:
        for hz in s.hazards:
            if hz.kind == "crater":
                hr = sum(a for a, _ in hz.cells) / len(hz.cells)
                hc = sum(b for _, b in hz.cells) / len(hz.cells)
                if math.hypot(hr - r, hc - c) <= 6.0:
                    hz.cells.append((r, c))
                    hz.seen_by.add(rb.rid)
                    return
        cid = f"C-{sum(1 for x in s.hazards if x.kind == 'crater') + 1:02d}"
        s.hazards.append(Hazard(hid=cid, kind="crater", cells=[(r, c)], seen_by={rb.rid}))
        add_event("hazard", f"☄️ Crater hazard {cid} confirmed by R{rb.rid + 1}")
    else:
        for hz in s.hazards:
            if hz.kind == "boulder" and any(math.hypot(a - r, b - c) <= 2.3 for a, b in hz.cells):
                hz.cells.append((r, c))
                hz.seen_by.add(rb.rid)
                return
        bid = f"B-{sum(1 for x in s.hazards if x.kind == 'boulder') + 1:02d}"
        s.hazards.append(Hazard(hid=bid, kind="boulder", cells=[(r, c)], seen_by={rb.rid}))
        if s.tick - s.boulder_log_tick > 12:
            s.boulder_log_tick = s.tick
            add_event("hazard", f"🪨 Boulder field {bid} flagged by R{rb.rid + 1}")


def process_sensed(s, rb, cells):
    big_team = len(s.teams[rb.team]) > 1
    for r, c in cells:
        v = rb.votes[r, c]
        tot = int(v.sum())
        if tot < 3:
            continue
        lab = int(v.argmax())
        conf = v[lab] / tot
        if rb.known[r, c] == lab + 1 or conf < 0.62:
            continue
        rb.known[r, c] = lab + 1
        s.global_known[r, c] = True
        s.fused[r, c] = lab
        if big_team:
            s.shared_mask[r, c] = True
        if lab != SAFE:
            register_object(s, rb, r, c, lab)


# ------------------------------------------------------------
# slam/ : frontier exploration with smooth waypoint motion
# ------------------------------------------------------------
def pick_frontier(s, rb, rnd):
    known = rb.known > 0
    if not known.any():
        return (int(np.clip(rb.pos[0] + rnd.randint(-6, 6), 3, GRID - 4)),
                int(np.clip(rb.pos[1] + rnd.randint(-6, 6), 3, GRID - 4)))
    passable = (rb.known == 0) | (rb.known == SAFE + 1) | (rb.known == SCIENCE + 1)
    nbr = (np.roll(passable, 1, 0) | np.roll(passable, -1, 0) |
           np.roll(passable, 1, 1) | np.roll(passable, -1, 1))
    front = (~known) & nbr
    front[0, :] = front[-1, :] = front[:, 0] = front[:, -1] = False
    idx = np.argwhere(front)
    if not idx.size:
        nbr2 = (np.roll(known, 1, 0) | np.roll(known, -1, 0) |
                np.roll(known, 1, 1) | np.roll(known, -1, 1))
        front = (~known) & nbr2
        front[0, :] = front[-1, :] = front[:, 0] = front[:, -1] = False
        idx = np.argwhere(front)
        if not idx.size:
            return None
    d = np.hypot(idx[:, 0] - rb.pos[0], idx[:, 1] - rb.pos[1])
    score = d
    if s.reassign_center is not None and s.tick <= s.reassign_until:
        dc = np.hypot(idx[:, 0] - s.reassign_center[0], idx[:, 1] - s.reassign_center[1])
        score = 0.6 * d + 0.8 * dc
    k = min(len(idx), 5)
    pick = idx[np.argsort(score)[:k][rnd.randrange(k)]]
    return (int(pick[0]), int(pick[1]))


def update_goal(s, rb, rnd):
    if rb.goal is not None and math.hypot(rb.pos[0] - rb.goal[0], rb.pos[1] - rb.goal[1]) > 1.2:
        return
    rb.goal = None
    rb.wp = None
    for p in s.science:
        if p.status == "assigned" and p.assigned_to == rb.rid:
            rb.goal = p.centroid
            return
    g = pick_frontier(s, rb, rnd)
    rb.goal = g if g is not None else (rnd.randint(4, GRID - 5), rnd.randint(4, GRID - 5))


def step_robot(s, rb, rnd):
    """Continuous waypoint motion: robots glide between cells instead of hopping."""
    if rb.goal is None:
        return
    gr, gc = rb.goal

    need_wp = rb.wp is None
    if not need_wp and math.hypot(rb.pos[0] - rb.wp[0], rb.pos[1] - rb.wp[1]) < 0.35:
        need_wp = True
    if need_wp:
        rb.wp = None
        r0, c0 = int(round(rb.pos[0])), int(round(rb.pos[1]))
        best, best_sc = None, 1e18
        for dr, dc in DIRS:
            nr, nc = r0 + dr, c0 + dc
            if not (1 <= nr < GRID - 1 and 1 <= nc < GRID - 1):
                continue
            cell = rb.known[nr, nc]
            pen = 2.8 if cell == CRATER + 1 else (1.6 if cell == BOULDER + 1 else 0.0)
            sc = math.hypot(nr - gr, nc - gc) + pen + rnd.random() * 0.08
            if sc < best_sc:
                best_sc, best = sc, (nr, nc)
        if best is None:
            return
        if best_sc > 3.5 and rnd.random() < 0.3:
            return                                   # hesitate near hazards (no jitter)
        rb.wp = best

    wr, wc = rb.wp
    dr_, dc_ = wr - rb.pos[0], wc - rb.pos[1]
    d = math.hypot(dr_, dc_)
    if d <= 1e-6:
        return
    step = min(MOVE_SPEED, d)
    rb.pos = [rb.pos[0] + dr_ / d * step, rb.pos[1] + dc_ / d * step]
    rb.trail.append((rb.pos[0], rb.pos[1]))
    if len(rb.trail) > 90:
        rb.trail.pop(0)


# ------------------------------------------------------------
# mapping_fusion/
# ------------------------------------------------------------
def do_merge(s, A: Robot, B: Robot):
    S = np.clip(A.votes.astype(np.int32) + B.votes.astype(np.int32), 0, 3000)
    tot = S.sum(-1)
    conf = S.max(-1) / np.maximum(tot, 1)
    ok = (tot >= 3) & (conf >= 0.62)
    lab = S.argmax(-1).astype(np.int8)
    new_known = np.where(ok, lab + 1, 0).astype(np.int8)

    A.votes = S.astype(np.int16).copy()
    B.votes = S.astype(np.int16).copy()
    A.known = new_known.copy()
    B.known = new_known.copy()

    union = new_known > 0
    s.global_known |= union
    s.fused[ok] = lab[ok]

    if A.team != B.team:
        s.teams[A.team] |= s.teams[B.team]
        for rid in s.teams[A.team]:
            s.robots[rid].team = A.team
        s.teams[B.team] = set()
    s.shared_mask |= union

    n_dup = sum(1 for p in s.science if len(p.seen_by) >= 2)
    add_event("merge",
              f"🤝 MAP MERGE R{A.rid + 1}⇄R{B.rid + 1}: {int(union.sum())} cells fused · "
              f"{n_dup} science target(s) on persistent IDs (duplicates resolved)")
    s.merge_count += 1
    s.last_merge = (s.tick, ((A.pos[0] + B.pos[0]) / 2, (A.pos[1] + B.pos[1]) / 2))


def comms_check(s, rnd):
    act = [r for r in s.robots if r.active]
    for i in range(len(act)):
        for j in range(i + 1, len(act)):
            A, B = act[i], act[j]
            dist = math.hypot(A.pos[0] - B.pos[0], A.pos[1] - B.pos[1])
            ka, kb = A.known > 0, B.known > 0
            overlap = int(np.count_nonzero(ka & kb))
            if dist > s.comm_range and overlap < 12:
                continue
            key = (A.rid, B.rid)
            if s.tick - s.pair_cool.get(key, -999) < 26:
                continue
            new_info = int(np.count_nonzero(ka & ~kb) + np.count_nonzero(kb & ~ka))
            if new_info < 25:
                s.pair_cool[key] = s.tick - 18
                continue
            do_merge(s, A, B)
            s.pair_cool[key] = s.tick


# ------------------------------------------------------------
# coordination/
# ------------------------------------------------------------
def assigned_load(s, rid) -> int:
    return sum(1 for p in s.science if p.status == "assigned" and p.assigned_to == rid)


def run_auctions(s, rnd):
    if s.tick < s.next_auction:
        return
    pending = [p for p in s.science if p.status == "discovered"]
    act = [r for r in s.robots if r.active]
    if not pending or not act:
        return
    p = pending[0]
    cr, cc = p.centroid
    bids = []
    for r in act:
        d = math.hypot(r.pos[0] - cr, r.pos[1] - cc)
        load = 1.0 if assigned_load(s, r.rid) == 0 else 0.45
        bids.append((load / (1.0 + d) * (1.0 + 0.15 * rnd.random()), r))
    bid, winner = max(bids, key=lambda t: t[0])
    p.status, p.assigned_to = "assigned", winner.rid
    winner.goal = (cr, cc)
    add_event("auction", f"🏆 AUCTION WIN: {p.pid} → R{winner.rid + 1} (bid {bid:.3f}, {len(bids)} bidders)")
    s.next_auction = s.tick + 6


def survey_check(s):
    for p in s.science:
        if p.status != "assigned":
            continue
        rb = s.robots[p.assigned_to]
        if not rb.active:
            continue
        cr, cc = p.centroid
        if math.hypot(rb.pos[0] - cr, rb.pos[1] - cc) <= 1.8:
            p.status = "surveyed"
            add_event("target", f"✅ {p.pid} SURVEYED by R{rb.rid + 1} — spectral scan stored to shared map")


def simulate_fault():
    s = st.session_state
    act = [r for r in s.robots if r.active]
    if s.done or len(act) <= 1:
        return
    rb = s.rnd.choice(act)
    rb.active = False
    add_event("fault", f"🛑 ROBOT FAULT: R{rb.rid + 1} offline at ({int(rb.pos[0])},{int(rb.pos[1])})")
    released = 0
    for p in s.science:
        if p.status == "assigned" and p.assigned_to == rb.rid:
            p.status, p.assigned_to = "discovered", None
            released += 1
            add_event("reallocate", f"♻️ {p.pid} released from R{rb.rid + 1} → back to auction pool")
    s.next_auction = s.tick
    s.reassign_center = tuple(rb.pos)
    s.reassign_until = s.tick + 280
    add_event("reallocate", f"🧭 Swarm re-allocating R{rb.rid + 1}'s unexplored sector ({released} task(s) re-auctioned)")


# ------------------------------------------------------------
# Simulation kernel
# ------------------------------------------------------------
def tick(s):
    if s.done:
        return
    rnd = s.rnd
    s.tick += 1
    for rb in s.robots:
        if not rb.active:
            continue
        update_goal(s, rb, rnd)
        step_robot(s, rb, rnd)
        cells = sense(s, rb, rnd)
        process_sensed(s, rb, cells)
    comms_check(s, rnd)
    run_auctions(s, rnd)
    survey_check(s)

    cov = coverage(s)
    if cov >= s.cov_milestone + 25:
        s.cov_milestone = int(cov // 25) * 25
        add_event("coverage", f"📈 Combined coverage reached {s.cov_milestone}%")
    if cov >= 99.0 or s.tick >= 4000:
        s.done = True
        surveyed = sum(1 for p in s.science if p.status == "surveyed")
        add_event("mission", f"🏁 MISSION COMPLETE — final combined coverage {cov:.1f}% · "
                             f"{surveyed}/{len(s.science)} targets surveyed · {s.merge_count} merges")


def restart():
    init_state(random.SystemRandom().randint(0, 10 ** 6), int(st.session_state.get("swarm_n", 4)))


# ------------------------------------------------------------
# Rendering (PNG layout-image + locked viewport = no shake)
# ------------------------------------------------------------
def build_image_rgb(s):
    h = s.world["h"]
    base = 22.0 + h * 118.0
    img = np.stack([base * 0.94, base * 0.98, base * 1.10], axis=-1)

    def blend(mask, color, a):
        if not mask.any():
            return
        for i in range(3):
            img[..., i][mask] = img[..., i][mask] * (1 - a) + color[i] * a

    for rb in s.robots:
        blend((rb.known > 0) & ~s.shared_mask, hex2rgb(rb.color) if rb.active else FAULT_RGB, 0.40)
    blend(s.shared_mask & s.global_known, SHARED_RGB, 0.42)
    blend((s.fused == SCIENCE) & s.global_known, (255, 77, 231), 0.65)
    blend((s.fused == CRATER) & s.global_known, (255, 110, 60), 0.30)
    blend((s.fused == BOULDER) & s.global_known, (168, 178, 190), 0.30)
    return np.clip(img, 0, 255).astype(np.uint8)


def build_figure(s):
    fig = go.Figure()

    # static base as a compact PNG layout image (small payload, no flicker)
    fig.add_layout_image(dict(
        source=png_b64(build_image_rgb(s)),
        xref="x", yref="y", x=0, y=0, sizex=GRID, sizey=GRID,
        sizing="stretch", opacity=1.0, layer="below"))

    # trails (spline-smoothed, float positions)
    for rb in s.robots:
        fig.add_trace(go.Scatter(
            x=[c + 0.5 for _, c in rb.trail], y=[r + 0.5 for r, _ in rb.trail],
            mode="lines", hoverinfo="skip", opacity=0.55,
            line=dict(width=1.6, shape="spline", smoothing=0.9,
                      color=rb.color if rb.active else "#7d828a")))

    # boulders
    bx = [b + 0.5 for hz in s.hazards if hz.kind == "boulder" for _, b in hz.cells]
    by = [a + 0.5 for hz in s.hazards if hz.kind == "boulder" for a, _ in hz.cells]
    fig.add_trace(go.Scatter(x=bx, y=by, mode="markers", hoverinfo="skip",
                             marker=dict(symbol="diamond", size=6, color="#d7dde6", opacity=0.9)))

    # craters
    cx, cy, csz = [], [], []
    for hz in s.hazards:
        if hz.kind != "crater":
            continue
        cx.append(sum(b for _, b in hz.cells) / len(hz.cells) + 0.5)
        cy.append(sum(a for a, _ in hz.cells) / len(hz.cells) + 0.5)
        csz.append(9 + 2.2 * math.sqrt(len(hz.cells)))
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers", hoverinfo="skip",
                             marker=dict(symbol="circle-open", size=csz, color="#ff9f43",
                                         line=dict(width=2.5))))

    # science targets
    sx, sy, stxt, scol, ssz = [], [], [], [], []
    for p in s.science:
        cr, cc = p.centroid
        sx.append(cc + 0.5)
        sy.append(cr + 0.5)
        stxt.append(p.pid)
        if p.status == "surveyed":
            scol.append("#3dff8e"); ssz.append(12)
        elif p.status == "assigned":
            scol.append("#ffd23f"); ssz.append(15)
        else:
            scol.append("#ff4df0"); ssz.append(15)
    fig.add_trace(go.Scatter(x=sx, y=sy, mode="markers+text", text=stxt,
                             textposition="top center", hoverinfo="skip",
                             textfont=dict(size=9, color="#ffffff"),
                             marker=dict(symbol="star", size=ssz, color=scol,
                                         line=dict(width=1, color="#ffffff"))))

    # robots
    rx, ry, rtxt, rcol, rsym = [], [], [], [], []
    for rb in s.robots:
        rx.append(rb.pos[1] + 0.5)
        ry.append(rb.pos[0] + 0.5)
        rtxt.append(f"R{rb.rid + 1}")
        if rb.active:
            rcol.append(rb.color); rsym.append("circle")
        else:
            rcol.append("#8a8f98"); rsym.append("x")
    fig.add_trace(go.Scatter(x=rx, y=ry, mode="markers+text", text=rtxt,
                             textposition="bottom center", hoverinfo="skip",
                             textfont=dict(size=11, color="#ffffff", family="monospace"),
                             marker=dict(size=13, color=rcol, symbol=rsym,
                                         line=dict(width=2, color="#ffffff"))))

    # merge flash
    if s.last_merge and s.tick - s.last_merge[0] <= 18:
        age = s.tick - s.last_merge[0]
        mr, mc = s.last_merge[1]
        rad = 2.5 + age * 1.2
        fig.add_shape(type="circle", x0=mc + 0.5 - rad, x1=mc + 0.5 + rad,
                      y0=mr + 0.5 - rad, y1=mr + 0.5 + rad,
                      line=dict(color="#ffe97a", width=3), opacity=max(0.0, 1 - age / 18))

    fig.update_layout(
        height=560, template="plotly_dark", uirevision="lsv", hovermode=False,
        margin=dict(l=2, r=2, t=6, b=2), dragmode=False, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#05070d",
        xaxis=dict(range=[0, GRID], fixedrange=True, visible=False,
                   constrain="domain", scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[GRID, 0], fixedrange=True, visible=False, constrain="domain"))
    return fig


# ------------------------------------------------------------
# UI — fast fragment (map) + slow fragment (panels)
# ------------------------------------------------------------
def fast_body():
    s = st.session_state
    if not s.paused and not s.done:
        for _ in range(int(s.speed)):
            tick(s)

    # fixed-height status bar: no layout shift between states
    if s.last_merge and (s.tick - s.last_merge[0]) <= 14:
        st.markdown(f"<div class='statusbar mergeflash'>🛰️ <b>MAP MERGE</b>&nbsp; — t={s.last_merge[0]} · "
                    f"local maps fused into shared chart · persistent target IDs reconciled</div>",
                    unsafe_allow_html=True)
    elif s.done:
        st.markdown(f"<div class='statusbar donebar'>🏁 Mission complete — final combined coverage "
                    f"{coverage(s):.1f}% · use 'Restart Mission' for new terrain</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='statusbar clock'>MISSION CLOCK t={s.tick:04d} · comms range "
                    f"{s.comm_range} cells · {sum(1 for r in s.robots if r.active)}/{len(s.robots)} "
                    f"rovers active</div>", unsafe_allow_html=True)

    cov = coverage(s)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🗺 Combined Coverage", f"{cov:.1f}%")
    m2.metric("🔬 Science Targets", f"{sum(1 for p in s.science if p.status == 'surveyed')}/{len(s.science)}")
    m3.metric("🤝 Map Merges", s.merge_count)
    m4.metric("🤖 Active Rovers", f"{sum(1 for r in s.robots if r.active)}/{len(s.robots)}")
    m5.metric("⚠️ Hazards Mapped", len(s.hazards))

    st.plotly_chart(build_figure(s), use_container_width=True, config=CHART_CONFIG, key="swarm_map")
    st.caption("🎨 Robot-colored trails = individual maps · white/cream = merged shared map · "
               "⭐ science targets · ◯ craters · ◆ boulders")


def slow_body():
    s = st.session_state
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("🤖 Swarm Roster")
        box = st.container(height=250, border=True)
        for rb in s.robots:
            col = rb.color if rb.active else "#8a8f98"
            if not rb.active:
                stat = "FAULT ✕"
            else:
                tgt = next((p.pid for p in s.science
                            if p.status == "assigned" and p.assigned_to == rb.rid), None)
                stat = f"en route → {tgt}" if tgt else "frontier exploring"
            box.markdown(
                f"<div style='line-height:2.1'><span style='color:{col};font-size:16px'>●</span> "
                f"<b>R{rb.rid + 1}</b> <span style='opacity:.7;font-size:12.5px'>— {stat}</span></div>",
                unsafe_allow_html=True)

    with c2:
        st.subheader("🎯 Task Allocation (auction)")
        if s.science:
            rows = []
            for p in s.science:
                icon = {"discovered": "🔎 open", "assigned": "🏆 assigned", "surveyed": "✅ surveyed"}[p.status]
                rows.append({
                    "Target": p.pid,
                    "Status": icon,
                    "Robot": f"R{p.assigned_to + 1}" if p.assigned_to is not None else "—",
                    "Seen by": ",".join(f"R{i + 1}" for i in sorted(p.seen_by)),
                })
            st.dataframe(rows, hide_index=True, height=222, use_container_width=True)
        else:
            st.caption("No targets yet — rovers are sweeping for anomalies…")

    with c3:
        st.subheader("📜 Event Log")
        log = st.container(height=250, border=True)
        lines = [f"t={e['t']:04d}  [{e['type']:<9}] {e['msg']}" for e in s.events[-70:]][::-1]
        log.markdown("<pre style='font-size:11.5px;margin:0;white-space:pre-wrap'>"
                     + "<br>".join(lines) + "</pre>", unsafe_allow_html=True)


# ------------------------------------------------------------
# Page scaffolding
# ------------------------------------------------------------
st.markdown("""
<style>
.statusbar{height:46px;display:flex;align-items:center;padding:0 1rem;border-radius:.6rem;
  margin:.1rem 0 .6rem 0;overflow:hidden;white-space:nowrap;font-size:.98rem}
.mergeflash{color:#1a1200;border:1px solid #ffdf80;
  background:linear-gradient(90deg,#ffe97a,#ffd23f,#ffe97a);background-size:200% 100%;
  animation:slide 1.1s linear infinite alternate, pulse .8s ease-in-out infinite alternate}
.donebar{color:#06281a;background:linear-gradient(90deg,#79f2b0,#3dff8e);border:1px solid #8fffc6}
.clock{color:#9fb4d8;background:#0d1420;border:1px solid #1d2c44;font-family:monospace}
@keyframes slide{from{background-position:0% 0}to{background-position:100% 0}}
@keyframes pulse{from{opacity:.75}to{opacity:1}}
</style>
""", unsafe_allow_html=True)

if "world" not in st.session_state:
    init_state(seed=int(np.random.default_rng().integers(0, 10 ** 6)), n_robots=4)

st.sidebar.title("🎛 Mission Control")
st.sidebar.slider("Simulation speed (ticks / frame)", 1, 5, 2, key="speed")
st.sidebar.toggle("⏸ Pause", key="paused")
st.sidebar.slider("📡 Comm range (cells)", 8, 26, 16, key="comm_range")
st.sidebar.selectbox("Swarm size (applies on restart)", [3, 4, 5], index=1, key="swarm_n")
st.sidebar.button("🛰 Simulate Robot Fault", on_click=simulate_fault,
                  use_container_width=True, disabled=st.session_state.get("done", False))
st.sidebar.button("🔁 Restart Mission (new terrain)", on_click=restart, use_container_width=True)
st.sidebar.divider()
st.sidebar.download_button(
    "⬇ Export structured event log (JSON)",
    json.dumps(st.session_state.get("events", []), indent=1),
    file_name="lunarswarm_event_log.json", mime="application/json")

with st.sidebar.expander("📡 Communication-constraint assumptions"):
    st.markdown("""
- **No GPS / no ground-truth access.** Each rover senses only a radius-2 disk through a noisy
  classifier and accumulates votes in its own local map (SLAM stand-in).
- **Shared odometry frame.** Rovers align frames via a visual handshake during comms windows
  (simulated as exact alignment).
- **Comms windows only.** Maps exchange only when two rovers are within comm range *or* their
  explored trails overlap — there is no central server; the dashboard shows the fused *output*.
- **Merge policy.** Per-cell vote consensus; detections are deduplicated by proximity into
  persistent IDs (`SCI-xx`, `C-xx`, `B-xx`), so duplicates never double-count.
- **Tasking.** First-price auction, bid = availability ÷ (1 + distance). Assigned targets leave
  the pool (no duplicate assignment). Faulted rovers' tasks are re-auctioned and their
  unexplored sector is re-weighted into teammates' frontier selection.
""")

st.title("🌙 LunarSwarm Vision AI")
st.caption("Simulated multi-robot lunar exploration · per-robot noisy SLAM · frontier exploration · "
           "cooperative map merging · auction-based science-task allocation")

if hasattr(st, "fragment"):
    fast_panel = st.fragment(run_every=0.15)(fast_body)
    slow_panel = st.fragment(run_every=0.8)(slow_body)
else:  # fallback: simple full-page rerun loop
    def fast_panel():
        fast_body()

    def slow_panel():
        slow_body()

fast_panel()
slow_panel()

if not hasattr(st, "fragment"):
    time.sleep(0.2)
    st.rerun()
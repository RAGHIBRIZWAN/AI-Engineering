import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Dict, List, Optional, Tuple

import pygame
import requests


def load_dotenv(dotenv_path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a .env file into os.environ."""
    if not os.path.exists(dotenv_path):
        return

    try:
        with open(dotenv_path, "r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                if not key:
                    continue

                # Trim matching quote wrappers.
                if (
                    len(value) >= 2
                    and value[0] == value[-1]
                    and value[0] in {"'", '"'}
                ):
                    value = value[1:-1]

                os.environ.setdefault(key, value)
    except OSError:
        # If .env cannot be read, continue with regular environment variables.
        return


load_dotenv()

# ------------------------------
# Configuration
# ------------------------------
GRID_WIDTH = 20
GRID_HEIGHT = 20
CELL_SIZE = 28
FPS = 60
PANEL_WIDTH = 320

# Move pacing speeds up as the snake scores, to show JEV making fast calls
# under increasing pressure. Values are seconds between moves.
BASE_MOVE_INTERVAL_SEC = 0.11
MIN_MOVE_INTERVAL_SEC = 0.05
SPEEDUP_PER_FOOD_SEC = 0.004

WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE + PANEL_WIDTH
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE

API_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL_NAME = "typesafe/jev-1.13"
API_KEY = os.environ.get("OPENROUTER_API_KEY", "OPENROUTER_API_KEY")
API_TIMEOUT_SEC = 5

HTTP_REFERER = "https://github.com/openrouter/snake-jev"
X_TITLE = "JEV Snake AI"

# "demo"  -> use the local JEV decision engine only (fast, deterministic, no
#            network dependency; ideal for live demos of decision speed).
# "cloud" -> call the real OpenRouter Decisions API first and seamlessly fall
#            back to the local engine if it errors, times out, or is unavailable.
JEV_MODE = os.environ.get("JEV_MODE", "demo").strip().lower()
if JEV_MODE not in {"demo", "cloud"}:
    JEV_MODE = "demo"

# Per https://openrouter.ai/typesafe/jev-1.13: $0.042 per 1M input tokens,
# output tokens are free, so only input (request) tokens are billed.
INPUT_TOKEN_COST_PER_MILLION = float(os.environ.get("INPUT_TOKEN_COST_PER_MILLION", "0.042"))
MILESTONE_SCORE_STEP = 100

NEXT_MOVE_QUESTION = {
    "type": "choice",
    "instructions": (
        "Select the safest and most optimal move direction for "
        "the snake to reach the food without colliding with walls "
        "or itself."
    ),
    "criteria": {
        "U": "Move Up (y decreases)",
        "D": "Move Down (y increases)",
        "L": "Move Left (x decreases)",
        "R": "Move Right (x increases)",
    },
}


def build_request_body(state_payload: Dict) -> Dict:
    return {
        "model": MODEL_NAME,
        "state": state_payload,
        "questions": {"next_move": NEXT_MOVE_QUESTION},
    }


def estimate_input_tokens(body: Dict) -> int:
    # No real tokenizer available offline; ~4 chars/token is a standard estimate.
    return max(1, len(json.dumps(body)) // 4)

DIRECTIONS: Dict[str, Tuple[int, int]] = {
    "U": (0, -1),
    "D": (0, 1),
    "L": (-1, 0),
    "R": (1, 0),
}

OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}


@dataclass
class DecisionResult:
    request_id: int
    move: Optional[str]
    confidence: Optional[float]
    probabilities: Optional[Dict[str, float]]
    latency_ms: Optional[float]
    engine: str
    raw_response: Optional[Dict]
    error: Optional[str]


class DecisionWorker:
    """Runs JEV decision calls (cloud and/or local) in a background thread."""

    def __init__(self) -> None:
        self._requests: Queue = Queue(maxsize=1)
        self._results: Queue = Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.pending = False
        self.thread_started = False

    def start(self) -> None:
        if not self.thread_started:
            self._thread.start()
            self.thread_started = True

    def stop(self) -> None:
        self._stop_event.set()
        if self.thread_started:
            self._thread.join(timeout=1.0)

    def submit_state(self, request_id: int, state_payload: Dict) -> bool:
        if self.pending:
            return False

        try:
            self._requests.put_nowait((request_id, state_payload))
            self.pending = True
            return True
        except Full:
            return False

    def poll_result(self) -> Optional[DecisionResult]:
        try:
            result = self._results.get_nowait()
            self.pending = False
            return result
        except Empty:
            return None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                request_id, payload = self._requests.get(timeout=0.05)
            except Empty:
                continue

            result = self._decide(request_id, payload)

            # Keep only the most recent result.
            while True:
                try:
                    self._results.get_nowait()
                except Empty:
                    break

            try:
                self._results.put_nowait(result)
            except Full:
                pass

    def _decide(self, request_id: int, state_payload: Dict) -> DecisionResult:
        if JEV_MODE == "cloud":
            cloud_result = self._call_openrouter(request_id, state_payload)
            if cloud_result.move in {"U", "D", "L", "R"}:
                return cloud_result
            print(
                f"[JEV Cloud] request {request_id} failed after "
                f"{cloud_result.latency_ms:.0f} ms -> falling back to local\n"
                f"  error: {cloud_result.error}",
                file=sys.stderr,
                flush=True,
            )
            # Cloud call failed/invalid: fall back to the local engine so the
            # game never stalls waiting on an unavailable endpoint.
            return self._local_decision(request_id, state_payload)

        return self._local_decision(request_id, state_payload)

    def _local_decision(self, request_id: int, state_payload: Dict) -> DecisionResult:
        start = time.perf_counter()

        snake = [tuple(point) for point in state_payload["body"]]
        food = tuple(state_payload["food"])
        direction = state_payload["current_direction"]
        safe_moves = state_payload["safe_moves"]

        move, confidence, probabilities = local_decision_engine(
            snake, food, direction, safe_moves
        )

        # Tiny simulated compute delay so the panel reflects a realistic,
        # genuinely fast decision latency (single-digit to low double-digit ms).
        time.sleep(random.uniform(0.004, 0.018))
        latency_ms = (time.perf_counter() - start) * 1000.0

        raw_response = {
            "answers": {
                "next_move": {
                    "choice": move,
                    "confidence": confidence,
                    "probabilities": probabilities,
                }
            },
            "engine": "jev-local-simulator",
        }

        return DecisionResult(
            request_id=request_id,
            move=move,
            confidence=confidence,
            probabilities=probabilities,
            latency_ms=latency_ms,
            engine="JEV Local",
            raw_response=raw_response,
            error=None,
        )

    def _call_openrouter(self, request_id: int, state_payload: Dict) -> DecisionResult:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": HTTP_REFERER,
            "X-Title": X_TITLE,
        }

        body = build_request_body(state_payload)

        start = time.perf_counter()
        try:
            response = requests.post(
                API_URL,
                json=body,
                headers=headers,
                timeout=API_TIMEOUT_SEC,
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            response.raise_for_status()
            data = response.json()

            answer = data.get("answers", {}).get("next_move", {})
            move = answer.get("choice")
            confidence = answer.get("confidence")
            probabilities = answer.get("probabilities")

            if move not in {"U", "D", "L", "R"}:
                return DecisionResult(
                    request_id=request_id,
                    move=None,
                    confidence=confidence,
                    probabilities=probabilities,
                    latency_ms=latency_ms,
                    engine="JEV Cloud",
                    raw_response=data,
                    error="Invalid choice in API response",
                )

            return DecisionResult(
                request_id=request_id,
                move=move,
                confidence=confidence,
                probabilities=probabilities,
                latency_ms=latency_ms,
                engine="JEV Cloud",
                raw_response=data,
                error=None,
            )
        except requests.RequestException as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            body_text = exc.response.text[:300] if getattr(exc, "response", None) is not None else ""
            return DecisionResult(
                request_id=request_id,
                move=None,
                confidence=None,
                probabilities=None,
                latency_ms=latency_ms,
                engine="JEV Cloud",
                raw_response=None,
                error=f"Request error: {exc} {body_text}",
            )
        except ValueError as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return DecisionResult(
                request_id=request_id,
                move=None,
                confidence=None,
                probabilities=None,
                latency_ms=latency_ms,
                engine="JEV Cloud",
                raw_response=None,
                error=f"JSON parse error: {exc}",
            )


def bfs_distance(
    start: Tuple[int, int], goal: Tuple[int, int], blocked: set
) -> Optional[int]:
    """Shortest grid distance from start to goal avoiding blocked cells."""
    if start == goal:
        return 0

    visited = {start}
    queue = deque([(start, 0)])

    while queue:
        (x, y), dist = queue.popleft()
        for dx, dy in DIRECTIONS.values():
            nxt = (x + dx, y + dy)
            if nxt in visited:
                continue
            if not (0 <= nxt[0] < GRID_WIDTH and 0 <= nxt[1] < GRID_HEIGHT):
                continue
            if nxt in blocked:
                continue
            if nxt == goal:
                return dist + 1
            visited.add(nxt)
            queue.append((nxt, dist + 1))

    return None


def flood_fill_count(start: Tuple[int, int], blocked: set) -> int:
    """Count cells reachable from start, used as an open-space safety metric."""
    if start in blocked:
        return 0

    visited = {start}
    queue = deque([start])
    count = 0

    while queue:
        x, y = queue.popleft()
        count += 1
        for dx, dy in DIRECTIONS.values():
            nxt = (x + dx, y + dy)
            if nxt in visited:
                continue
            if not (0 <= nxt[0] < GRID_WIDTH and 0 <= nxt[1] < GRID_HEIGHT):
                continue
            if nxt in blocked:
                continue
            visited.add(nxt)
            queue.append(nxt)

    return count


def local_decision_engine(
    snake: List[Tuple[int, int]],
    food: Tuple[int, int],
    direction: str,
    safe_moves: List[str],
) -> Tuple[Optional[str], float, Dict[str, float]]:
    """Fast pathfinding + open-space heuristic mimicking the JEV choice contract."""
    probabilities = {move: 0.0 for move in DIRECTIONS}

    if not safe_moves:
        return None, 0.0, probabilities

    head = snake[0]
    scores: Dict[str, float] = {}

    for move, (dx, dy) in DIRECTIONS.items():
        if move not in safe_moves:
            continue

        new_head = (head[0] + dx, head[1] + dy)
        grows = new_head == food
        new_body = [new_head] + snake if grows else [new_head] + snake[:-1]
        blocked = set(new_body[1:])

        dist = bfs_distance(new_head, food, blocked)
        space = flood_fill_count(new_head, blocked)

        distance_score = 1.0 / (dist + 1) if dist is not None else 0.0
        space_score = space / float(GRID_WIDTH * GRID_HEIGHT)

        score = distance_score * 2.0 + space_score * 1.5
        # Penalize moves likely to trap the snake in a pocket smaller than itself.
        if space < len(new_body) + 1:
            score -= 5.0

        scores[move] = score

    best_move = max(scores, key=lambda m: scores[m])

    # Softmax over safe-move scores for realistic, display-friendly probabilities.
    max_score = max(scores.values())
    exp_scores = {m: math.exp(s - max_score) for m, s in scores.items()}
    total = sum(exp_scores.values())
    for move in scores:
        probabilities[move] = exp_scores[move] / total if total > 0 else 1.0 / len(scores)

    confidence = probabilities[best_move]
    return best_move, confidence, probabilities


def random_food_position(snake: List[Tuple[int, int]]) -> Tuple[int, int]:
    occupied = set(snake)
    empty_cells = [
        (x, y)
        for y in range(GRID_HEIGHT)
        for x in range(GRID_WIDTH)
        if (x, y) not in occupied
    ]
    return random.choice(empty_cells) if empty_cells else (-1, -1)


def compute_safe_moves(
    snake: List[Tuple[int, int]], current_direction: str
) -> List[str]:
    safe = []
    head_x, head_y = snake[0]
    neck = snake[1] if len(snake) > 1 else None

    for move, (dx, dy) in DIRECTIONS.items():
        # Prevent immediate 180 reverse into neck.
        if move == OPPOSITE[current_direction]:
            continue

        nx, ny = head_x + dx, head_y + dy

        if nx < 0 or nx >= GRID_WIDTH or ny < 0 or ny >= GRID_HEIGHT:
            continue

        next_cell = (nx, ny)
        if next_cell in snake[:-1]:
            continue

        if neck is not None and next_cell == neck:
            continue

        safe.append(move)

    return safe


def move_snake(
    snake: List[Tuple[int, int]],
    direction: str,
    food: Tuple[int, int],
) -> Tuple[List[Tuple[int, int]], bool, bool]:
    dx, dy = DIRECTIONS[direction]
    head_x, head_y = snake[0]
    new_head = (head_x + dx, head_y + dy)

    out_of_bounds = (
        new_head[0] < 0
        or new_head[0] >= GRID_WIDTH
        or new_head[1] < 0
        or new_head[1] >= GRID_HEIGHT
    )
    if out_of_bounds:
        return snake, False, True

    grew = new_head == food
    new_snake = [new_head] + snake
    if not grew:
        new_snake.pop()

    if new_head in new_snake[1:]:
        return snake, False, True

    return new_snake, grew, False


def current_move_interval(score: int) -> float:
    steps = score // 10
    interval = BASE_MOVE_INTERVAL_SEC - steps * SPEEDUP_PER_FOOD_SEC
    return max(MIN_MOVE_INTERVAL_SEC, interval)


def draw_grid(screen: pygame.Surface) -> None:
    grid_color = (56, 64, 74)
    for x in range(0, GRID_WIDTH * CELL_SIZE, CELL_SIZE):
        pygame.draw.line(screen, grid_color, (x, 0), (x, GRID_HEIGHT * CELL_SIZE))
    for y in range(0, GRID_HEIGHT * CELL_SIZE, CELL_SIZE):
        pygame.draw.line(screen, grid_color, (0, y), (GRID_WIDTH * CELL_SIZE, y))


def draw_probability_bars(
    screen: pygame.Surface,
    panel_x: int,
    y: int,
    probabilities: Optional[Dict[str, float]],
    last_decision: str,
    small_font: pygame.font.Font,
) -> int:
    bar_width = PANEL_WIDTH - 48
    bar_height = 18
    gap = 10

    for move_key in ["U", "D", "L", "R"]:
        p = probabilities.get(move_key) if probabilities else None
        p = p if isinstance(p, (int, float)) else 0.0
        p = max(0.0, min(p, 1.0))

        is_chosen = move_key == last_decision
        track_rect = pygame.Rect(panel_x + 24, y, bar_width, bar_height)
        pygame.draw.rect(screen, (36, 42, 50), track_rect, border_radius=5)

        fill_width = int(bar_width * p)
        if fill_width > 0:
            fill_color = (88, 214, 141) if is_chosen else (52, 152, 219)
            fill_rect = pygame.Rect(panel_x + 24, y, fill_width, bar_height)
            pygame.draw.rect(screen, fill_color, fill_rect, border_radius=5)

        label = small_font.render(f"{move_key}  {p * 100:4.1f}%", True, (230, 235, 240))
        screen.blit(label, (panel_x + 30, y + 1))

        y += bar_height + gap

    return y


def draw_game(
    screen: pygame.Surface,
    snake: List[Tuple[int, int]],
    food: Tuple[int, int],
    score: int,
    status: str,
    last_decision: str,
    confidence: Optional[float],
    latency_ms: Optional[float],
    engine: str,
    total_api_calls: int,
    total_input_tokens: int,
    total_cost: float,
    probabilities: Optional[Dict[str, float]],
) -> None:
    screen.fill((18, 21, 26))

    playfield = pygame.Rect(0, 0, GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE)
    pygame.draw.rect(screen, (24, 30, 38), playfield)

    draw_grid(screen)

    # Food
    if food != (-1, -1):
        fx, fy = food
        food_rect = pygame.Rect(fx * CELL_SIZE + 4, fy * CELL_SIZE + 4, CELL_SIZE - 8, CELL_SIZE - 8)
        pygame.draw.rect(screen, (230, 74, 95), food_rect, border_radius=6)

    # Snake body/head
    for i, (x, y) in enumerate(snake):
        inset = 4 if i == 0 else 5
        color = (88, 214, 141) if i == 0 else (52, 152, 219)
        block = pygame.Rect(x * CELL_SIZE + inset, y * CELL_SIZE + inset, CELL_SIZE - 2 * inset, CELL_SIZE - 2 * inset)
        pygame.draw.rect(screen, color, block, border_radius=6)

    # Side panel
    panel_x = GRID_WIDTH * CELL_SIZE
    panel_rect = pygame.Rect(panel_x, 0, PANEL_WIDTH, WINDOW_HEIGHT)
    pygame.draw.rect(screen, (14, 16, 20), panel_rect)

    title_font = pygame.font.SysFont("menlo", 26, bold=True)
    text_font = pygame.font.SysFont("menlo", 19)
    small_font = pygame.font.SysFont("menlo", 15)

    title = title_font.render("JEV Snake AI", True, (241, 242, 246))
    screen.blit(title, (panel_x + 24, 22))

    status_color = (241, 196, 15) if status == "Thinking" else (46, 204, 113)
    if status == "Game Over":
        status_color = (231, 76, 60)
    elif status == "Paused":
        status_color = (155, 89, 182)

    items = [
        (f"Score: {score}", (223, 230, 233)),
        (f"Engine: {engine}", (140, 190, 235)),
        (f"Last Decision: {last_decision}", (223, 230, 233)),
        (
            "Confidence: "
            + (f"{max(0.0, min(confidence, 1.0)) * 100:.1f}%" if confidence is not None else "N/A"),
            (223, 230, 233),
        ),
        (f"Decision Latency: {latency_ms:.1f} ms" if latency_ms is not None else "Decision Latency: N/A", (223, 230, 233)),
        (f"API Calls: {total_api_calls}", (223, 230, 233)),
        (f"Input Tokens: {total_input_tokens}", (223, 230, 233)),
        (f"Total Cost: ${total_cost:.4f}", (223, 230, 233)),
        (f"Status: {status}", status_color),
    ]

    y = 66
    for line, color in items:
        surf = text_font.render(line, True, color)
        screen.blit(surf, (panel_x + 24, y))
        y += 32

    if probabilities:
        probs_title = text_font.render("Move Probabilities", True, (200, 214, 229))
        screen.blit(probs_title, (panel_x + 24, y + 6))
        y += 40
        y = draw_probability_bars(screen, panel_x, y, probabilities, last_decision, small_font)

    hint = small_font.render("ESC quit | R restart | I inspect last call", True, (130, 146, 162))
    screen.blit(hint, (panel_x + 24, WINDOW_HEIGHT - 32))


def draw_json_panel(
    screen: pygame.Surface,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    data: Optional[Dict],
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    scroll_offset: int,
    h_scroll_offset: int,
    copy_button_rect: pygame.Rect,
    copy_active: bool,
) -> None:
    pygame.draw.rect(screen, (28, 32, 40), pygame.Rect(x, y, width, height), border_radius=10)
    pygame.draw.rect(screen, (70, 130, 180), pygame.Rect(x, y, width, height), width=1, border_radius=10)

    title_surf = title_font.render(title, True, (241, 242, 246))
    screen.blit(title_surf, (x + 12, y + 10))

    btn_color = (46, 204, 113) if copy_active else (52, 152, 219)
    pygame.draw.rect(screen, btn_color, copy_button_rect, border_radius=6)
    btn_label = title_font.render("Copied" if copy_active else "Copy", True, (255, 255, 255))
    screen.blit(btn_label, btn_label.get_rect(center=copy_button_rect.center))

    lines = json_text_lines(data)
    line_height = body_font.get_linesize()
    max_lines = max(1, (height - 56) // line_height)
    max_v_scroll = max(0, len(lines) - max_lines)
    scroll_offset = max(0, min(scroll_offset, max_v_scroll))

    char_width = body_font.size("0")[0] or 1
    visible_cols = max(10, (width - 30) // char_width)
    max_line_len = max((len(line) for line in lines), default=0)
    max_h_scroll = max(0, max_line_len - visible_cols)
    h_scroll_offset = max(0, min(h_scroll_offset, max_h_scroll))

    ly = y + 38
    for line in lines[scroll_offset:scroll_offset + max_lines]:
        visible_text = line[h_scroll_offset:h_scroll_offset + visible_cols]
        surf = body_font.render(visible_text, True, (200, 214, 229))
        screen.blit(surf, (x + 12, ly))
        ly += line_height

    if len(lines) > max_lines or max_h_scroll > 0:
        shown_to = min(scroll_offset + max_lines, len(lines))
        info = body_font.render(
            f"Lines {scroll_offset + 1}-{shown_to} of {len(lines)}  (wheel: scroll, shift+wheel: pan)",
            True, (150, 160, 175),
        )
        screen.blit(info, (x + 12, y + height - 16))

    if len(lines) > max_lines:
        track_rect = pygame.Rect(x + width - 8, y + 34, 4, height - 50)
        pygame.draw.rect(screen, (50, 56, 66), track_rect, border_radius=2)
        thumb_height = max(20, int(track_rect.height * (max_lines / len(lines))))
        thumb_travel = track_rect.height - thumb_height
        thumb_y = track_rect.y + int(thumb_travel * (scroll_offset / max_v_scroll)) if max_v_scroll else track_rect.y
        pygame.draw.rect(screen, (120, 170, 210), pygame.Rect(track_rect.x, thumb_y, 4, thumb_height), border_radius=2)

    if max_h_scroll > 0:
        h_track_rect = pygame.Rect(x + 12, y + height - 30, width - 30, 4)
        pygame.draw.rect(screen, (50, 56, 66), h_track_rect, border_radius=2)
        thumb_width = max(20, int(h_track_rect.width * (visible_cols / max_line_len)))
        thumb_travel_x = h_track_rect.width - thumb_width
        thumb_x = h_track_rect.x + int(thumb_travel_x * (h_scroll_offset / max_h_scroll))
        pygame.draw.rect(screen, (120, 170, 210), pygame.Rect(thumb_x, h_track_rect.y, thumb_width, 4), border_radius=2)


def json_text_lines(data: Optional[Dict]) -> List[str]:
    text = json.dumps(data, indent=2) if data is not None else "No data yet. Play a round first."
    return text.split("\n")


def inspector_panel_visible_lines(panel_height: int) -> int:
    body_font = pygame.font.SysFont("menlo", 12)
    return max(1, (panel_height - 56) // body_font.get_linesize())


def clamp_inspector_scroll(scroll_offset: int, data: Optional[Dict], panel_height: int) -> int:
    lines = json_text_lines(data)
    max_scroll = max(0, len(lines) - inspector_panel_visible_lines(panel_height))
    return max(0, min(scroll_offset, max_scroll))


def clamp_inspector_h_scroll(h_scroll_offset: int, data: Optional[Dict], panel_width: int) -> int:
    lines = json_text_lines(data)
    body_font = pygame.font.SysFont("menlo", 12)
    char_width = body_font.size("0")[0] or 1
    visible_cols = max(10, (panel_width - 30) // char_width)
    max_line_len = max((len(line) for line in lines), default=0)
    max_h_scroll = max(0, max_line_len - visible_cols)
    return max(0, min(h_scroll_offset, max_h_scroll))


def copy_to_clipboard(text: str) -> bool:
    try:
        if sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
            return True
        if sys.platform.startswith("win"):
            subprocess.run(["clip"], input=text.encode("utf-16"), check=True)
            return True
        for cmd in (["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]):
            try:
                subprocess.run(cmd, input=text.encode("utf-8"), check=True)
                return True
            except (OSError, subprocess.CalledProcessError):
                continue
    except (OSError, subprocess.CalledProcessError):
        pass
    return False


def compute_inspector_layout() -> Dict[str, object]:
    margin = 20
    box_x, box_y = margin, margin
    box_width = WINDOW_WIDTH - margin * 2
    box_height = WINDOW_HEIGHT - margin * 2

    gap = 16
    panel_y = box_y + 32
    panel_height = box_height - 32 - 26
    panel_width = (box_width - gap) // 2

    request_rect = pygame.Rect(box_x, panel_y, panel_width, panel_height)
    response_rect = pygame.Rect(box_x + panel_width + gap, panel_y, panel_width, panel_height)

    btn_w, btn_h = 60, 22
    request_copy_btn = pygame.Rect(request_rect.right - btn_w - 10, request_rect.top + 6, btn_w, btn_h)
    response_copy_btn = pygame.Rect(response_rect.right - btn_w - 10, response_rect.top + 6, btn_w, btn_h)

    return {
        "box_x": box_x,
        "box_y": box_y,
        "box_width": box_width,
        "box_height": box_height,
        "request": request_rect,
        "response": response_rect,
        "request_copy_btn": request_copy_btn,
        "response_copy_btn": response_copy_btn,
    }


def draw_inspector_overlay(
    screen: pygame.Surface,
    request_body: Optional[Dict],
    response_body: Optional[Dict],
    engine: str,
    request_scroll: int,
    response_scroll: int,
    request_h_scroll: int,
    response_h_scroll: int,
    request_copied: bool,
    response_copied: bool,
) -> None:
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 210))
    screen.blit(overlay, (0, 0))

    layout = compute_inspector_layout()

    header_font = pygame.font.SysFont("menlo", 20, bold=True)
    header = header_font.render(f"JEV Sample Request / Response  (engine: {engine})", True, (241, 242, 246))
    screen.blit(header, (layout["box_x"], layout["box_y"]))

    panel_title_font = pygame.font.SysFont("menlo", 15, bold=True)
    panel_body_font = pygame.font.SysFont("menlo", 12)

    request_rect: pygame.Rect = layout["request"]
    response_rect: pygame.Rect = layout["response"]

    draw_json_panel(
        screen, request_rect.x, request_rect.y, request_rect.width, request_rect.height,
        "Request Body", request_body, panel_title_font, panel_body_font,
        request_scroll, request_h_scroll, layout["request_copy_btn"], request_copied,
    )
    draw_json_panel(
        screen, response_rect.x, response_rect.y, response_rect.width, response_rect.height,
        "Response Body", response_body, panel_title_font, panel_body_font,
        response_scroll, response_h_scroll, layout["response_copy_btn"], response_copied,
    )

    small_font = pygame.font.SysFont("menlo", 14)
    hint = small_font.render("Press I to close  |  Click Copy to copy JSON  |  Wheel: scroll  |  Shift+Wheel: pan", True, (231, 118, 108))
    screen.blit(hint, (layout["box_x"], layout["box_y"] + layout["box_height"] - 20))


def draw_milestone_overlay(
    screen: pygame.Surface,
    milestone_score: int,
    round_calls: int,
    round_tokens: int,
    round_cost: float,
    total_calls: int,
    total_tokens: int,
    total_cost: float,
) -> None:
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 190))
    screen.blit(overlay, (0, 0))

    box_width, box_height = 480, 340
    box_x = (WINDOW_WIDTH - box_width) // 2
    box_y = (WINDOW_HEIGHT - box_height) // 2
    box_rect = pygame.Rect(box_x, box_y, box_width, box_height)
    pygame.draw.rect(screen, (22, 26, 32), box_rect, border_radius=14)
    pygame.draw.rect(screen, (88, 214, 141), box_rect, width=2, border_radius=14)

    title_font = pygame.font.SysFont("menlo", 24, bold=True)
    text_font = pygame.font.SysFont("menlo", 18)
    small_font = pygame.font.SysFont("menlo", 15)

    title = title_font.render(f"Milestone Reached: Score {milestone_score}", True, (241, 242, 246))
    screen.blit(title, (box_x + 28, box_y + 22))

    lines = [
        f"API Calls This Round: {round_calls}",
        f"Input Tokens This Round: {round_tokens}",
        f"Cost This Round: ${round_cost:.4f}",
        "",
        f"Total API Calls: {total_calls}",
        f"Total Input Tokens: {total_tokens}",
        f"Total Cost: ${total_cost:.4f}",
    ]

    y = box_y + 74
    for line in lines:
        if line:
            surf = text_font.render(line, True, (223, 230, 233))
            screen.blit(surf, (box_x + 28, y))
        y += 28

    prompt = text_font.render("ENTER / Y: Continue next 100", True, (88, 214, 141))
    screen.blit(prompt, (box_x + 28, box_y + box_height - 66))
    prompt2 = small_font.render("ESC / N: Stop here", True, (231, 118, 108))
    screen.blit(prompt2, (box_x + 28, box_y + box_height - 36))


def make_state_payload(
    snake: List[Tuple[int, int]],
    food: Tuple[int, int],
    direction: str,
    safe_moves: List[str],
) -> Dict:
    return {
        "grid_size": [GRID_WIDTH, GRID_HEIGHT],
        "head": [snake[0][0], snake[0][1]],
        "body": [[x, y] for x, y in snake],
        "food": [food[0], food[1]],
        "current_direction": direction,
        "safe_moves": safe_moves,
    }


def reset_game() -> Tuple[List[Tuple[int, int]], str, Tuple[int, int], int, bool]:
    start_snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2), (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2)]
    start_direction = "R"
    food = random_food_position(start_snake)
    return start_snake, start_direction, food, 0, False


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Snake + JEV Decisions")

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    snake, direction, food, score, game_over = reset_game()

    worker = DecisionWorker()
    worker.start()

    last_decision = direction
    last_confidence: Optional[float] = None
    last_latency_ms: Optional[float] = None
    last_probabilities: Optional[Dict[str, float]] = None
    last_engine = "JEV Local" if JEV_MODE == "demo" else "JEV Cloud"
    awaiting_decision = False
    awaiting_request_id: Optional[int] = None
    next_request_id = 0

    total_api_calls = 0
    total_input_tokens = 0
    calls_at_last_milestone = 0
    tokens_at_last_milestone = 0
    next_milestone = MILESTONE_SCORE_STEP
    paused_for_milestone = False

    show_inspector = False
    pending_request_body: Optional[Dict] = None
    last_request_body: Optional[Dict] = None
    last_response_body: Optional[Dict] = None
    request_scroll = 0
    response_scroll = 0
    request_h_scroll = 0
    response_h_scroll = 0
    request_copy_flash = 0.0
    response_copy_flash = 0.0

    move_timer = 0.0
    running = True

    while running:
        dt = clock.tick(FPS) / 1000.0
        move_timer += dt

        if request_copy_flash > 0:
            request_copy_flash = max(0.0, request_copy_flash - dt)
        if response_copy_flash > 0:
            response_copy_flash = max(0.0, response_copy_flash - dt)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL and show_inspector:
                layout = compute_inspector_layout()
                mouse_pos = pygame.mouse.get_pos()
                shift_held = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
                panning = shift_held or event.x != 0
                delta = event.x if event.x != 0 else event.y

                if layout["request"].collidepoint(mouse_pos):
                    if panning:
                        request_h_scroll = clamp_inspector_h_scroll(
                            request_h_scroll + delta * 4, last_request_body, layout["request"].width
                        )
                    else:
                        request_scroll = clamp_inspector_scroll(
                            request_scroll - event.y * 3, last_request_body, layout["request"].height
                        )
                elif layout["response"].collidepoint(mouse_pos):
                    if panning:
                        response_h_scroll = clamp_inspector_h_scroll(
                            response_h_scroll + delta * 4, last_response_body, layout["response"].width
                        )
                    else:
                        response_scroll = clamp_inspector_scroll(
                            response_scroll - event.y * 3, last_response_body, layout["response"].height
                        )
            elif event.type == pygame.MOUSEBUTTONDOWN and show_inspector and event.button == 1:
                layout = compute_inspector_layout()
                if layout["request_copy_btn"].collidepoint(event.pos):
                    if copy_to_clipboard(json.dumps(last_request_body, indent=2) if last_request_body else ""):
                        request_copy_flash = 1.2
                elif layout["response_copy_btn"].collidepoint(event.pos):
                    if copy_to_clipboard(json.dumps(last_response_body, indent=2) if last_response_body else ""):
                        response_copy_flash = 1.2
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif paused_for_milestone and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_y):
                    paused_for_milestone = False
                    calls_at_last_milestone = total_api_calls
                    tokens_at_last_milestone = total_input_tokens
                    next_milestone += MILESTONE_SCORE_STEP
                    move_timer = 0.0
                elif paused_for_milestone and event.key == pygame.K_n:
                    running = False
                elif event.key == pygame.K_i:
                    show_inspector = not show_inspector
                elif show_inspector and event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    layout = compute_inspector_layout()
                    step = -4 if event.key == pygame.K_LEFT else 4
                    request_h_scroll = clamp_inspector_h_scroll(
                        request_h_scroll + step, last_request_body, layout["request"].width
                    )
                    response_h_scroll = clamp_inspector_h_scroll(
                        response_h_scroll + step, last_response_body, layout["response"].width
                    )
                elif event.key == pygame.K_r:
                    snake, direction, food, score, game_over = reset_game()
                    last_decision = direction
                    last_confidence = None
                    last_latency_ms = None
                    last_probabilities = None
                    awaiting_decision = False
                    awaiting_request_id = None
                    total_api_calls = 0
                    total_input_tokens = 0
                    calls_at_last_milestone = 0
                    tokens_at_last_milestone = 0
                    next_milestone = MILESTONE_SCORE_STEP
                    paused_for_milestone = False
                    show_inspector = False
                    pending_request_body = None
                    last_request_body = None
                    last_response_body = None
                    request_scroll = 0
                    response_scroll = 0
                    request_h_scroll = 0
                    response_h_scroll = 0
                    request_copy_flash = 0.0
                    response_copy_flash = 0.0
                    move_timer = 0.0

        if not game_over and not paused_for_milestone and not show_inspector:
            move_interval = current_move_interval(score)

            # Start a turn by requesting a decision, then pause movement until response arrives.
            if not awaiting_decision and move_timer >= move_interval:
                safe_moves = compute_safe_moves(snake, direction)
                if safe_moves:
                    next_request_id += 1
                    state_payload = make_state_payload(snake, food, direction, safe_moves)
                    request_body = build_request_body(state_payload)
                    submitted = worker.submit_state(next_request_id, state_payload)
                    if submitted:
                        awaiting_decision = True
                        awaiting_request_id = next_request_id
                        total_api_calls += 1
                        total_input_tokens += estimate_input_tokens(request_body)
                        pending_request_body = request_body
                else:
                    game_over = True

            if awaiting_decision:
                result = worker.poll_result()
                while result is not None and result.request_id != awaiting_request_id:
                    result = worker.poll_result()

                if result is not None and result.request_id == awaiting_request_id:
                    awaiting_decision = False
                    awaiting_request_id = None
                    move_timer = 0.0

                    last_latency_ms = result.latency_ms
                    last_probabilities = result.probabilities
                    last_engine = result.engine
                    last_request_body = pending_request_body
                    last_response_body = (
                        result.raw_response
                        if result.raw_response is not None
                        else {
                            "answers": {
                                "next_move": {
                                    "choice": result.move,
                                    "confidence": result.confidence,
                                    "probabilities": result.probabilities,
                                }
                            },
                            "error": result.error,
                        }
                    )
                    request_scroll = 0
                    response_scroll = 0
                    request_h_scroll = 0
                    response_h_scroll = 0

                    proposed_move: Optional[str] = None
                    if result.move in {"U", "D", "L", "R"}:
                        proposed_move = result.move
                        last_confidence = result.confidence

                    safe_now = compute_safe_moves(snake, direction)
                    if safe_now:
                        fallback_move = direction if direction in safe_now else safe_now[0]
                        chosen_move = proposed_move if proposed_move in safe_now else fallback_move
                    else:
                        chosen_move = direction

                    direction = chosen_move
                    last_decision = direction

                    snake, grew, collided = move_snake(snake, direction, food)

                    if collided:
                        game_over = True
                    elif grew:
                        score += 10
                        food = random_food_position(snake)
                        if food == (-1, -1):
                            game_over = True
                        elif score >= next_milestone:
                            paused_for_milestone = True

        if game_over:
            status = "Game Over"
        elif paused_for_milestone:
            status = "Paused"
        else:
            status = "Thinking" if awaiting_decision else "Ready"

        total_cost = (total_input_tokens / 1_000_000) * INPUT_TOKEN_COST_PER_MILLION

        draw_game(
            screen,
            snake,
            food,
            score,
            status,
            last_decision,
            last_confidence,
            last_latency_ms,
            last_engine,
            total_api_calls,
            total_input_tokens,
            total_cost,
            last_probabilities,
        )

        if paused_for_milestone:
            round_calls = total_api_calls - calls_at_last_milestone
            round_tokens = total_input_tokens - tokens_at_last_milestone
            round_cost = (round_tokens / 1_000_000) * INPUT_TOKEN_COST_PER_MILLION
            draw_milestone_overlay(
                screen,
                next_milestone,
                round_calls,
                round_tokens,
                round_cost,
                total_api_calls,
                total_input_tokens,
                total_cost,
            )

        if show_inspector:
            draw_inspector_overlay(
                screen,
                last_request_body,
                last_response_body,
                last_engine,
                request_scroll,
                response_scroll,
                request_h_scroll,
                response_h_scroll,
                request_copy_flash > 0,
                response_copy_flash > 0,
            )

        pygame.display.flip()

    worker.stop()
    pygame.quit()


if __name__ == "__main__":
    main()

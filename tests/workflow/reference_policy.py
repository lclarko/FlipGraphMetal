"""Independent controlled-v1 controller model with SCRIPTED arithmetic.

This is a policy-ordering oracle, not a scalar arithmetic implementation. An
ArithmeticEvent supplies the outcome and number of arithmetic RNG words. This
module never claims to verify tensors, candidate lists, proposal selection, or
GPU execution. The production controller must not import this test module.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import deque
from copy import deepcopy

ARITHMETIC_REFERENCE = "9ea5bfc144b528184c32c178cae0b49bc60e8e3d"
MODEL_KIND = "scripted-arithmetic-policy-reference-v1"
U32 = (1 << 32) - 1
U64 = (1 << 64) - 1


def uint(value, maximum=U64):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("unsigned integer outside declared range")
    return value


def add(a, b):
    return uint(uint(a) + uint(b))


def multiply(a, b):
    return uint(uint(a) * uint(b))


class WorkerRNG:
    def __init__(self, seed, worker):
        uint(seed, U32)
        uint(worker, U32)
        self.state = seed ^ ((0x9e3779b9 * ((worker + 1) & U32)) & U32)
        self.state = self.state or 1
        self.draws = 0
        self.words = []

    def next(self):
        x = self.state
        x ^= (x << 13) & U32
        x ^= x >> 17
        x ^= (x << 5) & U32
        self.state = x & U32
        self.draws += 1
        self.words.append(self.state)
        return self.state

    def bounded(self, bound):
        uint(bound)
        if bound == 0:
            raise ValueError("zero bound")
        return self.next() % bound

    def interval(self, minimum, maximum):
        uint(minimum)
        uint(maximum)
        if minimum > maximum:
            raise ValueError("reversed interval")
        return add(minimum, self.bounded(add(maximum - minimum, 1)))

    def permutation(self, size):
        uint(size, 1500)
        result = list(range(size))
        for i in range(size - 1, 0, -1):
            j = self.bounded(i + 1)
            result[i], result[j] = result[j], result[i]
        return result


@dataclass(frozen=True)
class Config:
    seed: int = 0
    worker: int = 0
    mode: str = "alternatives"
    anchor: int = 23
    dimensions: tuple = (3, 3, 3)
    excursion: int = 2
    rank_capacity: int = 350
    interval_min: int = 1
    interval_max: int = 1
    reduction_q: int = 0
    flip_budget: int = 100
    control_budget: int = 100
    stagnation_limit: int = 100
    optional_quota: int = 2
    target_rank: int | None = None

    def validate(self):
        uint(self.seed, U32)
        uint(self.worker, U32)
        uint(self.reduction_q, U32)
        if self.mode not in ("alternatives", "rank-reduction"):
            raise ValueError("unknown mode")
        for value in (self.anchor, self.rank_capacity, self.stagnation_limit):
            if uint(value) == 0:
                raise ValueError("positive configuration required")
        if len(self.dimensions) != 3:
            raise ValueError("three dimensions required")
        for value in self.dimensions:
            if uint(value) == 0:
                raise ValueError("positive dimensions required")
        for value in (self.excursion, self.flip_budget, self.control_budget,
                      self.optional_quota, self.interval_min, self.interval_max):
            uint(value)
        if self.interval_min < 1 or self.interval_min > self.interval_max:
            raise ValueError("invalid expansion interval")
        add(self.interval_max - self.interval_min, 1)
        if self.target_rank is not None:
            uint(self.target_rank)
        self.ceiling()

    def ceiling(self):
        naive = multiply(multiply(self.dimensions[0], self.dimensions[1]), self.dimensions[2])
        return min(add(self.anchor, self.excursion), naive, self.rank_capacity)


@dataclass(frozen=True)
class ArithmeticEvent:
    """Explicit fixture result, not calculated arithmetic or verified validity.

    draws accounts only for the scripted arithmetic helper; the controller owns
    reduction-decision, recovery-count, operator and countdown draws separately.
    expected_operator may bind an expansion script to a chosen operator.
    """
    outcome: str
    rank: int | None = None
    draws: int = 0
    expected_operator: int | None = None
    state: dict | None = None
    removed_terms: int | None = None
    native_observations: list | None = None


class Scripts:
    def __init__(self, flips=(), reductions=(), expansions=()):
        self.queues = {"flip": deque(flips), "reduction": deque(reductions),
                       "expansion": deque(expansions)}

    def take(self, operation):
        if not self.queues[operation]:
            raise AssertionError("missing explicit arithmetic script: " + operation)
        return self.queues[operation].popleft()

    def remaining(self):
        return {key: len(value) for key, value in self.queues.items()}


class Policy:
    """One worker. Host verification/commit and parent choice are explicit inputs."""
    def __init__(self, config, parent_rank, *, parent_verified=True):
        config.validate()
        self.config = config
        self.rank = self._admit_rank(parent_rank)
        self.best = self.rank
        self.state = None
        self.initial_state = None
        self.best_state = None
        self.removed_terms = None
        self.pending_target_state = None
        self.rng = None
        self.countdown = None
        self.stagnation = 0
        self.flips = 0
        self.controls = 0
        self.pending_restart = False
        self.terminal = None
        self.dispatch_complete = True
        self.mandatory = None
        self.mandatory_committed = True
        self.optional = []
        self.optional_encounters = 0
        self.optional_drops = 0
        self.events = []
        self.applied = {"flip": 0, "reduction": 0, "expansion": 0}
        self.attempted = {"flip": 0, "reduction": 0, "expansion": 0}
        self.capture_batches = []
        if not parent_verified:
            raise ValueError("parent verification required")
        if self._target(self.rank):
            self.terminal = "existing_target_pending"
            self._record("existing_target_pending", rank=self.rank)
            return
        self.rng = WorkerRNG(config.seed, config.worker)
        self._redraw("initial_countdown")

    def _admit_rank(self, rank):
        if not 1 <= uint(rank) <= self.config.rank_capacity:
            raise ValueError("parent outside representation capacity")
        return rank

    def _record(self, kind, **values):
        self.events.append({"event": kind, **values,
                            "countdown": self.countdown, "stagnation": self.stagnation,
                            "flips": self.flips, "controls": self.controls,
                            "applied": self.applied.copy(), "attempted": self.attempted.copy(),
                            "optional_encounters": self.optional_encounters,
                            "optional_drops": self.optional_drops,
                            "rng": None if self.rng is None else self.rng.state,
                            "draws": 0 if self.rng is None else self.rng.draws})

    def _target(self, rank):
        return self.config.target_rank is not None and rank <= self.config.target_rank

    def _redraw(self, kind):
        self.countdown = self.rng.interval(self.config.interval_min, self.config.interval_max)
        self._record(kind, countdown=self.countdown)

    def _exhausted(self):
        return self.flips >= self.config.flip_budget or self.controls >= self.config.control_budget

    def _stop_budget(self):
        self.terminal = "budget_exhausted"
        self._record("budget_exhausted")

    def bind_state(self, state):
        """Bind ordered factors/candidates without changing scripted rank behavior."""
        if state["scheme"]["m"] != self.rank:
            raise ValueError("arithmetic state/rank mismatch")
        self.state = deepcopy(state)
        if self.terminal == "existing_target_pending":
            self.pending_target_state = deepcopy(state)
        if self.initial_state is None:
            self.initial_state = deepcopy(state)
        if self.best_state is None:
            self.best_state = deepcopy(state)
            self.removed_terms = {"flip": 0, "reduction": 0, "total": 0}

    def _observe(self, operation):
        if self.rank < self.best:
            self.best = self.rank
            self.best_state = deepcopy(self.state)
            self.stagnation = 0
        sample = {"rank": self.rank, "operation": operation, "control": self.controls}
        if self.state is not None:
            sample["state"] = deepcopy(self.state)
        if self.mandatory is None or self.rank < self.mandatory["rank"]:
            self.mandatory = deepcopy(sample)
            self.mandatory_committed = False
        eligible = (self.rank == self.config.anchor if self.config.mode == "alternatives"
                    else self.rank < self.config.anchor)
        if eligible:
            self.optional_encounters += 1
            if len(self.optional) < self.config.optional_quota:
                self.optional.append(deepcopy(sample))
            else:
                self.optional_drops += 1
        self._record("observed", **sample, best=self.best)
        if self._target(self.rank):
            self.terminal = "target_pending"
            self._record("target_pending", rank=self.rank)

    def _arithmetic(self, operation, script, operator=None):
        self.attempted[operation] = add(self.attempted[operation], 1)
        event = script.take(operation)
        uint(event.draws, 1000000)
        if event.expected_operator is not None and event.expected_operator != operator:
            raise AssertionError("scripted expansion operator mismatch")
        allowed = {"applied", "unsuccessful", "tuple_rejection", "coefficient_rejection",
                   "proposal_exhausted", "capacity_error"}
        if event.outcome not in allowed:
            raise ValueError("unknown scripted outcome")
        for _ in range(event.draws):
            self.rng.next()
        if event.outcome == "capacity_error":
            self.dispatch_complete = False
            self.terminal = "capacity_error"
            self._record("capacity_error", operation=operation,
                         **({"native_observations": deepcopy(event.native_observations)}
                            if event.native_observations is not None else {}))
            return False
        if event.outcome == "applied":
            if event.rank is None:
                raise ValueError("applied script requires rank")
            rank = self._admit_rank(event.rank)
            if operation == "expansion" and rank > add(self.rank, 1):
                raise ValueError("script expands by more than one primitive")
            if operation != "expansion" and rank > self.rank:
                raise ValueError("flip/reduction cannot increase rank")
            if operation == "reduction" and rank >= self.rank:
                raise ValueError("applied explicit reduction must lower rank")
            self.rank = rank
            self.applied[operation] += 1
        elif event.rank is not None:
            raise ValueError("unsuccessful script cannot change rank")
        if event.state is not None:
            self.bind_state(event.state)
        if event.removed_terms is not None:
            removed = uint(event.removed_terms)
            if operation not in ("flip", "reduction") or self.removed_terms is None:
                raise ValueError("removal accounting requires bound flip/reduction state")
            self.removed_terms[operation] = add(self.removed_terms[operation], removed)
            self.removed_terms["total"] = add(self.removed_terms["total"], removed)
        self._record(operation, outcome=event.outcome, rank=self.rank, operator=operator,
                     **({"removed_terms": event.removed_terms} if event.removed_terms is not None else {}),
                     **({"native_observations": deepcopy(event.native_observations)}
                        if event.native_observations is not None else {}))
        if event.outcome == "applied":
            self._observe(operation)
        return event.outcome == "applied"

    def _expansion(self, script, recovery):
        applied = 0
        if add(self.rank, 1) > self.config.ceiling():
            self._record("rank_blocked", operation="recovery" if recovery else "scheduled")
        else:
            count = 1 + self.rng.bounded(2) if recovery else 1
            self._record("expansion_event", recovery=recovery, count=count)
            for _ in range(count):
                if add(self.rank, 1) > self.config.ceiling():
                    self._record("rank_blocked", operation="primitive")
                    break
                operator = self.rng.bounded(3)
                applied += self._arithmetic("expansion", script, operator)
                if self.terminal:
                    return
        self._redraw("event_countdown")
        if recovery and applied == 0:
            self.pending_restart = True
            self._record("restart_requested", reason="unsuccessful_recovery")

    def step(self, script):
        if self.terminal:
            return
        if self._exhausted():
            self._stop_budget()
            return
        if self.pending_restart:
            return
        self.controls += 1
        self.flips += 1
        self.countdown -= 1
        self.stagnation += 1
        successful = self._arithmetic("flip", script)
        if self.terminal:
            return
        if self._exhausted():
            self._stop_budget()
            return
        if successful:
            word = self.rng.next()
            self._record("reduction_decision", word=word, threshold=self.config.reduction_q)
            if word <= self.config.reduction_q:
                self._arithmetic("reduction", script)
                if self.terminal:
                    return
        if not successful or self.countdown <= 0:
            self._expansion(script, recovery=not successful)
            if self.terminal:
                return
        if self.stagnation >= self.config.stagnation_limit and not self.pending_restart:
            self.pending_restart = True
            self._record("restart_requested", reason="stagnation")

    def commit_observations(self, *, verified, durable):
        if not self.dispatch_complete or not verified or not durable:
            raise ValueError("complete verified durable evidence required")
        self.mandatory_committed = True
        if self.terminal in ("target_pending", "existing_target_pending"):
            self.terminal = "target_met"
            self._record("target_met")

    def batch_boundary(self):
        """Clear capture slots only after host commitment; preserve policy/RNG."""
        if not self.dispatch_complete or not self.mandatory_committed:
            raise ValueError("uncommitted or incomplete batch")
        self.capture_batches.append({"mandatory": deepcopy(self.mandatory),
                                     "optional": deepcopy(self.optional),
                                     "encounters": self.optional_encounters, "drops": self.optional_drops})
        self.mandatory = None
        self.optional = []
        self.optional_encounters = 0
        self.optional_drops = 0

    def install_restart(self, parent_rank, *, parent_verified=True, parent_state=None):
        if not self.pending_restart or self.terminal:
            raise ValueError("no installable restart request")
        if not self.dispatch_complete or not self.mandatory_committed:
            raise ValueError("prior mandatory commitment required")
        rank = self._admit_rank(parent_rank)
        if self.state is not None and parent_state is None:
            raise ValueError("bound arithmetic restart requires full parent state")
        if parent_state is not None and parent_state["scheme"]["m"] != rank:
            raise ValueError("restart state/rank mismatch")
        if not parent_verified:
            raise ValueError("parent verification required")
        if self._target(rank):
            self.pending_target_state = deepcopy(parent_state)
            self.terminal = "existing_target_pending"
            self._record("existing_target_pending", rank=rank)
            return
        if self._exhausted():
            self._stop_budget()
            return
        self.controls += 1
        self.rank = self.best = rank
        if parent_state is not None:
            self.state = deepcopy(parent_state)
            self.best_state = deepcopy(parent_state)
        self.stagnation = 0
        self.pending_restart = False
        self._record("parent_installed", rank=rank)
        self._redraw("restart_countdown")

    def charge_stage_credit(self):
        """Host chooses lowest-ID worker with credit; this helper charges one."""
        if self.controls >= self.config.control_budget:
            raise ValueError("no stage control credit")
        self.controls += 1
        self._record("stage_credit")

    def comparison_record(self):
        """Versioned worker comparison; host pool/journal decisions are not modeled."""
        words = [] if self.rng is None else self.rng.words
        initial = None if self.rng is None else WorkerRNG(self.config.seed, self.config.worker).state
        events, previous_draws, previous_rng = [], 0, initial
        for event in self.events:
            entry = deepcopy(event)
            entry['rng_before'] = previous_rng
            entry['words'] = words[previous_draws:entry['draws']]
            events.append(entry)
            previous_draws, previous_rng = entry['draws'], entry['rng']
        native = [item for event in events for item in event.get('native_observations', [])]
        native_outcomes = {}
        for item in native:
            key = item['result']['outcome']
            native_outcomes[key] = native_outcomes.get(key, 0) + 1
        return {'schema': 'fgm-controlled-worker-comparison-v1',
                'arithmetic_reference': ARITHMETIC_REFERENCE,
                'config': asdict(self.config), 'worker_rng_initial': initial,
                'initial_state': deepcopy(self.initial_state),
                'native_call_count': len(native), 'native_outcome_counts': native_outcomes,
                'worker_rng_words': list(words), 'summary': self.summary(),
                'timers': {'countdown': self.countdown, 'stagnation': self.stagnation},
                'remaining': {'flips': self.config.flip_budget-self.flips,
                              'controls': self.config.control_budget-self.controls},
                'attempted': self.attempted.copy(), 'applied': self.applied.copy(),
                'dispatch_complete': self.dispatch_complete,
                'mandatory_committed': self.mandatory_committed,
                'completed_capture_batches': deepcopy(self.capture_batches),
                'events': events,
                'host_selection_and_journal': 'not modeled'}

    def summary(self):
        return {**({"pending_target_state": deepcopy(self.pending_target_state)}
                    if self.pending_target_state is not None else {}),
                **({"state": deepcopy(self.state), "best_state": deepcopy(self.best_state),
                    "removed_terms": deepcopy(self.removed_terms)} if self.state is not None else {}),
                "rank": self.rank, "best": self.best, "countdown": self.countdown,
                "rng": None if self.rng is None else self.rng.state,
                "draws": 0 if self.rng is None else self.rng.draws,
                "flips": self.flips, "controls": self.controls,
                "terminal": self.terminal, "pending_restart": self.pending_restart,
                "mandatory": deepcopy(self.mandatory), "optional": deepcopy(self.optional),
                "optional_encounters": self.optional_encounters,
                "optional_drops": self.optional_drops,
                "events": [item["event"] for item in self.events]}

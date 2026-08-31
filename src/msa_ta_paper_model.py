"""
Simple modular implementation of the Mental-State-Aware Task Allocation model.

This file implements the computational model described in the paper draft:
tasks, robots, operators, mental-state dynamics, feasibility constraints,
stage cost, and a simple receding-horizon planner.

The optimizer is intentionally simple. It uses a beam-search approximation over
candidate feasible start actions, so the model can run without external MILP
solvers. The functions are written so that a stronger solver can later replace
`solve_receding_horizon`.
"""

from __future__ import annotations

import random
import argparse
import csv
import json
from dataclasses import asdict, dataclass
from itertools import product
from math import exp
from pathlib import Path
from statistics import mean, stdev
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:
    from ortools.sat.python import cp_model  # type: ignore
except Exception:  # pragma: no cover
    cp_model = None

try:
    import optuna  # type: ignore
except Exception:  # pragma: no cover
    optuna = None


TaskId = int
OperatorId = int
RobotId = int
Action = Tuple[Tuple[OperatorId, RobotId, TaskId], ...]
MethodName = str


METHODS = (
    "random_feasible",
    "greedy_priority_deadline",
    "myopic_optimization",
    "human_unaware_rh",
    "proposed_msa_rh",
)


@dataclass(frozen=True)
class Task:
    task_id: int
    processing_time: int
    priority: float
    due: int
    release: int = 0


@dataclass
class ModelParams:
    # Objective weights.
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    eta: float = 1.0
    rho: float = 1.0
    kappa: float = 1.0
    delta_w: float = 1.0

    # Stress dynamics.
    alpha_s: float = 0.3
    beta_L: float = 0.2
    beta_P: float = 0.2
    beta_W: float = 0.1
    beta_R: float = 0.05

    # Fatigue dynamics.
    zeta_L: float = 0.05
    zeta_S: float = 0.05
    zeta_C: float = 0.05
    zeta_T: float = 0.05
    zeta_D: float = 0.02
    cmax: int = 5

    # Impairment score.
    omega_f: float = 0.45
    omega_s: float = 0.40
    omega_fs: float = 0.15

    # Belief mapping.
    sigma_z: float = 0.12
    n_clusters: int = 3

    # Approximate solver settings.
    beam_width: int = 10
    max_actions_per_step: int = 20


@dataclass
class ProblemData:
    tasks: List[Task]
    n_operators: int
    n_robots: int
    horizon: int
    mission_horizon: int
    eligibility: List[List[int]]  # eligibility[robot][task] in {0, 1}
    base_lmax: List[float]  # shared fresh-to-degraded capacity profile
    robot_availability: Optional[List[List[int]]] = None  # availability[time][robot]
    lmax_schedule: Optional[Callable[[int, int, int, "ProblemData"], float]] = None


def lmax_value(data: ProblemData, operator: int, cluster: int, time: int) -> float:
    """Return Lmax for one operator, cluster, and time.

    The default paper simplification uses the same capacity profile for every
    operator. Scenario-specific schedules can override this, for example to
    reduce operator 2's capacity during the dynamic-disturbance scenario.
    """
    if data.lmax_schedule is not None:
        return data.lmax_schedule(operator, cluster, time, data)
    return data.base_lmax[cluster]


def lmax_profile(data: ProblemData, operator: int, time: int) -> List[float]:
    return [lmax_value(data, operator, k, time) for k in range(len(data.base_lmax))]


def validate_problem(data: ProblemData, params: ModelParams) -> None:
    if params.n_clusters != len(data.base_lmax):
        raise ValueError("params.n_clusters must match len(data.base_lmax).")
    if len(data.eligibility) != data.n_robots:
        raise ValueError("eligibility must contain one row per robot.")
    for row in data.eligibility:
        if len(row) != len(data.tasks):
            raise ValueError("each eligibility row must contain one value per task.")


@dataclass
class State:
    time: int
    task_state: List[int]  # 0 not started, 1 in progress, 2 completed
    remaining_time: List[int]
    robot_task: List[int]  # task id, or -1 if idle
    robot_operator: List[int]  # operator id, or -1 if idle
    fatigue: List[float]
    stress: List[float]
    supervision_counter: List[int]
    assignment_memory: List[List[int]]  # memory[operator][task] in {0, 1}
    completion_time: List[int]

    def clone(self) -> "State":
        return State(
            time=self.time,
            task_state=self.task_state.copy(),
            remaining_time=self.remaining_time.copy(),
            robot_task=self.robot_task.copy(),
            robot_operator=self.robot_operator.copy(),
            fatigue=self.fatigue.copy(),
            stress=self.stress.copy(),
            supervision_counter=self.supervision_counter.copy(),
            assignment_memory=[row.copy() for row in self.assignment_memory],
            completion_time=self.completion_time.copy(),
        )


def initialize_state(data: ProblemData) -> State:
    n_tasks = len(data.tasks)
    return State(
        time=0,
        task_state=[0] * n_tasks,
        remaining_time=[0] * n_tasks,
        robot_task=[-1] * data.n_robots,
        robot_operator=[-1] * data.n_robots,
        fatigue=[0.0] * data.n_operators,
        stress=[0.0] * data.n_operators,
        supervision_counter=[0] * data.n_operators,
        assignment_memory=[[0] * n_tasks for _ in range(data.n_operators)],
        completion_time=[data.mission_horizon + 1] * n_tasks,
    )


def delta_zero(value: int | float) -> int:
    return 1 if value == 0 else 0


def heaviside(value: int | float) -> int:
    return 1 if value >= 0 else 0


def clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def known_tasks(data: ProblemData, time: int) -> List[int]:
    return [task.task_id for task in data.tasks if task.release <= time]


def operator_load(state: State, data: ProblemData) -> List[int]:
    return [
        sum(1 for assigned_operator in state.robot_operator if assigned_operator == i)
        for i in range(data.n_operators)
    ]


def time_pressure(state: State, data: ProblemData, known_time: int) -> float:
    known = known_tasks(data, known_time)
    denominator = sum(data.tasks[d].priority for d in known)
    if denominator <= 0:
        return 0.0

    numerator = 0.0
    for d in known:
        task = data.tasks[d]
        unfinished = state.task_state[d] != 2
        overdue = state.time > task.due
        if unfinished and overdue:
            numerator += task.priority
    return numerator / denominator


def duration_input(state: State, data: ProblemData) -> List[float]:
    values: List[float] = []
    for i in range(data.n_operators):
        total = 0.0
        for task in data.tasks:
            d = task.task_id
            if state.assignment_memory[i][d] and state.task_state[d] == 1:
                elapsed = task.processing_time - state.remaining_time[d]
                total += elapsed / max(task.processing_time, 1)
        values.append(total)
    return values


def switching_signal(action: Action, data: ProblemData) -> List[int]:
    signal = [0] * data.n_operators
    for i, _, _ in action:
        signal[i] += 1
    return signal


def impairment_scores(state: State, params: ModelParams) -> List[float]:
    z_values = []
    for f_i, s_i in zip(state.fatigue, state.stress):
        z_values.append(params.omega_f * f_i + params.omega_s * s_i + params.omega_fs * f_i * s_i)
    return [clip01(z) for z in z_values]


def belief_vectors(state: State, params: ModelParams) -> List[List[float]]:
    z_values = impairment_scores(state, params)
    n_clusters = params.n_clusters
    thresholds = [k / n_clusters for k in range(n_clusters + 1)]
    midpoints = [(thresholds[k] + thresholds[k + 1]) / 2.0 for k in range(n_clusters)]
    sigma_sq = max(params.sigma_z, 1e-9) ** 2

    beliefs: List[List[float]] = []
    for z_i in z_values:
        raw = [exp(-((z_i - midpoint) ** 2) / (2.0 * sigma_sq)) for midpoint in midpoints]
        normalizer = sum(raw)
        beliefs.append([value / normalizer for value in raw])
    return beliefs


def effective_capacity(
    state: State,
    data: ProblemData,
    params: ModelParams,
    mental_aware_capacity: bool = True,
) -> List[float]:
    if not mental_aware_capacity:
        return [lmax_value(data, i, 0, state.time) for i in range(data.n_operators)]

    beliefs = belief_vectors(state, params)
    capacities: List[float] = []
    for i in range(data.n_operators):
        capacities.append(sum(beliefs[i][k] * lmax_value(data, i, k, state.time) for k in range(params.n_clusters)))
    return capacities


def robot_available(data: ProblemData, robot: int, time: int) -> bool:
    if data.robot_availability is None:
        return True
    if time < 0 or time >= len(data.robot_availability):
        return True
    return bool(data.robot_availability[time][robot])


def action_is_feasible(
    state: State,
    action: Action,
    data: ProblemData,
    params: ModelParams,
    known_time: int,
    mental_aware_capacity: bool = True,
) -> bool:
    known = set(known_tasks(data, known_time))
    used_robots = set()
    used_tasks = set()
    load = operator_load(state, data)
    capacity = effective_capacity(state, data, params, mental_aware_capacity)

    for i, j, d in action:
        if not (0 <= i < data.n_operators and 0 <= j < data.n_robots and 0 <= d < len(data.tasks)):
            return False
        if d not in known or data.tasks[d].release > state.time:
            return False
        if state.task_state[d] != 0:
            return False
        if state.robot_task[j] != -1:
            return False
        if not robot_available(data, j, state.time):
            return False
        if data.eligibility[j][d] != 1:
            return False
        if j in used_robots or d in used_tasks:
            return False

        used_robots.add(j)
        used_tasks.add(d)
        load[i] += 1
        if load[i] > capacity[i] + 1e-9:
            return False

    return True


def constraints_satisfied(
    state: State,
    action: Action,
    data: ProblemData,
    params: ModelParams,
    known_time: int,
    mental_aware_capacity: bool = True,
) -> bool:
    return action_is_feasible(state, action, data, params, known_time, mental_aware_capacity)


def mental_state_cost(state: State, params: ModelParams) -> float:
    beliefs = belief_vectors(state, params)
    total = 0.0
    for belief in beliefs:
        for k, b_ik in enumerate(belief):
            total += params.delta_w * (k**2) * b_ik
    return total


def unfinished_cost(state: State, data: ProblemData, known_time: int) -> float:
    total = 0.0
    for d in known_tasks(data, known_time):
        if state.task_state[d] != 2:
            total += data.tasks[d].priority
    return total


def overdue_cost(state: State, data: ProblemData, known_time: int) -> float:
    total = 0.0
    for d in known_tasks(data, known_time):
        task = data.tasks[d]
        if state.task_state[d] != 2 and state.time > task.due:
            total += task.priority
    return total


def assignment_risk_cost(state: State, action: Action, data: ProblemData, params: ModelParams) -> float:
    beliefs = belief_vectors(state, params)
    total = 0.0
    for i, _, d in action:
        lambda_bar = 0.0
        for k, b_ik in enumerate(beliefs[i]):
            lambda_k = k / max(params.n_clusters - 1, 1)
            lambda_bar += lambda_k * b_ik
        total += lambda_bar * data.tasks[d].priority
    return total


def start_reward(state: State, action: Action, data: ProblemData, params: ModelParams) -> float:
    total = 0.0
    for _, _, d in action:
        task = data.tasks[d]
        timely_start = heaviside(task.due - state.time - task.processing_time)
        total += task.priority * (1.0 + params.kappa * timely_start)
    return total


def stage_cost(
    state: State,
    action: Action,
    data: ProblemData,
    params: ModelParams,
    known_time: int,
    use_mental_objective: bool = True,
) -> float:
    c_u = unfinished_cost(state, data, known_time)
    c_m = mental_state_cost(state, params) if use_mental_objective else 0.0
    c_r = assignment_risk_cost(state, action, data, params) if use_mental_objective else 0.0
    c_o = overdue_cost(state, data, known_time)
    reward = start_reward(state, action, data, params)
    return params.alpha * c_u + params.beta * c_m + params.gamma * c_r + params.eta * c_o - params.rho * reward


def exact_one_step_action(
    state: State,
    data: ProblemData,
    params: ModelParams,
    known_time: int,
    mental_aware_capacity: bool = True,
    use_mental_objective: bool = True,
) -> Optional[Action]:
    """Solve the current one-step assignment exactly with CP-SAT.

    This is exact only for the single decision step with current mental-state
    quantities treated as constants. The full receding-horizon model remains
    nonlinear because future belief, stress, fatigue, and capacity depend on
    previous binary actions through clipping, products, and softmax mapping.
    """
    if cp_model is None:
        return None

    pending = available_task_ids(state, data, known_time)
    if not pending:
        return tuple()

    current_load = operator_load(state, data)
    capacities = effective_capacity(state, data, params, mental_aware_capacity)
    beliefs = belief_vectors(state, params)

    model = cp_model.CpModel()
    variables: Dict[Tuple[int, int, int], object] = {}
    objective_terms = []
    scale = 1000

    for d in pending:
        for i in range(data.n_operators):
            for j in range(data.n_robots):
                triple = (i, j, d)
                if not action_is_feasible(state, (triple,), data, params, known_time, mental_aware_capacity):
                    continue
                var = model.NewBoolVar(f"a_{i}_{j}_{d}")
                variables[triple] = var

                task = data.tasks[d]
                risk = 0.0
                if use_mental_objective:
                    for k, b_ik in enumerate(beliefs[i]):
                        lambda_k = k / max(params.n_clusters - 1, 1)
                        risk += lambda_k * b_ik * task.priority
                reward = task.priority * (1.0 + params.kappa * heaviside(task.due - state.time - task.processing_time))
                coeff = params.gamma * risk - params.rho * reward
                objective_terms.append(int(round(scale * coeff)) * var)

    if not variables:
        return tuple()

    for j in range(data.n_robots):
        model.Add(sum(var for (i2, j2, d2), var in variables.items() if j2 == j) <= 1)
    for d in pending:
        model.Add(sum(var for (i2, j2, d2), var in variables.items() if d2 == d) <= 1)
    for i in range(data.n_operators):
        allowed_new = max(0, int(capacities[i] + 1e-9) - current_load[i])
        model.Add(sum(var for (i2, j2, d2), var in variables.items() if i2 == i) <= allowed_new)

    model.Minimize(sum(objective_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return tuple()

    action = tuple(triple for triple, var in variables.items() if solver.BooleanValue(var))
    return action


def update_assignment_memory(state: State, action: Action) -> None:
    for i, _, d in action:
        state.assignment_memory[i][d] = 1


def update_task_robot_states(old: State, new: State, action: Action, data: ProblemData) -> None:
    for j, d in enumerate(old.robot_task):
        if d == -1:
            continue
        if old.remaining_time[d] > 1:
            new.task_state[d] = 1
            new.remaining_time[d] = old.remaining_time[d] - 1
            new.robot_task[j] = d
            new.robot_operator[j] = old.robot_operator[j]
        else:
            new.task_state[d] = 2
            new.remaining_time[d] = 0
            new.completion_time[d] = min(new.completion_time[d], old.time + 1)
            new.robot_task[j] = -1
            new.robot_operator[j] = -1

    for i, j, d in action:
        processing_time = data.tasks[d].processing_time
        if processing_time <= 1:
            new.task_state[d] = 2
            new.remaining_time[d] = 0
            new.completion_time[d] = min(new.completion_time[d], old.time + 1)
            new.robot_task[j] = -1
            new.robot_operator[j] = -1
        else:
            new.task_state[d] = 1
            new.remaining_time[d] = processing_time - 1
            new.robot_task[j] = d
            new.robot_operator[j] = i


def update_operator_states(old: State, new: State, action: Action, data: ProblemData, params: ModelParams, known_time: int) -> None:
    load = operator_load(old, data)
    pressure = time_pressure(old, data, known_time)
    switches = switching_signal(action, data)
    durations = duration_input(old, data)

    for i in range(data.n_operators):
        fresh_capacity = max(lmax_value(data, i, 0, old.time), 1e-9)
        normalized_load = load[i] / fresh_capacity
        if load[i] > 0:
            new.supervision_counter[i] = old.supervision_counter[i] + 1
        else:
            new.supervision_counter[i] = 0

        raw_stress = (
            (1.0 - params.alpha_s) * old.stress[i]
            + params.beta_L * normalized_load
            + params.beta_P * pressure
            + params.beta_W * switches[i] / max(data.n_robots, 1)
            - params.beta_R * (1.0 - normalized_load)
        )
        new.stress[i] = clip01(raw_stress)

        raw_fatigue = (
            old.fatigue[i]
            + params.zeta_L * normalized_load
            + params.zeta_S * old.stress[i]
            + params.zeta_C * min(1.0, old.supervision_counter[i] / max(params.cmax, 1))
            + params.zeta_T * durations[i]
            - params.zeta_D * (1.0 - normalized_load)
        )
        new.fatigue[i] = clip01(raw_fatigue)


def apply_action(
    state: State,
    action: Action,
    data: ProblemData,
    params: ModelParams,
    known_time: Optional[int] = None,
    mental_aware_capacity: bool = True,
) -> State:
    if known_time is None:
        known_time = state.time
    if not constraints_satisfied(state, action, data, params, known_time, mental_aware_capacity):
        action = tuple()

    new = state.clone()
    new.time = state.time + 1
    update_assignment_memory(new, action)
    update_task_robot_states(state, new, action, data)
    update_operator_states(state, new, action, data, params, known_time)
    return new


def available_task_ids(state: State, data: ProblemData, known_time: int) -> List[int]:
    return [
        d
        for d in known_tasks(data, known_time)
        if data.tasks[d].release <= state.time and state.task_state[d] == 0
    ]


def possible_assignments_for_task(state: State, data: ProblemData, task_id: int) -> List[Tuple[int, int, int]]:
    triples = []
    for i, j in product(range(data.n_operators), range(data.n_robots)):
        triples.append((i, j, task_id))
    return triples


def generate_candidate_actions(
    state: State,
    data: ProblemData,
    params: ModelParams,
    known_time: int,
    mental_aware_capacity: bool = True,
    use_mental_objective: bool = True,
) -> List[Action]:
    tasks = sorted(
        available_task_ids(state, data, known_time),
        key=lambda d: (-data.tasks[d].priority, data.tasks[d].due, data.tasks[d].processing_time),
    )

    actions: Dict[Action, None] = {tuple(): None}
    greedy_action: List[Tuple[int, int, int]] = []

    for d in tasks:
        best_single: Optional[Action] = None
        best_cost = float("inf")
        for triple in possible_assignments_for_task(state, data, d):
            candidate = tuple(greedy_action + [triple])
            if action_is_feasible(state, candidate, data, params, known_time, mental_aware_capacity):
                cost = stage_cost(state, candidate, data, params, known_time, use_mental_objective)
                if cost < best_cost:
                    best_cost = cost
                    best_single = candidate
        if best_single is not None:
            greedy_action = list(best_single)
            actions[best_single] = None

    for d in tasks[: data.n_robots]:
        for triple in possible_assignments_for_task(state, data, d):
            action = (triple,)
            if action_is_feasible(state, action, data, params, known_time, mental_aware_capacity):
                actions[action] = None

    scored = [(stage_cost(state, action, data, params, known_time, use_mental_objective), action) for action in actions]
    scored.sort(key=lambda item: item[0])
    return [action for _, action in scored[: max(1, params.max_actions_per_step)]]


def solve_receding_horizon(
    state: State,
    data: ProblemData,
    params: ModelParams,
    planning_horizon: Optional[int] = None,
    mental_aware_capacity: bool = True,
    use_mental_objective: bool = True,
) -> Action:
    known_time = state.time
    beam: List[Tuple[float, State, Action]] = [(0.0, state.clone(), tuple())]
    horizon = data.horizon if planning_horizon is None else planning_horizon

    for depth in range(horizon):
        next_beam: List[Tuple[float, State, Action]] = []
        for cumulative_cost, beam_state, first_action in beam:
            if beam_state.time >= data.mission_horizon:
                next_beam.append((cumulative_cost, beam_state, first_action))
                continue

            for action in generate_candidate_actions(
                beam_state,
                data,
                params,
                known_time,
                mental_aware_capacity,
                use_mental_objective,
            ):
                cost = stage_cost(beam_state, action, data, params, known_time, use_mental_objective)
                next_state = apply_action(beam_state, action, data, params, known_time, mental_aware_capacity)
                selected_first = action if depth == 0 else first_action
                next_beam.append((cumulative_cost + cost, next_state, selected_first))

        next_beam.sort(key=lambda item: item[0])
        beam = next_beam[: max(1, params.beam_width)]

    return beam[0][2] if beam else tuple()


def build_action_from_task_order(
    state: State,
    data: ProblemData,
    params: ModelParams,
    known_time: int,
    task_order: Sequence[int],
    mental_aware_capacity: bool = True,
    rng: Optional[random.Random] = None,
) -> Action:
    action_items: List[Tuple[int, int, int]] = []
    for d in task_order:
        triples = possible_assignments_for_task(state, data, d)
        if rng is not None:
            rng.shuffle(triples)

        best_triple: Optional[Tuple[int, int, int]] = None
        best_load = float("inf")
        for triple in triples:
            candidate = tuple(action_items + [triple])
            if not action_is_feasible(state, candidate, data, params, known_time, mental_aware_capacity):
                continue
            load_after = operator_load(state, data)[triple[0]] + sum(1 for i, _, _ in candidate if i == triple[0])
            if load_after < best_load:
                best_load = load_after
                best_triple = triple

        if best_triple is not None:
            action_items.append(best_triple)

    return tuple(action_items)


def random_feasible_action(
    state: State,
    data: ProblemData,
    params: ModelParams,
    known_time: int,
    rng: random.Random,
) -> Action:
    task_order = available_task_ids(state, data, known_time)
    rng.shuffle(task_order)
    return build_action_from_task_order(state, data, params, known_time, task_order, True, rng)


def greedy_priority_deadline_action(state: State, data: ProblemData, params: ModelParams, known_time: int) -> Action:
    task_order = sorted(
        available_task_ids(state, data, known_time),
        key=lambda d: (-data.tasks[d].priority, data.tasks[d].due, data.tasks[d].processing_time),
    )
    return build_action_from_task_order(state, data, params, known_time, task_order, True)


def select_action(
    method: MethodName,
    state: State,
    data: ProblemData,
    params: ModelParams,
    rng: random.Random,
) -> Tuple[Action, bool]:
    """Select one action and return the capacity mode used to validate it."""
    known_time = state.time
    if method == "random_feasible":
        return random_feasible_action(state, data, params, known_time, rng), True
    if method == "greedy_priority_deadline":
        return greedy_priority_deadline_action(state, data, params, known_time), True
    if method == "myopic_optimization":
        exact_action = exact_one_step_action(
            state,
            data,
            params,
            known_time,
            mental_aware_capacity=True,
            use_mental_objective=True,
        )
        if exact_action is not None:
            return exact_action, True
        return solve_receding_horizon(
            state,
            data,
            params,
            planning_horizon=1,
            mental_aware_capacity=True,
            use_mental_objective=True,
        ), True
    if method == "human_unaware_rh":
        return solve_receding_horizon(
            state,
            data,
            params,
            planning_horizon=data.horizon,
            mental_aware_capacity=False,
            use_mental_objective=False,
        ), False
    if method == "proposed_msa_rh":
        return solve_receding_horizon(
            state,
            data,
            params,
            planning_horizon=data.horizon,
            mental_aware_capacity=True,
            use_mental_objective=True,
        ), True
    raise ValueError(f"Unknown method: {method}. Valid methods are: {', '.join(METHODS)}")


def all_tasks_completed(state: State, data: ProblemData) -> bool:
    return all(state.task_state[task.task_id] == 2 for task in data.tasks)


def run_simulation(
    data: ProblemData,
    params: ModelParams,
    method: MethodName = "proposed_msa_rh",
    seed: int = 0,
) -> Tuple[List[State], List[Action]]:
    validate_problem(data, params)
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Valid methods are: {', '.join(METHODS)}")

    rng = random.Random(seed)
    state = initialize_state(data)
    states = [state.clone()]
    actions: List[Action] = []

    while state.time < data.mission_horizon and not all_tasks_completed(state, data):
        action, mental_aware_capacity = select_action(method, state, data, params, rng)
        actions.append(action)
        state = apply_action(state, action, data, params, mental_aware_capacity=mental_aware_capacity)
        states.append(state.clone())

    return states, actions


def run_receding_horizon(data: ProblemData, params: ModelParams) -> Tuple[List[State], List[Action]]:
    return run_simulation(data, params, method="proposed_msa_rh")


def total_weighted_tardiness(state: State, data: ProblemData) -> float:
    return sum(
        task.priority * max(0, state.completion_time[task.task_id] - task.due)
        for task in data.tasks
    )


def deadline_miss_rate(state: State, data: ProblemData) -> float:
    if not data.tasks:
        return 0.0
    misses = sum(1 for task in data.tasks if state.completion_time[task.task_id] > task.due)
    return misses / len(data.tasks)


def weighted_completion_ratio(state: State, data: ProblemData) -> float:
    total_priority = sum(task.priority for task in data.tasks)
    if total_priority <= 0:
        return 0.0
    completed_priority = sum(
        task.priority
        for task in data.tasks
        if state.completion_time[task.task_id] <= data.mission_horizon
    )
    return completed_priority / total_priority


def cumulative_impairment(states: Sequence[State], params: ModelParams) -> float:
    return sum(sum(impairment_scores(state, params)) for state in states)


def peak_impairment(states: Sequence[State], params: ModelParams) -> float:
    if not states:
        return 0.0
    return max(max(impairment_scores(state, params)) for state in states)


def evaluate_run(states: Sequence[State], data: ProblemData, params: ModelParams) -> Dict[str, float]:
    final_state = states[-1]
    return {
        "TWT": total_weighted_tardiness(final_state, data),
        "DMR": deadline_miss_rate(final_state, data),
        "WCR": weighted_completion_ratio(final_state, data),
        "CI": cumulative_impairment(states, params),
        "PI": peak_impairment(states, params),
    }


def generate_priorities(n_tasks: int, rng: random.Random) -> List[float]:
    n_low = round(0.4 * n_tasks)
    n_medium = round(0.4 * n_tasks)
    n_high = n_tasks - n_low - n_medium
    priorities = [1.0] * n_low + [2.0] * n_medium + [4.0] * n_high
    rng.shuffle(priorities)
    return priorities


def generate_initial_tasks(scenario_name: str, mission_horizon: int, n_tasks: int, rng: random.Random) -> List[Task]:
    priorities = generate_priorities(n_tasks, rng)
    tasks: List[Task] = []
    for d in range(n_tasks):
        duration = rng.randint(2, 6)
        priority = priorities[d]

        if scenario_name == "overload_finite_shift":
            due_low = max(3, duration + 1)
            due_high = mission_horizon + 4
        else:
            due_low = max(4, duration + 2)
            due_high = mission_horizon + 2

        due = rng.randint(due_low, due_high)
        if priority == 4.0:
            due = max(due_low, due - rng.randint(1, 2))

        tasks.append(Task(task_id=d, processing_time=duration, priority=priority, due=due, release=0))
    return tasks


def make_eligibility(n_robots: int, n_tasks: int, rng: random.Random, eligible_per_task: int = 3) -> List[List[int]]:
    eligibility = [[0 for _ in range(n_tasks)] for _ in range(n_robots)]
    for d in range(n_tasks):
        robots = rng.sample(range(n_robots), k=min(eligible_per_task, n_robots))
        for j in robots:
            eligibility[j][d] = 1
    return eligibility


def make_full_robot_availability(mission_horizon: int, n_robots: int) -> List[List[int]]:
    return [[1 for _ in range(n_robots)] for _ in range(mission_horizon + 1)]


def make_feasible_load_scenario(seed: int = 123) -> ProblemData:
    rng = random.Random(seed)
    n_operators = 2
    n_robots = 6
    mission_horizon = 18
    tasks = generate_initial_tasks("feasible_load", mission_horizon, 18, rng)
    return ProblemData(
        tasks=tasks,
        n_operators=n_operators,
        n_robots=n_robots,
        horizon=5,
        mission_horizon=mission_horizon,
        eligibility=make_eligibility(n_robots, len(tasks), rng),
        base_lmax=[3.0, 2.0, 1.0],
        robot_availability=make_full_robot_availability(mission_horizon, n_robots),
    )


def make_overload_finite_shift_scenario(seed: int = 123) -> ProblemData:
    rng = random.Random(seed)
    n_operators = 2
    n_robots = 6
    mission_horizon = 11
    tasks = generate_initial_tasks("overload_finite_shift", mission_horizon, 18, rng)
    return ProblemData(
        tasks=tasks,
        n_operators=n_operators,
        n_robots=n_robots,
        horizon=5,
        mission_horizon=mission_horizon,
        eligibility=make_eligibility(n_robots, len(tasks), rng),
        base_lmax=[3.0, 2.0, 1.0],
        robot_availability=make_full_robot_availability(mission_horizon, n_robots),
    )


def make_dynamic_disturbance_scenario(seed: int = 123) -> ProblemData:
    rng = random.Random(seed)
    n_operators = 2
    n_robots = 6
    mission_horizon = 18
    tasks = generate_initial_tasks("dynamic_disturbance", mission_horizon, 18, rng)
    tasks.extend(
        [
            Task(task_id=18, processing_time=3, priority=2.0, due=13, release=6),
            Task(task_id=19, processing_time=4, priority=4.0, due=15, release=6),
            Task(task_id=20, processing_time=2, priority=1.0, due=16, release=12),
            Task(task_id=21, processing_time=3, priority=4.0, due=17, release=12),
        ]
    )

    availability = make_full_robot_availability(mission_horizon, n_robots)
    for time in range(8, 12):
        availability[time][3] = 0

    return ProblemData(
        tasks=tasks,
        n_operators=n_operators,
        n_robots=n_robots,
        horizon=5,
        mission_horizon=mission_horizon,
        eligibility=make_eligibility(n_robots, len(tasks), rng),
        base_lmax=[3.0, 2.0, 1.0],
        robot_availability=availability,
        lmax_schedule=dynamic_operator_capacity_schedule,
    )


def make_paper_scenario(name: str, seed: int = 123) -> ProblemData:
    if name == "feasible_load":
        return make_feasible_load_scenario(seed)
    if name == "overload_finite_shift":
        return make_overload_finite_shift_scenario(seed)
    if name == "dynamic_disturbance":
        return make_dynamic_disturbance_scenario(seed)
    raise ValueError("Scenario must be feasible_load, overload_finite_shift, or dynamic_disturbance.")


def run_all_methods(data: ProblemData, params: ModelParams, seed: int = 0) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for method in METHODS:
        states, _ = run_simulation(data, params, method=method, seed=seed)
        results[method] = evaluate_run(states, data, params)
    return results


def normalize_weights(a: float, b: float, c: float) -> Tuple[float, float, float]:
    total = max(a + b + c, 1e-12)
    return a / total, b / total, c / total


def params_from_optuna_trial(trial) -> ModelParams:
    omega_f_raw = trial.suggest_float("omega_f_raw", 0.05, 1.0)
    omega_s_raw = trial.suggest_float("omega_s_raw", 0.05, 1.0)
    omega_fs_raw = trial.suggest_float("omega_fs_raw", 0.05, 1.0)
    omega_f, omega_s, omega_fs = normalize_weights(omega_f_raw, omega_s_raw, omega_fs_raw)

    return ModelParams(
        alpha=trial.suggest_float("alpha", 0.2, 3.0),
        beta=trial.suggest_float("beta", 0.0, 5.0),
        gamma=trial.suggest_float("gamma", 0.0, 6.0),
        eta=trial.suggest_float("eta", 0.2, 6.0),
        rho=trial.suggest_float("rho", 0.2, 8.0),
        kappa=trial.suggest_float("kappa", 0.0, 4.0),
        delta_w=trial.suggest_float("delta_w", 0.2, 4.0),
        alpha_s=trial.suggest_float("alpha_s", 0.05, 0.70),
        beta_L=trial.suggest_float("beta_L", 0.01, 0.40),
        beta_P=trial.suggest_float("beta_P", 0.01, 0.40),
        beta_W=trial.suggest_float("beta_W", 0.01, 0.35),
        beta_R=trial.suggest_float("beta_R", 0.0, 0.25),
        zeta_L=trial.suggest_float("zeta_L", 0.001, 0.15),
        zeta_S=trial.suggest_float("zeta_S", 0.001, 0.15),
        zeta_C=trial.suggest_float("zeta_C", 0.001, 0.15),
        zeta_T=trial.suggest_float("zeta_T", 0.0, 0.15),
        zeta_D=trial.suggest_float("zeta_D", 0.0, 0.10),
        cmax=trial.suggest_int("cmax", 3, 10),
        omega_f=omega_f,
        omega_s=omega_s,
        omega_fs=omega_fs,
        sigma_z=trial.suggest_float("sigma_z", 0.05, 0.30),
        n_clusters=3,
        beam_width=trial.suggest_int("beam_width", 5, 20),
        max_actions_per_step=trial.suggest_int("max_actions_per_step", 8, 30),
    )


def tuning_score(metrics: Dict[str, float]) -> float:
    return (
        metrics["TWT"]
        + 100.0 * metrics["DMR"]
        + 150.0 * (1.0 - metrics["WCR"])
        + 0.5 * metrics["CI"]
        + 3.0 * metrics["PI"]
    )


def tune_hyperparameters_for_scenario(
    scenario_name: str,
    n_trials: int = 30,
    seeds: Sequence[int] = (101, 202, 303),
    optuna_seed: int = 123,
) -> Tuple[ModelParams, List[Dict[str, float]]]:
    if optuna is None:
        raise RuntimeError("Optuna is not installed. Install it with: python -m pip install optuna")

    trial_records: List[Dict[str, float]] = []
    sampler = optuna.samplers.TPESampler(seed=optuna_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial) -> float:
        params = params_from_optuna_trial(trial)
        scores = []
        for seed in seeds:
            data = make_paper_scenario(scenario_name, seed=seed)
            states, _ = run_simulation(data, params, method="proposed_msa_rh", seed=seed)
            scores.append(tuning_score(evaluate_run(states, data, params)))

        score = sum(scores) / len(scores)
        record = {"trial": float(trial.number), "score": score}
        record.update({key: float(value) for key, value in asdict(params).items() if isinstance(value, (int, float))})
        trial_records.append(record)
        return score

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return params_from_optuna_trial(study.best_trial), trial_records


def action_log_rows(
    scenario: str,
    method: MethodName,
    run: int,
    actions: Sequence[Action],
    data: ProblemData,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for time, action in enumerate(actions):
        for operator, robot, task_id in action:
            task = data.tasks[task_id]
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "run": run,
                    "time": time,
                    "operator": operator + 1,
                    "robot": robot + 1,
                    "task": task_id + 1,
                    "priority": task.priority,
                    "duration": task.processing_time,
                    "due": task.due,
                    "release": task.release,
                }
            )
    return rows


def state_trajectory_rows(
    scenario: str,
    method: MethodName,
    run: int,
    states: Sequence[State],
    data: ProblemData,
    params: ModelParams,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for state in states:
        loads = operator_load(state, data)
        impairments = impairment_scores(state, params)
        completed_tasks = sum(1 for value in state.task_state if value == 2)
        in_progress_tasks = sum(1 for value in state.task_state if value == 1)
        row: Dict[str, object] = {
            "scenario": scenario,
            "method": method,
            "run": run,
            "time": state.time,
            "completed_tasks": completed_tasks,
            "in_progress_tasks": in_progress_tasks,
        }
        for i in range(data.n_operators):
            row[f"operator_{i + 1}_load"] = loads[i]
            row[f"operator_{i + 1}_fatigue"] = state.fatigue[i]
            row[f"operator_{i + 1}_stress"] = state.stress[i]
            row[f"operator_{i + 1}_impairment"] = impairments[i]
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_metric_rows(rows: Sequence[Dict[str, object]], metrics: Sequence[str]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for row in rows:
        key = (str(row["scenario"]), str(row["method"]))
        grouped.setdefault(key, {metric: [] for metric in metrics})
        for metric in metrics:
            grouped[key][metric].append(float(row[metric]))

    summary_rows: List[Dict[str, object]] = []
    for (scenario, method), values_by_metric in sorted(grouped.items()):
        out: Dict[str, object] = {"scenario": scenario, "method": method}
        for metric, values in values_by_metric.items():
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(out)
    return summary_rows


def run_paper_experiment(
    output_dir: Path,
    params: Optional[ModelParams] = None,
    runs: int = 1,
    base_seed: int = 123,
) -> Dict[str, Path]:
    if params is None:
        params = ModelParams()

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["TWT", "DMR", "WCR", "CI", "PI"]
    scenarios = ["feasible_load", "overload_finite_shift", "dynamic_disturbance"]

    detailed_rows: List[Dict[str, object]] = []
    action_rows: List[Dict[str, object]] = []
    trajectory_rows: List[Dict[str, object]] = []

    for scenario_index, scenario in enumerate(scenarios):
        for run in range(runs):
            scenario_seed = base_seed + 1000 * run + 101 * scenario_index
            for method in METHODS:
                data = make_paper_scenario(scenario, seed=scenario_seed)
                states, actions = run_simulation(data, params, method=method, seed=scenario_seed)
                run_metrics = evaluate_run(states, data, params)
                detailed_row: Dict[str, object] = {
                    "scenario": scenario,
                    "method": method,
                    "run": run,
                    "seed": scenario_seed,
                }
                detailed_row.update(run_metrics)
                detailed_rows.append(detailed_row)
                action_rows.extend(action_log_rows(scenario, method, run, actions, data))
                trajectory_rows.extend(state_trajectory_rows(scenario, method, run, states, data, params))

    summary_rows = summarize_metric_rows(detailed_rows, metrics)

    files = {
        "detailed_metrics": output_dir / "detailed_run_metrics.csv",
        "summary_metrics": output_dir / "summary_metrics.csv",
        "assignment_log": output_dir / "assignment_log.csv",
        "state_trajectories": output_dir / "state_trajectories.csv",
        "hyperparameters": output_dir / "hyperparameters.json",
        "methods": output_dir / "methods.json",
    }

    write_csv(files["detailed_metrics"], detailed_rows)
    write_csv(files["summary_metrics"], summary_rows)
    write_csv(files["assignment_log"], action_rows)
    write_csv(files["state_trajectories"], trajectory_rows)
    files["hyperparameters"].write_text(json.dumps(asdict(params), indent=2), encoding="utf-8")
    files["methods"].write_text(json.dumps(list(METHODS), indent=2), encoding="utf-8")
    return files


def run_tuned_paper_experiment(
    output_dir: Path,
    tune_trials: int = 5,
    tune_seed_count: int = 2,
    final_runs: int = 3,
    base_seed: int = 123,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = ["feasible_load", "overload_finite_shift", "dynamic_disturbance"]
    metrics = ["TWT", "DMR", "WCR", "CI", "PI"]

    detailed_rows: List[Dict[str, object]] = []
    action_rows: List[Dict[str, object]] = []
    trajectory_rows: List[Dict[str, object]] = []
    tuning_summary_rows: List[Dict[str, object]] = []
    best_params_by_scenario: Dict[str, Dict[str, object]] = {}

    for scenario_index, scenario in enumerate(scenarios):
        tune_seeds = tuple(base_seed + 10_000 + 1000 * k + 101 * scenario_index for k in range(tune_seed_count))
        print(f"Tuning {scenario} with {tune_trials} trials on seeds {list(tune_seeds)}...")
        best_params, trial_records = tune_hyperparameters_for_scenario(
            scenario,
            n_trials=tune_trials,
            seeds=tune_seeds,
            optuna_seed=base_seed + scenario_index,
        )

        best_params_by_scenario[scenario] = asdict(best_params)
        trial_path = output_dir / f"tuning_trials_{scenario}.csv"
        write_csv(trial_path, trial_records)
        (output_dir / f"best_hyperparameters_{scenario}.json").write_text(
            json.dumps(asdict(best_params), indent=2),
            encoding="utf-8",
        )

        best_score = min(float(row["score"]) for row in trial_records) if trial_records else float("nan")
        tuning_summary_rows.append({"scenario": scenario, "best_score": best_score, "n_trials": tune_trials})

        for run in range(final_runs):
            scenario_seed = base_seed + 1000 * run + 101 * scenario_index
            for method in METHODS:
                data = make_paper_scenario(scenario, seed=scenario_seed)
                states, actions = run_simulation(data, best_params, method=method, seed=scenario_seed)
                run_metrics = evaluate_run(states, data, best_params)
                row: Dict[str, object] = {
                    "scenario": scenario,
                    "method": method,
                    "run": run,
                    "seed": scenario_seed,
                    "tuned_for_scenario": scenario,
                }
                row.update(run_metrics)
                detailed_rows.append(row)
                action_rows.extend(action_log_rows(scenario, method, run, actions, data))
                trajectory_rows.extend(state_trajectory_rows(scenario, method, run, states, data, best_params))

    summary_rows = summarize_metric_rows(detailed_rows, metrics)
    files = {
        "detailed_metrics": output_dir / "detailed_run_metrics.csv",
        "summary_metrics": output_dir / "summary_metrics.csv",
        "assignment_log": output_dir / "assignment_log.csv",
        "state_trajectories": output_dir / "state_trajectories.csv",
        "tuning_summary": output_dir / "tuning_summary.csv",
        "best_hyperparameters_by_scenario": output_dir / "best_hyperparameters_by_scenario.json",
        "methods": output_dir / "methods.json",
        "solver_notes": output_dir / "solver_notes.txt",
    }

    write_csv(files["detailed_metrics"], detailed_rows)
    write_csv(files["summary_metrics"], summary_rows)
    write_csv(files["assignment_log"], action_rows)
    write_csv(files["state_trajectories"], trajectory_rows)
    write_csv(files["tuning_summary"], tuning_summary_rows)
    files["best_hyperparameters_by_scenario"].write_text(
        json.dumps(best_params_by_scenario, indent=2),
        encoding="utf-8",
    )
    files["methods"].write_text(json.dumps(list(METHODS), indent=2), encoding="utf-8")
    files["solver_notes"].write_text(
        "\n".join(
            [
                "The full receding-horizon MSA-TA model is nonlinear because fatigue, stress,",
                "impairment, softmax belief, and belief-weighted capacity propagate through",
                "future binary actions. The proposed_msa_rh and human_unaware_rh methods use",
                "beam-search receding-horizon optimization over feasible candidate actions.",
                "The myopic_optimization method uses OR-Tools CP-SAT for an exact current-step",
                "binary assignment solve when OR-Tools is available, then falls back to beam",
                "search if CP-SAT is unavailable.",
            ]
        ),
        encoding="utf-8",
    )
    return files


def make_toy_problem() -> ProblemData:
    tasks = [
        Task(task_id=0, processing_time=3, priority=4, due=5),
        Task(task_id=1, processing_time=2, priority=2, due=4),
        Task(task_id=2, processing_time=4, priority=1, due=8),
    ]
    return ProblemData(
        tasks=tasks,
        n_operators=2,
        n_robots=2,
        horizon=3,
        mission_horizon=8,
        eligibility=[
            [1, 1, 0],
            [1, 0, 1],
        ],
        base_lmax=[2.0, 1.5, 1.0],
    )


def dynamic_operator_capacity_schedule(operator: int, cluster: int, time: int, data: ProblemData) -> float:
    """Example schedule for the paper's dynamic-disturbance scenario.

    The base model assumes equal capacities for all operators. In the dynamic
    scenario, operator 2 has reduced capacity from t >= 10 onward. With zero
    based indexing, operator 2 is operator == 1.
    """
    if operator == 1 and time >= 10:
        reduced = [2.0, 1.0, 1.0]
        return reduced[cluster]
    return data.base_lmax[cluster]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the clean MSA-TA paper-model experiment.")
    parser.add_argument("--output-dir", type=str, default="cleaned output1")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tune", action="store_true", help="Tune parameters per scenario before the final experiment.")
    parser.add_argument("--tune-trials", type=int, default=5)
    parser.add_argument("--tune-seed-count", type=int, default=2)
    parser.add_argument("--toy", action="store_true", help="Run only the small toy smoke test.")
    args = parser.parse_args()

    model_params = ModelParams()
    if args.toy:
        problem = make_toy_problem()
        run_states, run_actions = run_simulation(problem, model_params, method="proposed_msa_rh")
        print("Actions:")
        for t, action_at_t in enumerate(run_actions):
            print(f"t={t}: {action_at_t}")
        print("Metrics:")
        print(evaluate_run(run_states, problem, model_params))
        return

    if args.tune:
        files = run_tuned_paper_experiment(
            Path(args.output_dir),
            tune_trials=args.tune_trials,
            tune_seed_count=args.tune_seed_count,
            final_runs=args.runs,
            base_seed=args.seed,
        )
    else:
        files = run_paper_experiment(Path(args.output_dir), model_params, runs=args.runs, base_seed=args.seed)
    print("Saved clean experiment outputs:")
    for name, path in files.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()

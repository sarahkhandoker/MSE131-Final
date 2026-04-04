import random
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ============================================================
# DEFAULT PROJECT DATA
# ============================================================
DEFAULT_TASKS = [
    {"task": "A", "activity": "Market Research", "phase": "Set-Up", "predecessors": "", "avg": 2.0, "opt": 1.5, "pess": 2.3, "resource": "business", "weekly_cost": 1200},
    {"task": "B", "activity": "Product Requirements and Design", "phase": "Set-Up", "predecessors": "A", "avg": 2.0, "opt": 1.3, "pess": 2.6, "resource": "business", "weekly_cost": 1400},
    {"task": "C", "activity": "Manufacturing and Partner Set-Up", "phase": "Set-Up", "predecessors": "A", "avg": 3.0, "opt": 2.7, "pess": 3.1, "resource": "external", "weekly_cost": 1600},
    {"task": "D", "activity": "App UI/UX", "phase": "Development", "predecessors": "B", "avg": 3.0, "opt": 2.8, "pess": 3.4, "resource": "dev", "weekly_cost": 2200},
    {"task": "E", "activity": "App Backend", "phase": "Development", "predecessors": "B", "avg": 5.0, "opt": 4.7, "pess": 5.0, "resource": "dev", "weekly_cost": 2600},
    {"task": "F", "activity": "Software/Card Integration Setup", "phase": "Development", "predecessors": "C,E", "avg": 4.0, "opt": 3.8, "pess": 4.2, "resource": "dev", "weekly_cost": 2500},
    {"task": "G", "activity": "Mobile App Front End Development", "phase": "Development", "predecessors": "D,E", "avg": 4.0, "opt": 3.9, "pess": 4.1, "resource": "dev", "weekly_cost": 2400},
    {"task": "H", "activity": "System Integration and Internal Testing", "phase": "Development", "predecessors": "F,G", "avg": 3.0, "opt": 3.0, "pess": 3.3, "resource": "dev", "weekly_cost": 2300},
    {"task": "I", "activity": "Pilot Testing and Bug Fixing", "phase": "Deployment", "predecessors": "H", "avg": 3.0, "opt": 2.1, "pess": 3.2, "resource": "dev", "weekly_cost": 2200},
    {"task": "J", "activity": "Marketing and Preparing for Launch", "phase": "Deployment", "predecessors": "B", "avg": 3.0, "opt": 2.5, "pess": 3.1, "resource": "business", "weekly_cost": 1500},
    {"task": "K", "activity": "Launch!", "phase": "Deployment", "predecessors": "I,J", "avg": 1.0, "opt": 1.0, "pess": 1.2, "resource": "business", "weekly_cost": 1000},
]

CRASHABLE_TASKS = {
    "E": {"label": "App Backend", "max_crash": 1.5, "cost_per_week": 3500},
    "G": {"label": "Mobile App Front End Development", "max_crash": 1.0, "cost_per_week": 3000},
    "H": {"label": "System Integration and Internal Testing", "max_crash": 1.0, "cost_per_week": 3200},
    "I": {"label": "Pilot Testing and Bug Fixing", "max_crash": 1.0, "cost_per_week": 2800},
}


def parse_predecessors(value):
    text = str(value).strip()
    if text == "":
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def get_task_row(tasks, task_name):
    for row in tasks:
        if row["task"] == task_name:
            return row
    return None


def build_successors(tasks):
    successors = defaultdict(list)
    for row in tasks:
        task = row["task"]
        preds = parse_predecessors(row["predecessors"])
        for pred in preds:
            successors[pred].append(task)
        if task not in successors:
            successors[task] = successors[task]
    return dict(successors)


def topological_order(tasks):
    remaining = [row["task"] for row in tasks]
    ordered = []
    preds_map = {row["task"]: parse_predecessors(row["predecessors"]) for row in tasks}

    while remaining:
        progress = False
        for task in remaining[:]:
            ready = True
            for pred in preds_map[task]:
                if pred not in ordered:
                    ready = False
                    break
            if ready:
                ordered.append(task)
                remaining.remove(task)
                progress = True
        if not progress:
            raise ValueError("The task network has a cycle or invalid predecessor.")
    return ordered


def sample_task_durations(tasks):
    sampled = {}
    for row in tasks:
        low = float(row["opt"])
        mode = float(row["avg"])
        high = float(row["pess"])
        if not (low <= mode <= high):
            raise ValueError(f"Task {row['task']} must satisfy optimistic <= average <= pessimistic.")
        sampled[row["task"]] = random.triangular(low, high, mode)
    return sampled


def resource_is_free(intervals, start, duration, capacity, step=0.1):
    t = start
    end = start + duration
    while t < end:
        active = 0
        for s, e in intervals:
            if s <= t < e:
                active += 1
        if active >= capacity:
            return False
        t += step
    return True


def earliest_with_capacity(intervals, earliest_time, duration, capacity, step=0.1):
    current = earliest_time
    limit = 10000
    checks = 0
    while checks < limit:
        if resource_is_free(intervals, current, duration, capacity, step):
            return current
        current += step
        checks += 1
    return current


def compute_schedule(tasks, durations, use_capacity=False, dev_capacity=2):
    order = topological_order(tasks)
    successors = build_successors(tasks)

    es = {}
    ef = {}
    resource_intervals = defaultdict(list)

    for task in order:
        row = get_task_row(tasks, task)
        preds = parse_predecessors(row["predecessors"])

        earliest_start = 0.0
        if preds:
            earliest_start = max(ef[p] for p in preds)

        start_time = earliest_start
        if use_capacity and row["resource"] == "dev":
            start_time = earliest_with_capacity(resource_intervals["dev"], earliest_start, durations[task], dev_capacity)

        finish_time = start_time + durations[task]
        es[task] = start_time
        ef[task] = finish_time

        if use_capacity and row["resource"] == "dev":
            resource_intervals["dev"].append((start_time, finish_time))

    project_duration = max(ef.values())

    ls = {}
    lf = {}
    for task in order[::-1]:
        succs = successors.get(task, [])
        if not succs:
            lf[task] = project_duration
        else:
            lf[task] = min(ls[s] for s in succs)
        ls[task] = lf[task] - durations[task]
    
    rows = []
    for task in order:
        row = get_task_row(tasks, task)
        slack = ls[task] - es[task]
        rows.append({
            "Task": task,
            "Activity": row["activity"],
            "Phase": row["phase"],
            "Resource": row["resource"],
            "Predecessors": row["predecessors"],
            "Duration": round(durations[task], 2),
            "ES": round(es[task], 2),
            "EF": round(ef[task], 2),
            "LS": round(ls[task], 2),
            "LF": round(lf[task], 2),
            "Slack": round(slack, 2),
            "Critical?": "Yes" if abs(slack) < 1e-6 else "No",
        })

    return pd.DataFrame(rows), project_duration


def apply_crashing(tasks, durations, crash_settings):
    extra_cost = 0.0
    for task, crash_weeks in crash_settings.items():
        if task not in durations or crash_weeks <= 0:
            continue

        row = get_task_row(tasks, task)
        max_crash = CRASHABLE_TASKS[task]["max_crash"]
        allowed_crash = min(crash_weeks, max_crash)
        new_duration = max(float(row["opt"]), durations[task] - allowed_crash)
        actual_reduction = durations[task] - new_duration
        durations[task] = new_duration
        extra_cost += actual_reduction * CRASHABLE_TASKS[task]["cost_per_week"]

    return durations, extra_cost
def apply_supplier_delay(durations, probability, min_delay, max_delay):
    happened = False
    added = 0.0
    if random.random() < probability:
        happened = True
        added = random.uniform(min_delay, max_delay)
        durations["C"] += added
    return durations, happened, added


def apply_rework(durations, probability, min_rework, max_rework):
    happened = False
    added = 0.0
    if random.random() < probability:
        happened = True
        added = random.uniform(min_rework, max_rework)
        durations["I"] += added
    return durations, happened, added


def apply_overtime(tasks, durations, target_weeks, projected_finish, overtime_on, reduction_pct, overtime_cost_per_week):
    extra_cost = 0.0
    happened = False

    if not overtime_on:
        return durations, extra_cost, happened

    if projected_finish > target_weeks:
        happened = True
        for task in ["H", "I", "K"]:
            row = get_task_row(tasks, task)
            original = durations[task]
            reduced = original * (1 - reduction_pct / 100.0)
            reduced = max(float(row["opt"]), reduced)
            saved = original - reduced
            durations[task] = reduced
            extra_cost += saved * overtime_cost_per_week

    return durations, extra_cost, happened
def compute_total_cost(tasks, durations, extra_cost):
    total_cost = 0.0
    for row in tasks:
        total_cost += durations[row["task"]] * float(row["weekly_cost"])
    total_cost += extra_cost
    return total_cost


def run_monte_carlo(
    tasks,
    iterations,
    target_weeks,
    crash_settings,
    rework_probability,
    rework_min,
    rework_max,
    capacity_on,
    dev_capacity,
    supplier_delay_probability,
    supplier_delay_min,
    supplier_delay_max,
    overtime_on,
    overtime_reduction_pct,
    overtime_cost_per_week,
    seed,
):
    random.seed(seed)

    completion_times = []
    total_costs = []
    on_time_count = 0
    critical_counts = defaultdict(int)

    rework_count = 0
    supplier_delay_count = 0
    overtime_count = 0

    for _ in range(iterations):
        durations = sample_task_durations(tasks)
        durations, crash_extra_cost = apply_crashing(tasks, durations, crash_settings)

        durations, supplier_happened, _ = apply_supplier_delay(
            durations,
            supplier_delay_probability,
            supplier_delay_min,
            supplier_delay_max,
        )
        if supplier_happened:
            supplier_delay_count += 1

        durations, rework_happened, _ = apply_rework(
            durations,
            rework_probability,
            rework_min,
            rework_max,
        )
        if rework_happened:
            rework_count += 1

        schedule_df, projected_finish = compute_schedule(
            tasks,
            durations,
            use_capacity=capacity_on,
            dev_capacity=dev_capacity,
        )

        durations, overtime_extra_cost, overtime_happened = apply_overtime(
            tasks,
            durations,
            target_weeks,
            projected_finish,
            overtime_on,
                        overtime_reduction_pct,
            overtime_cost_per_week,
        )
        if overtime_happened:
            overtime_count += 1

        schedule_df, final_finish = compute_schedule(
            tasks,
            durations,
            use_capacity=capacity_on,
            dev_capacity=dev_capacity,
        )

        total_extra_cost = crash_extra_cost + overtime_extra_cost
        total_cost = compute_total_cost(tasks, durations, total_extra_cost)

        completion_times.append(final_finish)
        total_costs.append(total_cost)

        if final_finish <= target_weeks:
            on_time_count += 1

        for _, row in schedule_df.iterrows():
            if row["Critical?"] == "Yes":
                critical_counts[row["Task"]] += 1

    critical_rows = []
    for row in tasks:
        task = row["task"]
        critical_rows.append({
            "Task": task,
            "Critical Frequency": critical_counts[task],
            "Critical Probability": critical_counts[task] / iterations,
        })

    return {
        "avg_completion": sum(completion_times) / len(completion_times),
        "min_completion": min(completion_times),
        "max_completion": max(completion_times),
         "probability_on_time": on_time_count / iterations,
        "avg_total_cost": sum(total_costs) / len(total_costs),
        "completion_times": completion_times,
        "critical_df": pd.DataFrame(critical_rows).sort_values("Critical Probability", ascending=False),
        "rework_rate": rework_count / iterations,
        "supplier_delay_rate": supplier_delay_count / iterations,
        "overtime_rate": overtime_count / iterations,
    }


def make_histogram(data, target_weeks):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(data, bins=20)
    ax.axvline(target_weeks, linestyle="--")
    ax.set_title("Simulated Project Completion Times")
    ax.set_xlabel("Completion time (weeks)")
    ax.set_ylabel("Frequency")
    return fig


def make_gantt(schedule_df):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    y_positions = list(range(len(schedule_df)))
    labels = []

    for i, (_, row) in enumerate(schedule_df.iterrows()):
        ax.barh(i, row["Duration"], left=row["ES"])
        labels.append(f"{row['Task']} - {row['Activity']}")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Weeks")
    ax.set_title("Baseline Schedule")
    return fig

st.set_page_config(page_title="Project Timeline Simulation", layout="wide")

st.title("📊 Startup App + Card Launch Simulation")
st.markdown(
    """
This model uses **project management principles** to compute the timeline:

- **Precedence relationships**
- **Forward pass / backward pass**
- **Critical path**
- **Slack**
- **Uncertain task durations**
- **Monte Carlo simulation**
"""
)

st.subheader("1) Task Data")
default_df = pd.DataFrame(DEFAULT_TASKS)
edited_df = st.data_editor(default_df, use_container_width=True, num_rows="dynamic")
tasks = edited_df.to_dict(orient="records")

st.subheader("2) Core Inputs")
col1, col2, col3 = st.columns(3)
target_weeks = col1.number_input("Target completion time (weeks)", min_value=1.0, value=20.0, step=0.5)
iterations = col2.slider("Number of simulation runs", min_value=200, max_value=10000, value=3000, step=200)
seed = col3.number_input("Random seed", min_value=0, value=42, step=1)

st.subheader("3) Five Extensions")

with st.expander("Extension 1: Crashing selected tasks", expanded=True):
    st.write("Reduce durations of selected tasks, but pay extra cost.")
    crash_settings = {}
    for task, info in CRASHABLE_TASKS.items():
        crash_settings[task] = st.slider(
            f"{task} - {info['label']} crash amount (weeks)",
            min_value=0.0,
            max_value=float(info["max_crash"]),
            value=0.0,
            step=0.1,
        )

with st.expander("Extension 2: Rework after pilot testing", expanded=False):
    rework_probability = st.slider("Probability of major rework after pilot testing", 0.0, 1.0, 0.25, 0.05)
    rework_min = st.number_input("Minimum rework added to task I (weeks)", min_value=0.0, value=0.5, step=0.1)
    rework_max = st.number_input("Maximum rework added to task I (weeks)", min_value=0.0, value=2.0, step=0.1)

with st.expander("Extension 3: Limited team capacity", expanded=False):
    capacity_on = st.checkbox("Turn on startup team capacity constraint", value=True)
    dev_capacity = st.slider("How many development tasks can run at once?", 1, 4, 2)

with st.expander("Extension 4: External supplier / approval delay", expanded=False):
    supplier_delay_probability = st.slider("Probability of external delay on task C", 0.0, 1.0, 0.30, 0.05)
    supplier_delay_min = st.number_input("Minimum supplier delay (weeks)", min_value=0.0, value=0.5, step=0.1)
    supplier_delay_max = st.number_input("Maximum supplier delay (weeks)", min_value=0.0, value=2.0, step=0.1)

with st.expander("Extension 5: Overtime near the deadline", expanded=False):
    overtime_on = st.checkbox("Allow overtime if the project is projected to miss the deadline", value=True)
    overtime_reduction_pct = st.slider("Late-stage duration reduction under overtime (%)", 0, 40, 15, 1)
    overtime_cost_per_week = st.number_input("Overtime premium cost per week saved ($)", min_value=0.0, value=3000.0, step=500.0)

baseline_durations = {row["task"]: float(row["avg"]) for row in tasks}
baseline_durations, baseline_crash_cost = apply_crashing(tasks, baseline_durations, crash_settings)

baseline_schedule, baseline_finish = compute_schedule(
    tasks,
    baseline_durations,
    use_capacity=capacity_on,
    dev_capacity=dev_capacity,
)
baseline_cost = compute_total_cost(tasks, baseline_durations, baseline_crash_cost)

st.subheader("4) Baseline CPM Results")
c1, c2, c3 = st.columns(3)
c1.metric("Baseline completion time", f"{baseline_finish:.2f} weeks")
c2.metric("Target launch time", f"{target_weeks:.2f} weeks")
c3.metric("Baseline total cost", f"${baseline_cost:,.0f}")

st.dataframe(baseline_schedule, use_container_width=True)
st.pyplot(make_gantt(baseline_schedule))

critical_tasks = baseline_schedule[baseline_schedule["Critical?"] == "Yes"]["Task"].tolist()
st.markdown(f"**Baseline critical path tasks:** {', '.join(critical_tasks)}")

st.subheader("5) Simulation Results")
results = run_monte_carlo(
    tasks=tasks,
    iterations=iterations,
    target_weeks=target_weeks,
    crash_settings=crash_settings,
    rework_probability=rework_probability,
    rework_min=rework_min,
    rework_max=rework_max,
    capacity_on=capacity_on,
    dev_capacity=dev_capacity,
    supplier_delay_probability=supplier_delay_probability,
    supplier_delay_min=supplier_delay_min,
    supplier_delay_max=supplier_delay_max,
    overtime_on=overtime_on,
    overtime_reduction_pct=overtime_reduction_pct,
    overtime_cost_per_week=overtime_cost_per_week,
    seed=seed,
)

r1, r2, r3, r4 = st.columns(4)
r1.metric("Average simulated completion time", f"{results['avg_completion']:.2f} weeks")
r2.metric("Probability of finishing on time", f"{results['probability_on_time'] * 100:.1f}%")
r3.metric("Average total cost", f"${results['avg_total_cost']:,.0f}")
r4.metric("Simulated range", f"{results['min_completion']:.2f} - {results['max_completion']:.2f} weeks")

st.pyplot(make_histogram(results["completion_times"], target_weeks))

st.markdown("### Extension event rates")
e1, e2, e3 = st.columns(3)
e1.metric("Rework triggered", f"{results['rework_rate'] * 100:.1f}% of runs")
e2.metric("Supplier delay triggered", f"{results['supplier_delay_rate'] * 100:.1f}% of runs")
e3.metric("Overtime activated", f"{results['overtime_rate'] * 100:.1f}% of runs")

st.markdown("### Critical-task frequencies")
critical_display = results["critical_df"].copy()
critical_display["Critical Probability"] = (critical_display["Critical Probability"] * 100).round(1).astype(str) + "%"
st.dataframe(critical_display, use_container_width=True)

st.subheader("6) What Each Extension Changes")
st.markdown(
    """
**1. Crashing selected tasks**  
Selected tasks can be shortened by paying an added crash cost. This reduces completion time but increases total project cost.

**2. Rework after pilot testing**  
Pilot testing can uncover major issues, which adds extra rework time to the pilot testing task. This increases both project duration and variability.

**3. Limited team capacity**  
Development tasks use a shared startup team. If too many development tasks try to run at the same time, some must wait. This creates internal waiting and can increase completion time.

**4. External supplier or approval delay**  
The manufacturing / partner setup task can receive a random external delay. This captures the risk of outside partners slowing down the project.

**5. Overtime near the deadline**  
If the project is projected to miss the target launch date, overtime can shorten some late-stage tasks. This improves the chance of finishing on time, but increases cost.
"""
)










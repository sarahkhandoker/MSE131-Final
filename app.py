import random
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT DATA
# ============================================================
DEFAULT_TASKS = [
    {"task": "A", "activity": "Market Research", "phase": "Set-Up",      "predecessors": "",    "avg": 2.0, "opt": 1.5, "pess": 2.3},
    {"task": "B", "activity": "Product Requirements and Design", "phase": "Set-Up",      "predecessors": "A",   "avg": 2.0, "opt": 1.3, "pess": 2.6},
    {"task": "C", "activity": "Manufacturing and Partner Set-Up", "phase": "Set-Up",      "predecessors": "A",   "avg": 3.0, "opt": 2.7, "pess": 3.1},
    {"task": "D", "activity": "App UI/UX", "phase": "Development", "predecessors": "B",   "avg": 3.0, "opt": 2.8, "pess": 3.4},
    {"task": "E", "activity": "App Backend", "phase": "Development", "predecessors": "B",   "avg": 5.0, "opt": 4.7, "pess": 5.0},
    {"task": "F", "activity": "Software/Card Integration Setup", "phase": "Development", "predecessors": "C,E", "avg": 4.0, "opt": 3.8, "pess": 4.2},
    {"task": "G", "activity": "Mobile App Front End Development", "phase": "Development", "predecessors": "D,E", "avg": 4.0, "opt": 3.9, "pess": 4.1},
    {"task": "H", "activity": "System Integration and Internal Testing", "phase": "Development", "predecessors": "F,G", "avg": 3.0, "opt": 3.0, "pess": 3.3},
    {"task": "I", "activity": "Pilot Testing and Bug Fixing", "phase": "Deployment", "predecessors": "H",   "avg": 3.0, "opt": 2.1, "pess": 3.2},
    {"task": "J", "activity": "Marketing and Preparing for Launch", "phase": "Deployment", "predecessors": "B",   "avg": 3.0, "opt": 2.5, "pess": 3.1},
    {"task": "K", "activity": "Launch!", "phase": "Deployment", "predecessors": "I,J", "avg": 1.0, "opt": 1.0, "pess": 1.2},
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def parse_predecessors(value):
    if value is None:
        return []
    text = str(value).strip()
    if text == "":
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


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
    # Simple loop-based topological sort
    remaining = [row["task"] for row in tasks]
    done = []
    preds_map = {row["task"]: parse_predecessors(row["predecessors"]) for row in tasks}

    while remaining:
        progress = False
        for task in remaining[:]:
            preds = preds_map[task]
            ready = True
            for p in preds:
                if p not in done:
                    ready = False
                    break
            if ready:
                done.append(task)
                remaining.remove(task)
                progress = True
        if not progress:
            raise ValueError("The precedence network has a cycle or invalid predecessor.")
    return done


def get_task_row(tasks, task_name):
    for row in tasks:
        if row["task"] == task_name:
            return row
    return None


def compute_schedule(tasks, duration_key="avg", sampled_durations=None):
    order = topological_order(tasks)
    successors = build_successors(tasks)

    durations = {}
    for row in tasks:
        task = row["task"]
        if sampled_durations is not None:
            durations[task] = float(sampled_durations[task])
        else:
            durations[task] = float(row[duration_key])

    # Forward pass
    ES = {}
    EF = {}
    for task in order:
        row = get_task_row(tasks, task)
        preds = parse_predecessors(row["predecessors"])

        if len(preds) == 0:
            ES[task] = 0.0
        else:
            max_ef = 0.0
            for p in preds:
                if EF[p] > max_ef:
                    max_ef = EF[p]
            ES[task] = max_ef
        EF[task] = ES[task] + durations[task]

    project_duration = max(EF.values())

    # Backward pass
    LF = {}
    LS = {}
    reversed_order = order[::-1]

    for task in reversed_order:
        succs = successors.get(task, [])
        if len(succs) == 0:
            LF[task] = project_duration
        else:
            min_ls = None
            for s in succs:
                if min_ls is None or LS[s] < min_ls:
                    min_ls = LS[s]
            LF[task] = min_ls
        LS[task] = LF[task] - durations[task]

    slack = {}
    critical_flag = {}
    for task in order:
        slack[task] = LS[task] - ES[task]
        critical_flag[task] = abs(slack[task]) < 1e-9

    rows = []
    for task in order:
        row = get_task_row(tasks, task)
        rows.append({
            "Task": task,
            "Activity": row["activity"],
            "Phase": row["phase"],
            "Predecessors": row["predecessors"],
            "Duration": round(durations[task], 3),
            "ES": round(ES[task], 3),
            "EF": round(EF[task], 3),
            "LS": round(LS[task], 3),
            "LF": round(LF[task], 3),
            "Slack": round(slack[task], 3),
            "Critical?": "Yes" if critical_flag[task] else "No",
        })

    return pd.DataFrame(rows), project_duration


def sample_task_durations(tasks):
    sampled = {}
    for row in tasks:
        # triangular(low, high, mode)
        low = float(row["opt"])
        mode = float(row["avg"])
        high = float(row["pess"])

        if not (low <= mode <= high):
            raise ValueError(f"Task {row['task']} must satisfy optimistic <= average <= pessimistic.")

        sampled[row["task"]] = random.triangular(low, high, mode)
    return sampled


def run_monte_carlo(tasks, iterations, target_weeks):
    completion_times = []
    on_time_count = 0
    critical_counts = defaultdict(int)

    for _ in range(iterations):
        sampled = sample_task_durations(tasks)
        schedule_df, duration = compute_schedule(tasks, sampled_durations=sampled)
        completion_times.append(duration)

        if duration <= target_weeks:
            on_time_count += 1

        for _, row in schedule_df.iterrows():
            if row["Critical?"] == "Yes":
                critical_counts[row["Task"]] += 1

    avg_completion = sum(completion_times) / len(completion_times)
    min_completion = min(completion_times)
    max_completion = max(completion_times)
    probability_on_time = on_time_count / iterations

    critical_rows = []
    for row in tasks:
        task = row["task"]
        critical_rows.append({
            "Task": task,
            "Critical Frequency": critical_counts[task],
            "Critical Probability": critical_counts[task] / iterations,
        })

    critical_df = pd.DataFrame(critical_rows).sort_values("Critical Probability", ascending=False)

    return {
        "completion_times": completion_times,
        "avg_completion": avg_completion,
        "min_completion": min_completion,
        "max_completion": max_completion,
        "probability_on_time": probability_on_time,
        "critical_df": critical_df,
    }


def make_histogram(data, target_weeks):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(data, bins=20)
    ax.axvline(target_weeks, linestyle="--")
    ax.set_title("Simulated Project Completion Time")
    ax.set_xlabel("Completion time (weeks)")
    ax.set_ylabel("Frequency")
    return fig


def make_gantt(schedule_df):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    y_positions = list(range(len(schedule_df)))
    labels = []

    for i, (_, row) in enumerate(schedule_df.iterrows()):
        start = row["ES"]
        duration = row["Duration"]
        ax.barh(i, duration, left=start)
        labels.append(f"{row['Task']} - {row['Activity']}")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Weeks")
    ax.set_title("Baseline Schedule Gantt Chart")
    return fig


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Project Timeline Simulation", layout="wide")

st.title("📊 Project Management Simulation: App + Card Launch")
st.markdown(
    """
Launch of the App/Card service using Product Management Principles**:

- **Precedence relationships** between tasks
- **Critical Path Method (CPM)** using forward pass and backward pass
- **Slack** for non-critical tasks
- **Uncertainty** in task times using a **triangular distribution**
- **Monte Carlo simulation** to estimate completion time risk
"""
)

st.subheader("1) Edit Task Inputs")

default_df = pd.DataFrame(DEFAULT_TASKS)
edited_df = st.data_editor(
    default_df,
    use_container_width=True,
    num_rows="dynamic",
)

target_weeks = st.number_input("Target completion time (weeks)", min_value=1.0, value=20.0, step=0.5)
iterations = st.slider("Number of simulation runs", min_value=100, max_value=10000, value=3000, step=100)

tasks = edited_df.to_dict(orient="records")

st.subheader("2) Baseline CPM Model")

try:
    baseline_df, baseline_duration = compute_schedule(tasks, duration_key="avg")

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline completion time", f"{baseline_duration:.2f} weeks")
    col2.metric("Target launch time", f"{target_weeks:.2f} weeks")
    col3.metric("Meets target?", "Yes" if baseline_duration <= target_weeks else "No")

    st.dataframe(baseline_df, use_container_width=True)

    gantt_fig = make_gantt(baseline_df)
    st.pyplot(gantt_fig)

    critical_tasks = baseline_df[baseline_df["Critical?"] == "Yes"]["Task"].tolist()
    st.markdown(f"**Critical path tasks in the baseline model:** {', '.join(critical_tasks)}")

except Exception as e:
    st.error(f"Baseline model error: {e}")
    st.stop()

st.subheader("3) Monte Carlo Simulation")

if st.button("Run Simulation"):
    try:
        results = run_monte_carlo(tasks, iterations, target_weeks)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average simulated completion time", f"{results['avg_completion']:.2f} weeks")
        c2.metric("Fastest simulated time", f"{results['min_completion']:.2f} weeks")
        c3.metric("Slowest simulated time", f"{results['max_completion']:.2f} weeks")
        c4.metric("Probability of finishing by target", f"{results['probability_on_time'] * 100:.1f}%")

        hist_fig = make_histogram(results["completion_times"], target_weeks)
        st.pyplot(hist_fig)

        st.markdown("### Critical-task frequencies")
        critical_display = results["critical_df"].copy()
        critical_display["Critical Probability"] = (critical_display["Critical Probability"] * 100).round(1).astype(str) + "%"
        st.dataframe(critical_display, use_container_width=True)

        st.markdown(
            """
### How this connects to your course
- **Completion time** is your first performance measure.
- **Probability of finishing by the launch date** is your second performance measure.
- **Critical path** shows the chain of tasks that directly controls total duration.
- **Slack** shows which tasks have flexibility before they delay the project.
- The simulation captures **uncertainty** by sampling task durations each run.
"""
        )

    except Exception as e:
        st.error(f"Simulation error: {e}")

st.subheader("4) Suggested Future Extensions")
st.markdown(
    """
Simple features you can add later:
- **Crashing:** shorten task times by paying extra cost
- **Cost-time tradeoff:** compare lower completion time vs higher budget
- **Resource limits:** restrict how many tasks can happen at once
- **Scenario analysis:** best case, expected case, worst case
- **Delay events:** supplier issues, testing failures, redesign loops
"""
)

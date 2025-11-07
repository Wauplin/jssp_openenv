import matplotlib.pyplot as plt

from .models import ScheduledEvent


def gantt_chart(scheduled_events: list[ScheduledEvent], title: str, makespan: int, save_to: str) -> None:
    """Generate and save a Gantt chart from schedule events using matplotlib."""
    if not scheduled_events:
        print("No schedule events to save.")
        return

    # Extract unique machines and jobs
    machines = sorted(set(event.machine_id for event in scheduled_events))
    jobs = sorted(set(event.job_id for event in scheduled_events))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, max(6, len(machines) * 0.8)))

    # Color map for different jobs
    colors = plt.cm.tab20(range(len(jobs)))
    job_color_map = {job_id: colors[i % len(colors)] for i, job_id in enumerate(jobs)}

    # Track which jobs have been added to legend
    legend_added = set()

    # Plot each schedule event as a horizontal bar
    for event in scheduled_events:
        machine_idx = machines.index(event.machine_id)
        duration = event.end_time - event.start_time

        # Only add label for legend if this job hasn't been added yet
        label = f"Job {event.job_id}" if event.job_id not in legend_added else ""
        if label:
            legend_added.add(event.job_id)

        ax.barh(
            machine_idx,
            duration,
            left=event.start_time,
            height=0.6,
            color=job_color_map[event.job_id],
            edgecolor="black",
            linewidth=0.5,
            label=label,
        )

        # Add job label in the middle of the bar
        mid_time = event.start_time + duration / 2
        ax.text(
            mid_time,
            machine_idx,
            f"J{event.job_id}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white" if sum(job_color_map[event.job_id][:3]) < 1.5 else "black",
        )

    # Customize the chart
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels([f"Machine {m}" for m in machines])
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Machine", fontsize=12)
    ax.set_title(f"{title} (Makespan: {makespan})", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")

    # Set x-axis limits with some padding
    max_time = max(event.end_time for event in scheduled_events) if scheduled_events else 0
    ax.set_xlim(0, max_time * 1.05)

    # Add legend
    ax.legend(loc="upper right", title="Jobs")

    plt.tight_layout()
    plt.savefig(save_to)
    plt.close(fig)

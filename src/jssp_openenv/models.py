"""
Data models for the JSSP Environment.
"""

from dataclasses import dataclass
from typing import Optional

from openenv_core import Action, Observation

JobT = list[tuple[int, int]]  # (machine_index, processing_time)


@dataclass(kw_only=True)
class JSSPAction(Action):
    """Action for the JSSP environment."""

    job_ids: list[int]

    def __post_init__(self):
        if isinstance(self.job_ids, str):
            # For web app
            self.job_ids = parse_job_ids(self.job_ids)


@dataclass(kw_only=True)
class MachineObservation:
    """Observation of a single machine in the JSSP environment."""

    machine_id: int
    busy_until: Optional[int]
    current_job_id: Optional[int]


@dataclass
class ReadyOperationObservation:
    job_id: int
    machine_id: int
    duration: int
    remaining_ops: int


@dataclass(kw_only=True)
class JSSPObservation(Observation):
    """Observation from the JSSP environment - the echoed message."""

    episode_id: str

    step_count: int
    machines: list[MachineObservation]
    ready_operations: list[ReadyOperationObservation]
    completed_jobs: int
    total_jobs: int


def parse_job_ids(job_ids: str) -> list[int]:
    """Parse job_ids from string (error out if cannot be parsed)."""
    try:
        return [int(job_id) for job_id in job_ids.split(",") if job_id.strip()]
    except ValueError:
        raise ValueError(f"Invalid job_ids: {job_ids}")


@dataclass
class ScheduledEvent:
    """Represents a scheduled operation on a machine.

    Used for plotting the schedule.
    Not used for the environment / policy / solver.
    """

    job_id: int
    machine_id: int
    start_time: int
    end_time: int

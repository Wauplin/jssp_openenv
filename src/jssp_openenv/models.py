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
class JobObservation:
    """Observation of a given Job in the JSSP environment."""

    job_id: int
    operations: JobT  # remaining operations to be scheduled
    busy_until: Optional[int]  # time until the current operation is complete (or none if available)


@dataclass(kw_only=True)
class JSSPObservation(Observation):
    """Observation from the JSSP environment - the echoed message."""

    episode_id: str

    step_count: int
    machines: list[MachineObservation]
    jobs: list[JobObservation]

    def available_machines(self) -> list[MachineObservation]:
        """Get available machines from observation."""
        return [m for m in self.machines if m.busy_until is None or m.busy_until <= self.step_count]

    def available_jobs(self) -> list[JobObservation]:
        """Get available jobs from observation."""
        available_machine_ids = [m.machine_id for m in self.available_machines()]
        return [
            job
            for job in self.jobs
            if (job.busy_until is None or job.busy_until <= self.step_count)
            and len(job.operations) > 0
            and job.operations[0][0] in available_machine_ids
        ]


def parse_job_ids(job_ids: str) -> list[int]:
    """Parse job_ids from string (error out if cannot be parsed)."""
    try:
        return [int(job_id) for job_id in job_ids.split(",") if job_id.strip()]
    except ValueError as e:
        raise ValueError(f"Invalid job_ids: {job_ids}") from e


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

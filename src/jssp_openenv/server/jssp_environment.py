import uuid
from copy import deepcopy
from typing import Optional

import simpy
from openenv_core.env_server import Environment

from ..models import JobObservation, JobT, JSSPAction, JSSPObservation, MachineObservation

# Example of JSSP initial jobs
# Each tuple is a (machine_index, processing_time)
#
# FT06: list[JobT] = [
#     [(2, 1), (0, 3), (1, 6), (3, 7), (5, 3), (4, 6)],
#     [(1, 8), (2, 5), (4, 10), (5, 10), (0, 10), (3, 4)],
#     [(2, 5), (3, 4), (5, 8), (0, 9), (1, 1), (4, 7)],
#     [(1, 5), (0, 5), (2, 5), (3, 3), (4, 8), (5, 9)],
#     [(2, 9), (1, 3), (4, 5), (5, 4), (0, 3), (3, 1)],
#     [(1, 3), (3, 3), (5, 9), (0, 10), (4, 4), (2, 1)],
# ]

PENALTY = 100


class JSSPEnvironment(Environment):
    def __init__(self, jobs: list[JobT]):
        super().__init__()
        self.init_jobs = jobs
        self.reset()

    def reset(self) -> JSSPObservation:
        """Reset the environment to initial state."""
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.jobs = deepcopy(self.init_jobs)
        self.nb_machines = max(max(machine for machine, _ in job) for job in self.jobs) + 1

        # SimPy environment for time tracking
        self.env = simpy.Environment()

        # Track which operation index each job is currently on
        self.job_progress = [0] * len(self.jobs)

        # Track machine states
        self.machine_busy_until: list[Optional[int]] = [None] * self.nb_machines
        self.machine_current_job: list[Optional[int]] = [None] * self.nb_machines

        # Track completed jobs
        self.completed_jobs = 0

        return self.state

    def _get_jobs(self) -> list[JobObservation]:
        """Get all jobs with their status and remaining operations."""
        jobs: list[JobObservation] = []
        for job_id in range(len(self.jobs)):
            # @dataclass
            # class JobObservation:
            #     """Observation of a given Job in the JSSP environment."""

            #     job_id: int
            #     operations: JobT  # remaining operations to be scheduled (not counting the current one)
            #     busy_until: Optional[int]  # time until the current operation is complete (or none if available)
            job_operations = self.jobs[job_id]
            job_progress = self.job_progress[job_id]
            job_remaining_operations = job_operations[job_progress:]

            job_busy_until = None
            for current_job, busy_until in zip(self.machine_current_job, self.machine_busy_until):
                if current_job is not None and current_job == job_id:
                    job_busy_until = busy_until

            jobs.append(JobObservation(job_id=job_id, operations=job_remaining_operations, busy_until=job_busy_until))

        return jobs

    def _at_decision_step(self) -> bool:
        """Check if we're at a decision step (at least one job can be scheduled)."""
        return len(self.state.available_jobs()) > 0

    def _validate_action(self, action: JSSPAction) -> bool:
        """Validate that an action is legal."""
        scheduled_machines = set()

        for job_id in action.job_ids:
            # Check job ID is valid
            if job_id < 0 or job_id >= len(self.jobs):
                return False

            # Check job is not already complete
            if self.job_progress[job_id] >= len(self.jobs[job_id]):
                return False

            # Get the machine needed for this job's next operation
            machine_id, _ = self.jobs[job_id][self.job_progress[job_id]]

            # Check machine is available now
            busy_until = self.machine_busy_until[machine_id]
            if busy_until is not None and busy_until > self.env.now:
                return False

            # Check we're not scheduling two jobs on the same machine
            if machine_id in scheduled_machines:
                return False

            scheduled_machines.add(machine_id)

        return True

    def _schedule_jobs(self, job_ids: list[int]):
        """Schedule the given jobs on their respective machines."""
        for job_id in job_ids:
            machine_id, duration = self.jobs[job_id][self.job_progress[job_id]]

            # Update machine state
            self.machine_busy_until[machine_id] = int(self.env.now) + duration
            self.machine_current_job[machine_id] = job_id

    def _advance_to_decision_step(self):
        """Advance simulation time until the next decision step."""
        while True:
            # Stop if we're at a decision step
            if self._at_decision_step():
                break

            # Stop if all jobs are complete
            if self.completed_jobs >= len(self.jobs):
                break

            # Find the next time when a machine becomes free
            future_times = [t for t in self.machine_busy_until if t is not None and t > self.env.now]

            if not future_times:
                # No machines will become free, but not all jobs complete
                # This shouldn't happen in a valid problem
                break

            next_time = min(future_times)

            # Advance time to when the next machine becomes free
            self.env.run(until=next_time)

            # Process completed operations and clear machine state
            for i in range(self.nb_machines):
                if self.machine_busy_until[i] is not None and self.machine_busy_until[i] <= self.env.now:
                    # Machine finished processing - advance the job's progress
                    job_id = self.machine_current_job[i]
                    if job_id is not None:
                        self.job_progress[job_id] += 1

                        # Check if job is now complete
                        if self.job_progress[job_id] >= len(self.jobs[job_id]):
                            self.completed_jobs += 1

                    # Clear machine state
                    self.machine_busy_until[i] = None
                    self.machine_current_job[i] = None

    def step(self, action: JSSPAction) -> JSSPObservation:
        """Process an action and advance simulation until next decision step.

        Returns observation with reward = -(elapsed time) for valid actions,
        or reward = -PENALTY for invalid actions (without updating state).
        """
        start_time = self.env.now

        # Validate action
        if not self._validate_action(action):
            # Invalid action - return current state with penalty
            obs = self.state
            obs.reward = -PENALTY
            return obs

        # Schedule the jobs
        self._schedule_jobs(action.job_ids)

        # Advance simulation to next decision step
        self._advance_to_decision_step()

        # Calculate reward as negative time elapsed
        time_elapsed = self.env.now - start_time
        reward = -time_elapsed

        # Increment step counter
        self.step_count = int(self.env.now)

        # Return observation with reward
        obs = self.state
        obs.reward = reward

        return obs

    @property
    def state(self) -> JSSPObservation:
        """Get the current state of the environment, without the reward."""
        machines = [
            MachineObservation(
                machine_id=i,
                busy_until=self.machine_busy_until[i],
                current_job_id=self.machine_current_job[i],
            )
            for i in range(self.nb_machines)
        ]

        jobs = self._get_jobs()

        return JSSPObservation(
            done=self.completed_jobs >= len(self.jobs),
            episode_id=self.episode_id,
            step_count=self.step_count,
            machines=machines,
            jobs=jobs,
            reward=0.0,  # Default, overwritten in step()
        )

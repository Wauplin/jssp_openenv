import re
from abc import ABC, abstractmethod

from openai import OpenAI

from .models import JobObservation, JSSPAction, JSSPObservation, MachineObservation


class JSSPEnvPolicy(ABC):
    @abstractmethod
    def act(self, observation: JSSPObservation) -> JSSPAction:
        """Act based on the observation."""


class JSSPFifoPolicy(JSSPEnvPolicy):
    def act(self, observation: JSSPObservation) -> JSSPAction:
        """
        FIFO scheduling: schedule available jobs in order of job_id.

        This policy schedules jobs in FIFO order (by job_id), respecting machine availability.
        It only schedules jobs for machines that are currently available (not busy).
        """
        # Filter to only available jobs with available machines + sort by job_id
        sorted_jobs = sorted(observation.available_jobs(), key=lambda job: job.job_id)

        # Track which machines we've already scheduled to avoid conflicts
        scheduled_machines = set()
        scheduled_job_ids = []

        # Schedule jobs in FIFO order, but skip if machine is already taken
        for job in sorted_jobs:
            machine_id = job.operations[0][0]
            if machine_id not in scheduled_machines:
                scheduled_job_ids.append(job.job_id)
                scheduled_machines.add(machine_id)

        return JSSPAction(job_ids=scheduled_job_ids)


class JSSPMaxMinPolicy(JSSPEnvPolicy):
    def act(self, observation: JSSPObservation) -> JSSPAction:
        """
        Max-Min scheduling: schedule the operation with the longest duration first.
        """
        # Sort available jobs by duration (max-min)
        sorted_jobs = sorted(observation.available_jobs(), key=lambda job: job.operations[0][1], reverse=True)

        # Track which machines we've already scheduled to avoid conflicts
        scheduled_machines = set()
        scheduled_job_ids = []

        # Schedule jobs in max-min order, but skip if machine is already taken
        for job in sorted_jobs:
            machine_id = job.operations[0][0]
            if machine_id not in scheduled_machines:
                scheduled_job_ids.append(job.job_id)
                scheduled_machines.add(machine_id)

        return JSSPAction(job_ids=scheduled_job_ids)


PROMPT_TEMPLATE = """
You are solving a Job Shop Scheduling Problem (JSSP). Your goal is to minimize the total completion time (makespan) by efficiently scheduling job operations across machines.

You must optimize for minimal makespan while respecting all constraints. Each job consists of multiple operations that must be completed in sequence, and each operation requires a specific machine for a given duration.

**Current step:** {step_count}

**Machine Status:**
{machines_status}

You must check machine availability before scheduling. Machines that are busy cannot start new operations until they finish their current task.

**Jobs:**
{jobs_list}

**Rules You Must Follow:**
1. You must schedule only **available** jobs. Do not attempt to schedule jobs that are not available.
2. Each machine can run **one job at a time**. You cannot schedule multiple jobs on the same machine simultaneously.
3. You must not schedule jobs on **busy** machines (`busy_until > current step`). Check machine availability before scheduling.
4. You may **schedule multiple** jobs on different machines in the same step, or you may choose to wait if no good scheduling opportunity exists.

**Legal actions:**
{legal_job_ids}

These are the valid job IDs you can schedule at this step. You must choose a subset from this list or choose to wait.

**Action format:**
- To schedule jobs: `"0,2"` or `"1"` (comma-separated job IDs)
- To wait: `""` (empty string)
Select the best subset of jobs to schedule to minimize the total makespan once all jobs are completed.

Response only with the action format specified above, and nothing else.
"""


class JSSPLLMPolicy(JSSPEnvPolicy):
    """LLM-based scheduling policy using OpenAI-compatible API."""

    # Job Shop Scheduling prompt template

    def __init__(self, client: OpenAI, model_id: str):
        """
        Initialize the LLM policy.

        Args:
            client: OpenAI-compatible client instance
            model_id: Name of the model to use
        """
        self.client = client
        self.model_id = model_id

    def act(self, observation: JSSPObservation) -> JSSPAction:
        """
        LLM scheduling: use an LLM to schedule the operations.

        Process:
        - Determine legal job IDs (available jobs on available machines)
        - Format a prompt
        - Call the LLM
        - Parse the response to return a scheduling action
        """
        available_jobs = observation.available_jobs()

        # If no legal actions, return empty action (wait)
        if not available_jobs:
            return JSSPAction(job_ids=[])

        # Format prompt
        machines_status = self._format_machines_status(observation.machines, observation.step_count)
        jobs_list = self._format_jobs(observation.jobs)

        prompt = PROMPT_TEMPLATE.format(
            step_count=observation.step_count,
            machines_status=machines_status,
            jobs_list=jobs_list,
            legal_job_ids=[job.job_id for job in available_jobs],
        )
        print(f"Step {observation.step_count}")

        # Call LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model_id, messages=[{"role": "user", "content": prompt}], temperature=0.0
            )
            llm_output = response.choices[0].message.content or ""
            job_ids = self._parse_action(llm_output, available_jobs)

            # Ensure we don't schedule multiple jobs on the same machine
            # Track which machines we've already scheduled to avoid conflicts
            scheduled_machines = set()
            filtered_job_ids = []
            for job_id in job_ids:
                # Find the operation for this job
                op = next((op for op in available_jobs if op.job_id == job_id), None)
                if op is not None and op.operations[0][0] not in scheduled_machines:
                    filtered_job_ids.append(job_id)
                    scheduled_machines.add(op.operations[0][0])

            return JSSPAction(job_ids=filtered_job_ids)

        except Exception as e:
            print(f"Error calling LLM: {e}")
            print(f"Prompt: {prompt}")
            # On error, fall back to empty action (wait)
            return JSSPAction(job_ids=[])

    @staticmethod
    def _format_machines_status(machines: list[MachineObservation], current_step: int) -> str:
        """Format machine status for prompt."""
        lines = []
        for machine in machines:
            if machine.busy_until is None or machine.busy_until <= current_step:
                status = "Available"
            else:
                status = f"Busy until t={machine.busy_until}"
            job_info = f" (job {machine.current_job_id})" if machine.current_job_id is not None else ""
            lines.append(f"  Machine {machine.machine_id}: {status}{job_info}")
        return "\n".join(lines) if lines else "  (No machines)"

    @staticmethod
    def _format_jobs(jobs: list[JobObservation]) -> str:
        """Format jobs for prompt."""
        lines = []
        for job in jobs:
            available = job.busy_until is None
            operations = ", ".join(f"(Machine {op[0]}, {op[1]} min)" for op in job.operations)
            if available:
                lines.append(f"  Job {job.job_id}: Available. Remaining operations: {operations}")
            else:
                lines.append(f"  Job {job.job_id}: Busy until t={job.busy_until}. Remaining operations: {operations}")
        return "\n".join(lines) if lines else "  (No jobs)"

    @staticmethod
    def _parse_action(text: str, available_jobs: list[JobObservation]) -> list[int]:
        """Parse comma-separated job IDs from model output."""
        legal_job_ids = [job.job_id for job in available_jobs]

        # First, we remove the reasoning section
        text = text.split("<think>")[-1].split("</think>")[-1].strip()

        # First, try to split by comma and extract numbers from each part
        # This handles "2,3" or "2, 3" correctly
        parts = text.split(",")
        job_ids = []

        # Extract numbers from each part (handles "2" or "job 2" or " 2 ")
        for part in parts:
            numbers = re.findall(r"\d+", part.strip())
            for num_str in numbers:
                try:
                    job_id = int(num_str)
                    if job_id in legal_job_ids:
                        job_ids.append(job_id)
                except ValueError:
                    continue

        # If no comma-separated values found, try extracting all numbers
        # (handles cases like "Schedule jobs 2 and 3")
        if not job_ids:
            numbers = re.findall(r"\d+", text)
            for num_str in numbers:
                try:
                    job_id = int(num_str)
                    if job_id in legal_job_ids:
                        job_ids.append(job_id)
                except ValueError:
                    continue

        # Remove duplicates while preserving order
        seen = set()
        unique_job_ids = []
        for job_id in job_ids:
            if job_id not in seen:
                seen.add(job_id)
                unique_job_ids.append(job_id)

        return unique_job_ids if unique_job_ids else []  # Return empty list if no valid jobs found

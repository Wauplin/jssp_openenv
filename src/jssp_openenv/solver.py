from .client import JSSPEnvClient
from .models import ScheduledEvent
from .policy import JSSPEnvPolicy


def solve_jssp(
    env_client: JSSPEnvClient, policy: JSSPEnvPolicy, max_steps: int, verbose: bool = False
) -> tuple[int, list[ScheduledEvent]]:
    """Solve a single JSSP instance using the given policy."""
    result = env_client.reset()
    obs = result.observation
    scheduled_events: list[ScheduledEvent] = []

    while not result.done:
        if verbose:
            print(f"Step {obs.step_count}: {', '.join([str(job.job_id) for job in obs.available_jobs()])}")
        action = policy.act(obs)
        if verbose:
            print(f"Action: {action}")

        # Record scheduled events
        if action.job_ids:
            for job_id in action.job_ids:
                job = next((job for job in obs.available_jobs() if job.job_id == job_id), None)
                assert job is not None
                event = ScheduledEvent(
                    job_id=job_id,
                    machine_id=job.operations[0][0],
                    start_time=obs.step_count,
                    end_time=obs.step_count + job.operations[0][1],
                )
                scheduled_events.append(event)

        # Execute action
        result = env_client.step(action)
        obs = result.observation

        # Safety check to avoid infinite loops
        if obs.step_count >= max_steps:
            print(f"\nWARNING: Exceeded max steps ({max_steps}), terminating")
            break

    # Extract makespan
    return obs.step_count, scheduled_events

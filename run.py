import os
from enum import Enum

import typer
from openai import OpenAI

from jssp_openenv.client import JSSPEnvClient
from jssp_openenv.gantt import gantt_chart
from jssp_openenv.policy import JSSPEnvPolicy, JSSPFifoPolicy, JSSPLLMPolicy, JSSPMaxMinPolicy
from jssp_openenv.solver import solve_jssp

SERVER_URL = "http://localhost:8000"
MAX_STEPS = 1000  # Maximum number of steps per instance
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

cli = typer.Typer()


class PolicyName(str, Enum):
    FIFO = "fifo"
    LLM = "llm"
    MAX_MIN = "maxmin"


@cli.command()
def solve(
    policy: PolicyName = typer.Argument(help="The policy to use"),
    server_url: str = typer.Option(SERVER_URL, help="The URL of the JSSP server"),
    max_steps: int = typer.Option(MAX_STEPS, help="The maximum number of steps per instance"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Whether to print verbose output"),
    model_id: str = typer.Option(None, "--model-id", "-m", help="The ID of the model to use"),
):
    """Solve a JSSP instance using the given policy."""
    env_client = JSSPEnvClient(base_url=server_url)

    policy_obj: JSSPEnvPolicy
    match policy:
        case PolicyName.FIFO:
            policy_obj = JSSPFifoPolicy()
            title = "FIFO Policy"
            filename = "gantt_fifo_policy.png"

        case PolicyName.LLM:
            if not model_id:
                raise ValueError("You must set --model-id to use the LLM policy")
            api_key = os.getenv("HF_TOKEN")
            if not api_key:
                raise ValueError("You must set the HF_TOKEN environment variable to use the LLM policy")
            client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=api_key)
            policy_obj = JSSPLLMPolicy(client=client, model_id=model_id)
            title = f"LLM Policy ({model_id})"
            filename = f"gantt_llm_policy_{model_id.replace('/', '_').replace(':', '_').replace('-', '_').replace(' ', '_')}.png"

        case PolicyName.MAX_MIN:
            policy_obj = JSSPMaxMinPolicy()
            title = "Max-Min Policy"
            filename = "gantt_maxmin_policy.png"

    makespan, scheduled_events = solve_jssp(env_client, policy_obj, max_steps, verbose)

    if verbose:
        print("Schedule events:")
        for event in scheduled_events:
            print(
                f"[{event.start_time}] Scheduling job {event.job_id} on machine {event.machine_id} for {event.end_time - event.start_time} minute(s)"
            )

    print(f"Solved in {makespan} steps")

    filepath = os.path.join(OUTPUT_DIR, filename)
    gantt_chart(scheduled_events, title=title, makespan=makespan, save_to=filepath)
    print(f"Saved Gantt chart to {filepath}")


if __name__ == "__main__":
    cli()

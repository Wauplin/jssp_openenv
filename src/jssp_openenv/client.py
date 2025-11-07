from openenv_core import HTTPEnvClient, StepResult

from .models import JobObservation, JSSPAction, JSSPObservation, MachineObservation


class JSSPEnvClient(HTTPEnvClient[JSSPAction, JSSPObservation]):
    def _step_payload(self, action: JSSPAction) -> dict:
        return {"job_ids": action.job_ids}

    def _parse_result(self, payload: dict) -> StepResult[JSSPObservation]:
        obs_data = payload["observation"]
        return StepResult[JSSPObservation](
            observation=JSSPObservation(
                machines=[MachineObservation(**machine) for machine in obs_data.pop("machines")],
                jobs=[JobObservation(**job) for job in obs_data.pop("jobs")],
                **obs_data,
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> dict:
        return payload

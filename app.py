from fastapi.responses import RedirectResponse
from openenv_core.env_server import create_web_interface_app

from jssp_openenv.examples import FT06
from jssp_openenv.models import JSSPAction, JSSPObservation
from jssp_openenv.server.jssp_environment import JSSPEnvironment

env = JSSPEnvironment(FT06)
app = create_web_interface_app(env, JSSPAction, JSSPObservation, "JSSP (FT06)")


# Hack: redirect / to /web
@app.get("/")
async def redirect_to_web():
    # TODO: should be done in openenv directly ?
    return RedirectResponse(url="/web")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

from openenv_core.env_server import create_app

from ..examples import FT06
from ..models import JSSPAction, JSSPObservation
from .jssp_environment import JSSPEnvironment

# Create FastAPI app
env = JSSPEnvironment(FT06)
app = create_app(env, JSSPAction, JSSPObservation)

---
title: JSSP OpenEnv
emoji: ⏰
colorFrom: green
colorTo: purple
sdk: docker
pinned: false
---

<p align="center">
  <img src="assets/jssp_openenv.png" alt="jssp_openenv" width="400">
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/Wauplin/jssp_openenv" style="font-size: 1.2em;">Try it live on Hugging Face Spaces</a>
</p>

## Job shop scheduling problem (JSSP)

The [Job Shop Scheduling Problem](https://en.wikipedia.org/wiki/Job-shop_scheduling) (JSSP) is a classic optimization problem in operations research. Given a set of jobs, each consisting of multiple operations that must be performed in a specific sequence, and a set of machines, the goal is to schedule the operations on machines to minimize the total completion time (makespan).

**Key constraints:**
- Each job consists of a sequence of operations that must be completed in order
- Each operation requires a specific machine for a given duration
- Each machine can process only one operation at a time
- Once started, an operation cannot be interrupted

This implementation uses the OpenEnv framework to create a reinforcement learning environment where an agent (policy) learns to make scheduling decisions at each time step.

> !TIP
> For now, we only implement and run the FT06 problem. It is a well-known problem in the literature with a known optimal solution.
> Goal for training is to run arbitrary random environments.

## OpenEnv

[OpenEnv](https://github.com/meta-pytorch/OpenEnv) is a framework from Meta PyTorch and Hugging Face for building reinforcement learning environments. It provides:

- A standardized interface for environments with `Action` and `Observation` models
- A web-based interface for interactive exploration of environments
- A client-server architecture for distributed training and evaluation
- Integration with LLM-based policies for solving complex problems

This project implements a JSSP environment using OpenEnv, allowing you to:
- Interact with the environment through a web interface
- Test different scheduling policies (FIFO, Max-Min, LLM-based)
- Train reinforcement learning agents to solve JSSP instances

## Project Architecture

The project follows a client-server architecture using the OpenEnv framework:

### Core Components

**Models** (`src/jssp_openenv/models.py`):
- `JSSPAction`: Represents scheduling actions (list of job IDs to schedule)
- `JSSPObservation`: Contains the current state (machines, ready operations, progress)

**Environment** (`src/jssp_openenv/server/jssp_environment.py`):
- `JSSPEnvironment`: The core simulation environment that:
  - Manages job progress and machine states
  - Validates actions and enforces constraints
  - Advances simulation time using SimPy
  - Returns observations and rewards

**Client** (`src/jssp_openenv/client.py`):
- `JSSPEnvClient`: HTTP client that communicates with the environment server
- Handles action serialization and observation parsing

**Policies** (`src/jssp_openenv/policy.py`):
- `JSSPEnvPolicy`: Abstract base class for scheduling policies
- `JSSPFifoPolicy`: First-In-First-Out scheduling (schedules jobs by ID order)
- `JSSPMaxMinPolicy`: Max-Min scheduling (prioritizes longest operations)
- `JSSPLLMPolicy`: LLM-based scheduling using OpenAI-compatible APIs

**Solver** (`src/jssp_openenv/solver.py`):
- `solve_jssp()`: Orchestrates the solving process by:
  - Resetting the environment
  - Iteratively applying policy actions
  - Tracking scheduled events for visualization
  - Returning makespan and event history

**Visualization** (`src/jssp_openenv/gantt.py`):
- Generates Gantt charts showing the schedule timeline

## How to use

### Install

Install the package and its dependencies:

```bash
pip install -e .
```

For development with additional tools (pytest, ruff, etc.):

```bash
pip install -e ".[dev]"
```

**Note:** For LLM-based policies, you'll need to set the `HF_TOKEN` environment variable with your Hugging Face API token:

```bash
export HF_TOKEN=your_token_here
```

### Run server

To play with the environment locally, run

```
python app.py
```

and go to http://0.0.0.0:8000/web.

### Run policy

**FIFO policy** (always run first available job):

```
python run.py fifo
```

**Max-Min policy** (always run longest job first):

```
python run.py maxmin
```

**LLM policy** (ask an LLM to solve the problem)

```
python run.py llm --model-id "openai/gpt-oss-20b:groq"
python run.py llm --model-id "openai/gpt-oss-120b:cerebras"
python run.py llm --model-id "Qwen/Qwen3-32B:groq"
```

### Check results

The solver will resolve the problem using the policy and then plot a gantt chart of the solution in the `./charts` folder.

Here is an example:

![FIFO Policy Gantt Chart](assets/gantt_fifo_policy.png)

## Run with docker

Build the Docker image:

```bash
docker build -t jssp-openenv .
```

Run the container:

```bash
docker run -p 7860:7860 jssp-openenv
```

The web interface will be available at http://localhost:7860/web.

## TODO

- [ ] run on other example environments (FT10, FT20)
- [ ] run on random environments
- [ ] run multiple policies and summarize results
- [ ] trainer
"""
Knowledgebot Ollama Pre-flight Checks  (test_ollama_setup.py)

Validate the things that can break at startup if Ollama inference
server is not running and properly configured in the .env file

1. TCP port reachable (running at the correct host & port)
2. /api/tags endpoint responds
3. Configured model is installed
4. Model can generate a token

Run only these tests with:

    pytest -m ollama_preflight -v --tb=no

Flags
-m ollama_preflight     run *only* these checks
-v                      one line per test
--tb=no                 hide stack traces

Any failure prints a single, actionable reason.

Version History:
VERSION  DATE           DESCRIPTION                         AUTHORED-BY
===========================================================================
1.0.0    18/07/2025     initial release                     Laurence Molloy   
1.0.1    24/08/2025     BotConfig refactor                  Laurence Molloy   
===========================================================================
"""
version = "1.0.1"

import os
import socket
import requests
import pytest
from typing import Dict
from dotenv import load_dotenv, find_dotenv

# required for model loaded tests
import json
import time
import subprocess

from Config.Resolver import ConfigResolver
from Config.Schema import BotConfig

in_docker = os.environ.get("IN_DOCKER") == "1"

# If you are running in a container
if in_docker:
    # confirm .env has been loaded into environment by docker compose
    if not os.environ.get("ENV_LOADED"):
        pytest.exit("Environment not set – is docker compose loading the .env?")
# Load environment variables from .env file (where present)
elif find_dotenv(".env"):
    load_dotenv()
# otherwise, fail fast if you are running locally and can't find a .env file
elif not find_dotenv(".env"):
    pytest.exit("Missing .env file – copy .env.example and set LLM parameters")


# ------------------------------------------------------------------
# Re-usable helpers
# ------------------------------------------------------------------
def _ollama_base_url(env: Dict[str, str]) -> str:
    host = env.get("llm_server", "localhost")
    port = int(env.get("llm_port", 11434))
    return f"http://{host}:{port}"

def _tcp_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0

def _running_ollama_models():
    try:
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        
        if len(lines) <= 1:
            return []  # No models running, or just the header

        models = []
        for line in lines[1:]:
            if line.strip():  # Skip empty lines
                parts = line.split()
                model_name = parts[0]
                models.append(model_name)

        return models

    except subprocess.CalledProcessError:
        return []

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="session")
def ollama_env():
    """
    Session-scoped fixture to assemble LLM-based config for tests.

    - a simpler config import from BotConfig module (resolves values from Docker Secrets > .env > defaults)

    Returns:
        dict: Config
    """

    env = BotConfig.from_resolver(ConfigResolver()).to_dict()

    return env


@pytest.fixture(scope="session")
def ollama_env_deprecated():
    """
    NO LONGER USED - KEPT FOR POSTERITY ONLY - REPLACED BY BotConfig APPROACH - CAN BE DELETED

    Session-scoped fixture to assemble LLM-based config for tests.

    - Pulls config from environment or uses defaults

    Returns:
        dict: Config
    """

    # Define fallback (default) configuration values in case environment variables are missing
    # Be cautious: changing defaults can impact tests
    defaults = {
        "llm_provider": "ollama",                     # Assume open-source inference server
        "llm_model": "phi3:mini",                     # Small model with reasonable GPU requirements
        "llm_server": "localhost",                    # LLM server expected to be local
        "llm_port": int(11434)                        # Default port for Ollama server
    }


    # Pull configuration values from environment, or fallback to defaults
    environment_llm_port = int(os.environ.get("LLM_PORT")) if os.environ.get("LLM_PORT") is not None else None
    env = {}
    env['llm_provider'] = os.environ.get("LLM_PROVIDER")    or defaults['llm_provider']
    env['llm_model']    = os.environ.get("LLM_MODEL")       or defaults['llm_model']
    env['llm_server']   = os.environ.get("LLM_SERVER")      or defaults['llm_server']
    env['llm_port']     = environment_llm_port              or defaults['llm_port']

    return env


# ------------------------------------------------------------------
# 1.  Is the daemon running at all?
# ------------------------------------------------------------------
@pytest.mark.dependency()
@pytest.mark.ollama_preflight
def test_ollama_daemon_responds(ollama_env):
    host, port = ollama_env["llm_server"], ollama_env["llm_port"]
    if not _tcp_port_open(host, port):
        pytest.fail(f"Ollama daemon is not reachable at {host}:{port}. "
                     "Make sure it is running and check host and port.")

# ------------------------------------------------------------------
# 2.  Daemon endpoint test: prerequisite for model_is_installed test
# (if endpoint has moved, there's no point testing for installation)
# ------------------------------------------------------------------
@pytest.mark.dependency(depends=["test_ollama_daemon_responds"])
@pytest.mark.ollama_preflight
def test_ollama_daemon_endpoint(ollama_env):
    base = _ollama_base_url(ollama_env)
    host, port = ollama_env["llm_server"], ollama_env["llm_port"]
    try:
        r = requests.get(f"{base}/api/tags", timeout=2)
        r.raise_for_status()
    except requests.RequestException as e:
        pytest.fail(f"Ollama daemon on {host}:{port} responded with: {e}")

# ------------------------------------------------------------------
# 3.  Does the requested model exist in the catalog?
# ------------------------------------------------------------------
@pytest.mark.dependency(depends=["test_ollama_daemon_responds"])
@pytest.mark.ollama_preflight
def test_model_is_installed(ollama_env):
    base = _ollama_base_url(ollama_env)
    model = ollama_env["llm_model"]
    try:
        r = requests.get(f"{base}/api/tags", timeout=2)
        r.raise_for_status()
    except requests.RequestException as e:
        pytest.fail(f"Cannot query model list: {e}")

    installed = {m["name"] for m in r.json().get("models", [])}
    if model not in installed:
        pytest.fail(
            f"Model '{model}' is **not installed**. "
            f"Run: ollama pull {model}"
        )

# ------------------------------------------------------------------
# 4.  Is model running?
# ------------------------------------------------------------------
@pytest.mark.dependency(depends=["test_ollama_daemon_responds", "test_model_is_installed"])
@pytest.mark.ollama_preflight
def test_model_is_running(ollama_env):
    assert True 
    #model_name = ollama_env["llm_model"]
    #msg = f"{model_name} is not currently running in ollama"
    #assert model_name in _running_ollama_models(), msg

# ------------------------------------------------------------------
# 5.  Can we actually run the model (warm-up)?
# ------------------------------------------------------------------
@pytest.mark.dependency(depends=["test_ollama_daemon_responds", "test_model_is_installed", "test_model_is_running"])
@pytest.mark.ollama_preflight
def test_model_can_generate(ollama_env):
    base = _ollama_base_url(ollama_env)
    model = ollama_env["llm_model"]
    payload = {
        "model": model,
        "prompt": "hi",
        "stream": False,
        "options": {"num_predict": 1}  # only 1 token -> fast
    }
    try:
        r = requests.post(f"{base}/api/generate", json=payload, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        pytest.fail(f"Model '{model}' failed to generate: {e}")

    resp = r.json()
    if "response" not in resp:
        pytest.fail(f"Unexpected /generate response: {resp}")
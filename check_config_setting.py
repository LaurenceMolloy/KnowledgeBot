# A simple script that can check a config value and identify its source (docker secret or environment)
#
#  USAGE: python.exe check_config_setting.py <SETTING_NAME>
#  e.g. python.exe check_config_setting.py "ENABLE_LLM"

import os
from dotenv import load_dotenv, find_dotenv

import sys
sys.path.append('.')

from Config.Resolver import ConfigResolver
from Config.Schema import BotConfig

def test_resolver(setting_name: str):

    in_docker = os.environ.get("IN_DOCKER") == "1"

    # If you are running in a container
    if in_docker:
        # confirm .env has been loaded into environment by docker compose
        if not os.environ.get("ENV_LOADED"):
            print("In docker container, but environment not set – is docker compose loading the .env?")
            exit()
    # Load environment variables from .env file (where present)
    elif find_dotenv(".env"):
        load_dotenv()
    # otherwise, fail fast if you are running locally and can't find a .env file
    elif not find_dotenv(".env"):
        print("In local environment, but cannot find .env file – copy .env.example and set LLM parameters.")
        exit()

    print(f"Setting: {setting_name}")
    try:
        resolver = ConfigResolver()
        value = resolver.get(setting_name, default=BotConfig.get_default(setting_name))
        print(f"Value: {value.value}")
        print(f"Source: {value.source}")
    except ValueError as e:
        print("Value: UNDEFINED")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        setting = sys.argv[1]
    else:
        print("No config setting has been provided, presuming SLACK_TOKEN by default...")
        setting = "SLACK_TOKEN"  # default fallback
    test_resolver(setting)

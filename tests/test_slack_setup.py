"""
KnowledgeBot Slack Setup Verification

Filename: test_slack_setup.py

Purpose:
Loads environment config (.env) and verifies Slack bot setup before
use with the following pre-flight checks

  1. A valid Slack bot token is present
  2. All configured channels exist and are accessible
  3. The bot is a member of every configured channel
  4. Required Slack OAuth scopes are granted:

     - channels:history      (read message contents)
     - channels:read         (inspect channel membership)
     - reactions:write       (mark processed messages)
     - users:read            (resolve usernames)

Tests are marked with @pytest.mark.slack_preflight to allow selective 
execution of these configuration checks only when validating bot setup.

If the bot throws errors at startup, run this script to pin-point 
misconfigurations:

    pytest -m slack_preflight -v --tb=no

-m slack_preflight  only tests marked with "startup_check" are run
-v                  each test result is reported independently 
--tb=no             concise output with no code traceback 

Clear and concise failure reasons are given for all test failures.

Version History:
VERSION  DATE           DESCRIPTION                         AUTHORED-BY
===========================================================================
1.0.0    17/07/2025     initial release                     Laurence Molloy      
===========================================================================
"""
version = "1.0.0"

import pytest
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


# fail fast if you can't find a .env file
if not find_dotenv(".env"):
    pytest.exit("Missing .env file – copy .env.example and set SLACK_TOKEN")


# ------------------------------------------------------------------
# Re-usable helpers
# ------------------------------------------------------------------

def get_channel_ids(client, target_channels, channel_types):
    """
    Map each requested channel name to its Slack channel ID.

    Args:
        client: Instance of slack_sdk.WebClient.
        target_channels (list[str]): Channel names to find.
        channel_types (str): Comma-separated types (e.g. "public_channel,private_channel").

    Returns:
        dict: {channel_name: {"id": channel_id}} for found channels. Empty {} if none found.

    Notes:
        - Paginates through results (Slack returns up to 1000 per page).
        - Skips archived channels.
        - Fails silently if the API call errors.
    """
    channel_map = {}
    # Use a set for efficient O(1) lookups and removal as we find channels
    remaining_names_to_find = set(target_channels)

    cursor = None
    try:
        while True:
            # Request a page of channels matching the given channel type
            response = client.conversations_list(
                types=channel_types,
                exclude_archived=True,  # Ignore archived channels
                limit=200,              # Max allowed by Slack is 1000
                cursor=cursor           # Pagination cursor
            )
            # Loop through channels in the page
            for channel in response["channels"]:
                # Found a match, return its ID
                if 'name' in channel and channel["name"] in remaining_names_to_find:
                    channel_map[channel['name']] = { "id": channel["id"] }
                    # Remove from our "to-find" list
                    remaining_names_to_find.remove(channel["name"]) 

                    # If we've found all the channels we were looking for, we can stop
                    if not remaining_names_to_find:
                        return channel_map
            # Get the cursor for the next page, if any
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break   # No more pages to fetch
    except SlackApiError:
        return channel_map # If API call failed
    return channel_map     # If channel name not found


def handle_slack_error(e, channel_name="unknown"):
    """
    Handles common SlackApiError cases in pytest-based tests.

    Args:
        e (SlackApiError): Raised exception.
        channel_name (str): Channel name involved, for context.

    Behaviour:
        - Fails with clear explanation if bot isn’t in the channel.
        - Fails with missing scopes if token lacks permissions.
        - Skips test if auth is invalid (auth failure handled elsewhere).
        - Fails for other known Slack errors (e.g. rate limiting).
    """
    # Extract the Slack error code (e.g., 'not_in_channel', 'missing_scope', etc.)
    error = e.response.get("error", "unknown_error")

    if error == "not_in_channel":
        # The bot isn't a member of the channel being tested.
        pytest.fail(
            f"Bot is not a member of the '{channel_name}' channel. "
            "Invite the bot before running this check."
        )

    elif error == "missing_scope":
        # The token used does not have the required scope to perform the API call.
        needed = e.response.get("needed", "unspecified")
        pytest.fail(
            f"Bot is missing required scopes. Missing: {needed}"
        )

    elif error == "invalid_auth":
        # Invalid or expired token. This is considered a skip, assuming another test handles token validity.
        pytest.skip("Skipping: bot token is invalid or expired.")

    else:
        # Generic fallback for unexpected or unhandled Slack API errors.
        pytest.fail(
            f"Unexpected Slack API error: {error}"
        )

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="session")
def environment():
    """
    Session-scoped fixture to assemble config and Slack setup for tests.

    - Pulls config from environment or uses defaults
    - Sets up Slack WebClient
    - Resolves channel IDs for testing

    Returns:
        dict: Config, including Slack client object.
    """
    # Converts common "truthy" strings to booleans for feature flags (e.g. ENABLE_LLM)
    def str_to_bool(val: str) -> bool:
        return str(val).strip().lower() in ("1", "true", "yes", "on")
    
    # Converts comma-separated environment strings to Python lists
    def str_to_list(val: str) -> list:
        return str(val).strip().split(',')

    # Define fallback (default) configuration values in case environment variables are missing
    # Be cautious: changing defaults can impact tests
    defaults = {
        "enable_llm": False,                          # Don't use LLM unless explicitly enabled
        "llm_provider": "ollama",                     # Assume open-source inference server
        "llm_model": "phi3:mini",                     # Small model with reasonable GPU requirements
        "llm_server": "localhost",                    # LLM server expected to be local
        "llm_port": int(11434),                       # Default port for Ollama server
        "export_folder": Path('./data'),              # Folder to dump extracted #KNOWLEDGE to
        "state_file": Path('./state.json'),           # Track last run state to avoid repeated processing
        "slack_edit_channel": "test_edit",            # Slack channel used to force re-processing (#EDIT messages)
        "slack_knowledge_channels": ["test_knowledge"], # Channels mined for #KNOWLEDGE
        "slack_channel_types": "public_channel",      # Channel type(s) to search in Slack
        "bot_emoji": "mortar_board"                   # Marker emoji to track processed messages
    }

    # Load environment variables from .env file (if present)
    load_dotenv()

    # Pull configuration values from environment, or fallback to defaults
    environment_llm_port = int(os.environ.get("LLM_PORT")) if os.environ.get("LLM_PORT") is not None else None
    env = {}
    env['slack_token']         = os.environ.get("SLACK_TOKEN")                             or None
    env['enable_llm']          = str_to_bool(os.environ.get("ENABLE_LLM"))                 or defaults['enable_llm']
    env['llm_provider']        = os.environ.get("LLM_PROVIDER")                            or defaults['llm_provider']
    env['llm_model']           = os.environ.get("LLM_MODEL")                               or defaults['llm_model']
    env['llm_server']          = os.environ.get("LLM_SERVER")                              or defaults['llm_server']
    env['llm_port']            = environment_llm_port                                      or defaults['llm_port']
    env['export_folder']       = Path(os.environ.get("EXPORT_FOLDER"))                     or defaults['export_folder']
    env['state_file']          = os.environ.get("STATE_FILE")                              or defaults['state_file']
    env['slack_edit_channel']  = os.environ.get("SLACK_EDIT_CHANNEL")                      or defaults['slack_edit_channel']
    env['slack_channels']      = str_to_list(os.environ.get("SLACK_KNOWLEDGE_CHANNELS"))   or defaults['slack_knowledge_channels']
    env['slack_channel_types'] = os.environ.get("SLACK_CHANNEL_TYPES")                     or defaults['slack_channel_types']
    env['bot_emoji']           = os.environ.get("BOT_EMOJI")                               or defaults['bot_emoji']

    # Create the Slack WebClient using the supplied token
    env['client'] = WebClient(token=env['slack_token'])

    # Build {name: {id: …}} for every configured channel
    target_channels = env['slack_channels'] + [env['slack_edit_channel']]
    env['channel_map'] = get_channel_ids(env['client'], target_channels, env['slack_channel_types'])

    return env


@pytest.fixture
def validated_token(environment):
    '''
    Run auth test to validate token and get bot's user ID
    Returns (if auth test passes):
        dict: Config, including Slack client object AND the bot's user ID
    '''
    try:
        auth_test = environment['client'].auth_test()
        environment['bot_user_id'] = auth_test['user_id']
        return environment
    except:
        # Skip tests if token is invalid, expired, or otherwise broken
        pytest.skip("Skipping: bot token is invalid or expired.")

# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


@pytest.mark.slack_preflight
def test_slack_token_is_set(environment):
    '''
    Confirms whether the bot's Slack OAuth token has been set in the .env file.
    '''
    failure_message = "missing Slack OAuth token. Please set this in your .env file."
    assert environment['slack_token'] is not None, failure_message
        

@pytest.mark.slack_preflight
def test_slack_token_is_valid(environment):
    '''
    Confirms the validity of the bot's Slack OAuth token by performing an auth_test() API call.
    Identifies specific OAuth token failure modes (invalid, inactive, revoked, other (unexpected))
    '''
    try: 
        response = environment['client'].auth_test()
        failure_message = "auth_test() did not return ok: True"
        assert response["ok"] is True, failure_message
    except SlackApiError as e:
        error_code = e.response.get("error", "")
        # Handle specific authentication error types for diagnostic purposes
        if error_code == "invalid_auth":
            failure_message = ("Your slack bot's OAuth token is invalid. "
                               "Replace with a valid one obtained from https://api.slack.com/apps.")
        elif error_code == "account_inactive":
            failure_message = ("Your slack bot's OAuth token is valid but the bot has been deactivated. "
                               "Reinstall the bot's slack app and update your environment with the new OAuth token.")
        elif error_code == "token_revoked":
            failure_message = "Your slack bot's OAuth token has been revoked."
        else:
            failure_message = ("Unexpected Slack API error during OAuth token validation: "
                               f"{error_code} - {e.response.get('warning', '')}")
        pytest.fail(failure_message)
    except Exception as e:
        pytest.fail(f"Unexpected python error. {type(e).__name__}: {e}")


@pytest.mark.slack_preflight
def test_channel_existence(validated_token):
    """Fail if any configured Slack channel isn’t found in the workspace."""
    env = validated_token
    configured_channels = set(env['slack_channels'] + [env['slack_edit_channel']])
    channels_found = set(env['channel_map'])
    missing = configured_channels - channels_found
    failure_message = f"Missing Slack channels: {', '.join(missing)}"
    assert not missing, failure_message


@pytest.mark.parametrize(
    "api_call",
    [
        # Check permission to read message history
        pytest.param(lambda c, cid, uid: c.conversations_history(channel=cid, limit=1), id="channels:history"),
        # Check permission to read channel membership
        pytest.param(lambda c, cid, uid: c.conversations_members(channel=cid), id="channels:read"),
        # Check permission to look up user info
        pytest.param(lambda c, cid, uid: c.users_info(user=uid), id="users:read"),
    ],
)
@pytest.mark.slack_preflight
def test_single_scope(validated_token, api_call):
    """Verify that each required Slack scope is granted."""
    env = validated_token
    channel_id = env['channel_map'][env['slack_edit_channel']]['id']
    try:
        api_call(env['client'], channel_id, env['bot_user_id'])
    except SlackApiError as e:
        handle_slack_error(e, env['slack_edit_channel'])


@pytest.mark.slack_preflight
def test_scope_reactions_write(validated_token):
    """
    Check that the bot can add and remove reactions (emojis) in the #EDIT channel.

    This confirms:
    - Bot is a member of the #EDIT channel
    - Bot has 'channels:history' and 'reactions:write' scopes
    - NOTE: 'channels:history' scope is also tested elsewhere

    Ensures the bot can:
    - Fetch the most recent message
    - Add its marker emoji as a reaction
    - Remove the reaction
    """
    env = validated_token
    try:
        client = env['client']
        channel_name = env['slack_edit_channel']
        channel_id = env['channel_map'][channel_name]['id']
        emoji = env['bot_emoji']

        # Get the latest message from the channel
        response = client.conversations_history(channel=channel_id, limit=1)
        if not response.get('messages', []):
            pytest.skip("No messages in channel, skipping test.")
        message_ts = response['messages'][0]['ts']

        # Remove existing emoji where already present
        # prevents subsequent add() test from failing due to duplicate emoji
        try:
            client.reactions_remove(channel=channel_id, name=emoji, timestamp=message_ts)
        except:
            pass  # ignore this error — we'll test remove() properly in the next step

        # Add then remove the bot's marker emoji
        client.reactions_add(channel=channel_id,name=emoji,timestamp=message_ts)
        client.reactions_remove(channel=channel_id,name=emoji,timestamp=message_ts)
    except SlackApiError as e:
        handle_slack_error(e, channel_name)
    except Exception as e:
        pytest.fail(f"Unexpected python error: {type(e).__name__}: {e}")


@pytest.mark.slack_preflight
def test_channel_memberships(validated_token):
    """
    Verifies the bot is a member of all configured channels.

    Ensures:
    - Bot user ID is valid
    - Bot appears in the member list for each channel in channel_map

    Fails:
    - If the bot is missing from any expected channel
    """
    env = validated_token
    # Invalid or expired token. This is considered a skip, assuming another test handles token validity.
    failed_channels = []

    client = env['client']
    channel_map = env['channel_map']
    bot_user_id = env['bot_user_id']

    for channel_name, info in channel_map.items():
        channel_id = info['id']
        try:
            # Retrieve member list for the channel
            response = client.conversations_members(channel=channel_id)
            # Check if bot is present
            if bot_user_id not in response['members']:
                failed_channels.append(channel_name)
        except SlackApiError as e:
            handle_slack_error(e, channel_name)
        except Exception as e:
            pytest.fail(f"Unexpected python error: {type(e).__name__}: {e}")

    if len(failed_channels):
        # The bot isn't a member of the channel being tested.
        pytest.fail(
            f"Bot is not a member of one or more channels: {failed_channels}. "
            "Invite the bot to all of these channels before re-running this check."
        )
# MINIMUM SETUP REQUIREMENTS CONDITIONS:
# 1. a valid token has been specified in the environment
# 2. the following scopes have been granted: XXX, YYY, ZZZ
# 3. all configured channels exist
# 4. the bot is a member of all configured channels (use /invite @KnowledgeBot in the channel)

### TODO:
### 0. [DONE] DOCKERISE!!!
### 1. [WIP] README.md
### 1a. [DONE] THREADS!!!
### 2. [WIP] processing of the #EDIT channel
### 3. [DONE] LLM: Extract out a class structure for all LLM-Based functionality
### 4. LLM: extraction of name (user) entities (which aren't Slack mentions)
### 5. [PARTIAL - Instructor] LLM: use of LLM as a judge (filter/editor) for its output
### 6. [NEXT] - Switch for inclusion/exclusion of LLM functionality
### 7. Deployment to AWS as a lambda function (minus the LLM part)
### 8. Creation/Deployment of an Ollama Server ECS (Docker Compose etc...)
### 9. Lambda-controlled activation/deactivation of the Ollama Server ECS (for batch-based processing)
### 10. Pipeline (Knowledge, LLM-Enhanced Knowledge, Database): Great Expectations/DBT/Airflow/Snowflake
### 11. Add OpenAI ChatGPT support as proof of being able to switch model providers.
### 12. Private LLM for Github solution: https://dev.to/mcasperson/private-llms-for-github-actions-4nfa
### 13. [DONE] Multiple channel operation
### 14. what happens if an #EDIT message has a link to a message that is subsequently deleted?
### 15. CI/CD & Dependabot?
### 16. an ***option*** to use AWS SECRETS (must retain .env option for those without AWS access & dev/test purposes)
### 17. add channel descriptions (where populated) to the exported message files - provides additional RAG context
### 18. Abstract/Concrete classing for knowledge sources (of which Slack is one) - see how Vector Stores is done.
### 19. Add the ability to collect multiple hashtags for storing in different export folders

### 20. [DONE] filename as channel-date(orig)-timestamp(orig)-ordinal.txt
### 21. [DONE] deletion of old & re-creation of new knowledge chunk files
### 21. when information is "attached" through forwarding or linking should I process the inner content independently
###     (currently it's just appended, amnd thus inherits the linking context - the source context is lost)
### 22. what happens with e.g.  #KNOWLEDGE within #KNOWLEDGE if you know #END what I mean #END 
### 23. use a logging module

import pytest
import argparse

import re
import sys
import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

# Add this import for .env file support
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from LLMService.LLMService import LLMService

import inspect


class KnowledgeBot:
    
    def __init__(self, slack_token: str = None, enable_llm: bool = None,
                 llm_provider: str = None, llm_model: str = None, 
                 llm_server: str = None, llm_port: int = None,
                 export_folder: str = None, state_file: str = None):
        """
        Initialise the KnowledgeBot and ensure Slack authentication and scopes, LLM access (if enabled), 
        state management, and export folder setup.

        Configuration is loaded in this priority order: __init__ argument > environment variable > internal default.
        All __init__ arguments are optional and serve only to override corresponding environment variables.
        """

        # Helper function to convert string environment variables to boolean
        def str_to_bool(val: str) -> bool:
            return str(val).strip().lower() in ("1", "true", "yes", "on")
        
        # Helper function to convert comma-separated string environment variables to python lists
        def str_to_list(val: str) -> list:
            return str(val).strip().split(',')

        # Define sensible default configuration values (exercise caution when modifying these)
        defaults = {
            "enable_llm": False,                          # LLM usage is opt-in
            "llm_provider": "ollama",                     # default to open-source inference server
            "llm_model": "phi3:mini",                     # a small capable model (requires 4GB GPU memory)
            "llm_server": "localhost",                    # assume locally hosted LLM
            "llm_port": int(11434),                       # default ollama port
            "export_folder": Path('./data'),              # directory for storing extracted #KNOWLEDGE
            "state_file": Path('./data/state.json'),      # stores bot state (e.g. last run timestamp)
            "slack_thread_max_age_days": int(7),          # maximum age (in days) of threads to search through
            "slack_edit_channel": "test_edit",            # slack channel for #EDIT references (forces re-processing)
            "slack_knowledge_channels": str_to_list("test_knowledge"),  # slack channels to mine for #KNOWLEDGE
            "slack_channel_types": "public_channel",      # channel types include: public_channel, private_channel, im, mpim
            "bot_emoji": "mortar_board"                   # the bot's chosen message processing marker
        }

        # Load configuration in the defined priority order
        env_llm_port = int(os.environ.get("LLM_PORT")) if os.environ.get("LLM_PORT") is not None else None
        self.slack_token    = slack_token   or os.environ.get("SLACK_TOKEN")                or None
        self.enable_llm     = enable_llm    or str_to_bool(os.environ.get("ENABLE_LLM"))    or defaults['enable_llm']
        self.llm_provider   = llm_provider  or os.environ.get("LLM_PROVIDER")               or defaults['llm_provider']
        self.llm_model      = llm_model     or os.environ.get("LLM_MODEL")                  or defaults['llm_model']
        self.llm_server     = llm_server    or os.environ.get("LLM_SERVER")                 or defaults['llm_server']
        self.llm_port       = llm_port      or env_llm_port                                 or defaults['llm_port']
        self.export_folder  = export_folder or Path(os.environ.get("EXPORT_FOLDER"))        or defaults['export_folder']
        self.state_file     = state_file    or os.environ.get("STATE_FILE")                 or defaults['state_file']

        # not currently available via __init__() argument
        env_slack_thread_max_age_days = int(os.environ.get("SLACK_THREAD_MAX_AGE_DAYS")) if os.environ.get("SLACK_THREAD_MAX_AGE_DAYS") is not None else None
        self.slack_thread_max_age_days =  env_slack_thread_max_age_days                     or defaults['slack_thread_max_age_days']
        self.slack_edit_channel  = os.environ.get("SLACK_EDIT_CHANNEL")                     or defaults['slack_edit_channel']
        self.slack_channels      = str_to_list(os.environ.get("SLACK_KNOWLEDGE_CHANNELS"))  or defaults['slack_knowledge_channels']
        self.slack_channel_types = os.environ.get("SLACK_CHANNEL_TYPES")                    or defaults['slack_channel_types']
        self.bot_emoji           = os.environ.get("BOT_EMOJI")                              or defaults['bot_emoji']

        # A cache for Slack user name lookups - avoids hammering the API
        # Structure: { user_id : user_name } 
        self.user_cache: Dict[str, str] = {}

        # Load bot's persistent state from disk (currently just last processed timestamp)
        # Structure: { 'last_run_timestamp' : epoch } 
        self.state = self.load_bot_state()

        # Initialise Slack WebClient, validate authentication token and record the bot's user_id
        self.client = WebClient(token=self.slack_token)
        self.user_id = self._validate_slack_token()

        # Construct a dictionary of id-keyed channel metadata look-ups for all configured channels (including the #EDIT channel)
        # Structure: { channel_id: { 'name': str, 'topic': str, 'purpose': str } }
        self.channel_metadata = self._get_channel_metadata(self.slack_channels + [self.slack_edit_channel])

        # Instantiate LLM interface. Its usage is conditional on self.enable_llm
        self.llm = LLMService(
            provider=self.llm_provider,
            model=self.llm_model,
            server=self.llm_server,
            port=self.llm_port
        )

        # Ensure the export directory exists; create it if missing
        self.export_folder.mkdir(parents=True, exist_ok=True)
        print(f"KnowledgeBot initialised. Exporting to: {self.export_folder.resolve()}")

        # take a timestamp just prior to processing so that on next run we don't overlook any new messages due to race conditions
        self.start_timestamp = int(time.time())

    @property
    def error_context(self) -> str:
        """
        returns location in code (file, function & line number) for exceptions reporting
        """
        frame = inspect.stack()[1]  # 0 is this function, 1 is the caller
        file_name = os.path.basename(frame.filename)
        function_name = frame.function
        line_no = frame.lineno
        return f"[{file_name} - {function_name}() - L{line_no}]"

    @property
    def slack_edit_channel_id(self) -> str:
        """
        performs edit channel id lookup from channel name
        returns NULL if channel metadata hasn't been compiled from Slack yet
        """
        return next((cid for cid, meta in self.channel_metadata.items() 
                     if meta.get("name") == self.slack_edit_channel), None)


    def load_bot_state(self) -> dict:
        """
        Reads the bot's state from the instance's JSON state_file.

        Returns:
            dict: The loaded state as a dictionary, typically {'last_run_timestamp': epoch_float}.
            Returns {'last_run_timestamp': 0} (Unix epoch start) if file is not found, corrupt, or unreadable.
        """
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {self.state_file}. Starting with default state.")
                    # Fallback for corrupt JSON, explicitly return the default
                    return {'last_run_timestamp': 0}
                except Exception as e:
                    print(f"Error reading {self.state_file}: {e}. Starting with default state.")
                    # Fallback for other read errors, explicitly return the default
                    return {'last_run_timestamp': 0}
        else:
            # Default state for a new or missing state file (Unix epoch start)
            return {'last_run_timestamp': 0} 


    def save_bot_state(self):
        """
        Writes the bot's state back to the instance's JSON state_file.
        """
        self.state['last_run_timestamp'] = self.start_timestamp
        with open(self.state_file, "w") as f:
            # Use indent for human-readable formatting
            json.dump(self.state, f, indent=4) 
        

    def _validate_slack_token(self):
        """
        Validates the Slack API token by performing an auth.test API call, initialising a Slack WebClient if needed.
        Raises specific exceptions based on different authentication failure types.

        Raises:
            ValueError: if the token is undefined or invalid.
            ConnectionError: if the token is valid but the bot's account is inactive.
            PermissionError: if the Slack token has been revoked.
            Exception: For any other unexpected Slack API errors during validation.

        Returns:
            Bot Slack ID
            #No explicit return is required; successful completion without exception implies validation passed.
        """
        # Ensure a Slack token has been specified
        if self.slack_token is None:
            raise ValueError("Slack auth token is undefined. Please set SLACK_TOKEN in your environment or configuration.")
        
        # Initialise WebClient if not already done.
        # Allows _validate_slack_token() to be called defensively, when client setup has been overlooked.
        if not hasattr(self, 'client'):
            self.client = WebClient(token=self.slack_token)

        # Attempt to validate the token using Slack's API 
        try:
            response = self.client.auth_test()
            return response['user_id']
        except SlackApiError as e:
            error_code = e.response.get("error", "")
            # Handle specific authentication error types for diagnostic purposes
            if error_code == "invalid_auth":
                raise ValueError("Your slack auth token is invalid. Please replace with a valid one obtained from https://api.slack.com/apps.")
            elif error_code == "account_inactive":
                raise ConnectionError("Your slack auth token is valid but KnowledgeBot has been deactivated.\n"
                                      "Action required: reinstall the Slack app and update your environment with the new OAuth token.")
            elif error_code == "token_revoked":
                raise PermissionError(f"Your slack auth token has been revoked.")
            else:
                # Catch any other unexpected API errors
                raise Exception("Unexpected Slack API error during token validation: "
                                f"{error_code} - {e.response.get('warning', '')}")


    def _get_channel_metadata(self, target_channel_names: list[str]) -> dict[str, str]:
        """
        Maps channel names to their Slack channel IDs - currently limited to PUBLIC channels.
        Paginates through the list of Slack channels and stops once all requested channel 
        names have been resolved to their IDs.

        Args:
            target_channel_names: a list of human-friendly channel names to map.

        Required Scope(s):
            channels:read   (view basic information about public channels in a workspace)

        Exceptions Handled:
            PermissionError     If the bot lacks required scope(s).
            ValueError          No channels for mapping have been supplied.
            SlackApiError       For other unhandled Slack API errors returned by the API.
            RuntimeError        For unexpected internal Python errors during the process.

        Returns:
            Dict[ channel_id: Dict['name': str, 'topic': str, 'purpose': str] ]
        """
        # validate the Slack token if we haven’t already done so
        if not hasattr(self, 'user_id'):
            self.user_id = self._validate_slack_token()

        if not target_channel_names:
            raise ValueError("No target channel names provided. Nothing to process. Quitting.")

        channel_metadata = {}
        # Use a set for efficient O(1) lookups and removal as we find channels
        remaining_names_to_find = set(target_channel_names) 

        cursor = None

        try:
            while True:
                response = self.client.conversations_list(
                    types="public_channel", # Add other channel types later if needed e.g. private_channel, mpim
                    limit=100,              # Fetch channels in batches
                    cursor=cursor
                )

                if response["ok"]:
                    for channel in response["channels"]:
                        # Check if the channel has a name and if it's one of our targets
                        if 'name' in channel and channel["name"] in remaining_names_to_find:
                            channel_metadata[channel["id"]] = {
                                "name": channel["name"],
                                "topic": channel["topic"]["value"],
                                "purpose": channel["purpose"]["value"]
                            }

                            # Remove from our "to-find" list
                            remaining_names_to_find.remove(channel["name"]) 
                            
                            # If we've found all the channels we were looking for, we can stop
                            if not remaining_names_to_find:
                                print("Existence of all specified channels validated.")
                                return channel_metadata
                    
                    # Check for pagination: if 'next_cursor' exists, there are more pages
                    cursor = response.get("response_metadata", {}).get("next_cursor")
                    # No more pages to fetch
                    if not cursor:
                        break 
        except SlackApiError as e:
            error_message = e.response.get("error", "Unknown error")
            if error_message == "missing_scope":
                missing_scopes = str(e.response.get("needed")).split(',')
                raise PermissionError(f"{self.error_context}\n"
                                      f"Missing required OAuth scope(s): {missing_scopes}.\n"
                                       "Action required: Add the missing scope(s) to the KnowledgeBot App and reinstall it.")
            else:
                raise Exception(f"Caught Slack API Error while getting channel metadata: {e.response['error']}.\n"
                                f"Full error response: {e.response}")
        except Exception as e:
            raise RuntimeError(f"An unexpected Python error occurred: {type(e).__name__}: {e}")
                    
        # If the loop completes but not all channels were found
        if remaining_names_to_find:
            raise ValueError(f"Could not find IDs for the following channels in your environment: {list(remaining_names_to_find)}")

        return channel_metadata


    def _get_user_name(self, user_id: str) -> str:
        """
        Retrieves a user's real name from their ID, using a cache to avoid redundant API calls.

        Args:
            user_id (str): The Slack user ID.

        Returns:
            The user's real name or 'UnknownUser' if not found.
        """
        if user_id in self.user_cache:
            return self.user_cache[user_id]
        try:
            user_info = self.client.users_info(user=user_id)
            name = user_info['user']['real_name']
            self.user_cache[user_id] = name
            return name
        except SlackApiError:
            return "UnknownUser"

    def _get_mentioned_user_names(self, text: str) -> Optional[List[str]]:
        """
        Finds all mentioned user IDs in a text and resolves them to real names.

        Args:
            text (str): The message text.

        Returns:
            A list of mentioned real names, or None if no mentions are found.
        """
        mention_ids = re.findall(r"<@([A-Z0-9]+)>", text)
        if not mention_ids:
            return None
        return [self._get_user_name(uid) for uid in mention_ids]


    def _get_channel_members(self, channel_id: str) -> List[str]:
        """
        Gets a list of all member names in a given channel.

        Args:
            channel_id (str): The ID of the channel.

        Returns:
            A list of real names of the channel members.
        """
        try:
            response = self.client.conversations_members(channel=channel_id)
            return [self._get_user_name(uid) for uid in response['members']]
        except SlackApiError as e:
            print(f"Error fetching channel members for {channel_id}: {e.response['error']}")
            return ["Unknown"]


    def _delete_knowledge_chunks(self, channel_name: str, date_str: str, ts_str: str):
        """
        Deletes all existing #KNOWLEDGE chunks for a specific message (uniquely identified by 
        channel, date and timestamp) from the filesystem - this is called prior to commencing
        any reprocessing of the containing message to export its currently marked #KNOWLEDGE
        chunks.
        
        Args:
            channel_name: The channel name (unsanitized)
            date_str: Date in YYYYMMDD format
            ts_str: Slack timestamp with period replaced by underscore
        """
        # Sanitize channel name for filename matching
        safe_channel_name = re.sub(r'[^\w\-.]', '_', channel_name)
        
        # Construct the filename prefix to match
        prefix = f"{safe_channel_name}_{date_str}_{ts_str}_"
        
        # Find and delete all matching files
        deleted_count = 0
        for file in self.export_folder.glob(f"{prefix}*.txt"):
            try:
                file.unlink()
                deleted_count += 1
            except OSError as e:
                print(f"Error deleting file {file.name}: {e}")
        
        if deleted_count > 0:
            print(f"Deleted {deleted_count} existing #KNOWLEDGE chunk(s) for message {prefix}*")


    def _write_knowledge_chunks(self, snippet_number: int, content: str, metadata: Dict[str, Any]):
        """
        Writes the extracted #KNOWLEDGE block, prefixed with metadata, 
        to a file with a structured filename.
        
        Filename format: <channel>_YYYYMMDD_<slack_timestamp>_<first_10_chars>.txt
        Where:
            - channel: sanitized channel name
            - YYYYMMDD: message creation date
            - slack_timestamp: message timestamp with period replaced by underscore
            - snippet_number: ordinal position of #KNOWLEDGE chunk in message (2 digits)
        """

        channel_name = metadata['channel_name']
        date_str = metadata['date_str']  # Already in YYYYMMDD format from caller
        ts_str = metadata.get('timestamp', '0.0').replace('.', '_')
        
        # Sanitize channel name for filename
        safe_channel_name = re.sub(r'[^\w\-.]', '_', channel_name)

        # Construct filename
        filename = f"{safe_channel_name}_{date_str}_{ts_str}_{snippet_number:02d}.txt"
        filepath = self.export_folder / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"Channel Name: {metadata.get('channel_name', 'N/A')}\n")
                f.write(f"Channel Members: {', '.join(metadata.get('members', []))}\n")
                f.write(f"Message Date: {datetime.strptime(date_str, '%Y%m%d').strftime('%d %B %Y')}\n")
                f.write(f"Message Author: {metadata.get('user_name', 'N/A')}\n")
                if metadata.get('mentions'):
                    f.write(f"Mentions: {', '.join(metadata['mentions'])}\n")
                if metadata.get('keywords'):
                    f.write(f"Keywords: {', '.join(metadata['keywords'])}\n")
                if metadata.get('summary'):
                    f.write(f"Summary: {metadata['summary']}\n")
                f.write("\n---\n\n")
                f.write(content.strip())
            print(f"Successfully exported knowledge to {filepath}")
        except IOError as e:
            print(f"Error writing to file {filepath}: {e}")


    def process_channel(self, channel_id: str):
        """
        Processes a single channel for knowledge messages.

        Args:
            channel_id (str): The ID of the channel to process.
            since_timestamp (float): The Unix timestamp to fetch messages from.
                                     (e.g., 1751414400.0 for 01 July 2025)
        """
        #channel_name = next((name for name, id in self.channel_metadata.items() if id == channel_id), None)
        if not channel_id in self.channel_metadata:
            return

        members = self._get_channel_members(channel_id)
        channel_name = self.channel_metadata[channel_id]['name']
        print(f"\nProcessing channel: {channel_id} ({channel_name})")

        cursor = None
        has_more = True
        while has_more:
            try:
                adjusted_ts = f"{(float(self.state['last_run_timestamp']) - float(self.slack_thread_max_age_days * 86400)):.6f}"
                response = self.client.conversations_history(
                    channel=channel_id,
                    oldest=adjusted_ts,
                    inclusive=True,      
                    cursor=cursor,
                    limit=1000
                )
                messages = response.get('messages', [])
                print(f"Fetched {len(messages)} messages...")

                print(f"Checking for EDITED messages...")
                self._handle_edit_flags(channel_id, messages)

                for message in messages:
                    
                    reply_response = self.client.conversations_replies(channel=channel_id, ts=float(message["ts"]))
                    thread_messages = [
                        msg for msg in reply_response.get('messages', [])
                        if float(msg["ts"]) > float(self.state['last_run_timestamp'])
                    ]                

                    # append original thread message only if it post-dates last run time
                    if float(message["ts"]) > float(self.state['last_run_timestamp']):
                        thread_messages.insert(0, message)

                    for thread_message in thread_messages:

                        # Get message timestamp for file operations
                        ts = float(thread_message['ts'])
                        ts_str = str(ts).replace('.', '_')
                        date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%d")

                        # DELETE all Pre-existing #KNOWLEDGE chunks for this message
                        # (they are about to be re-exported)
                        self._delete_knowledge_chunks(channel_name, date_str, ts_str)

                        text = thread_message.get('text', '')

                        attachments = thread_message.get('attachments', [])
                        quoted = [a.get('text', '') for a in attachments]
                        text = text + '\n'.join(quoted)

                        if '#KNOWLEDGE' not in text:
                            continue

                        # Extract content between #KNOWLEDGE and #END tags
                        matches = re.findall(r'#KNOWLEDGE(.*?)#END', text, re.DOTALL)

                        if not matches:
                            continue

                        user_id = thread_message.get('user')
                        user_name = self._get_user_name(user_id)
                        print(f"Found knowledge message from user {user_id} ({user_name})")

                        # Process each #KNOWLEDGE block in the message
                        snippet_count = 0  # Initialize a #KNOWLEDGE block counter for this message
                        for content_block in matches:
                            content_block = content_block.strip()
                            if not content_block:
                                continue

                            snippet_count += 1  # Increment #KNOWLEDGE block counter
                            metadata = {
                                "channel_name": self.channel_metadata[channel_id]['name'],
                                "members": members,
                                "date_str": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%d"),
                                "timestamp": str(ts),  # Add this line,
                                "user_name": self._get_user_name(thread_message.get('user', 'UNKNOWN')),
                                "mentions": self._get_mentioned_user_names(text),
                                "keywords": self.llm.get_keywords(content_block, top_n=5),
                                "summary": self.llm.get_summary(content_block, max_length=25),
                            }
                            self._write_knowledge_chunks(snippet_count, content_block, metadata)
                        self._react_to_message(channel_id, ts, "mortar_board")

                has_more = response.get('has_more', False)
                cursor = response.get('response_metadata', {}).get('next_cursor')

            except SlackApiError as e:
                print(f"Error fetching messages for {self.channel_metadata[channel_id]['name']}: {e}")
                break # Exit loop on API error



    def _process_message(self, message):
        text = message.get('text', '')
        if '#KNOWLEDGE' not in text:
            return

        # Extract content between #KNOWLEDGE and #END tags
        matches = re.findall(r'#KNOWLEDGE(.*?)#END', text, re.DOTALL)
        if not matches:
            return

        user_id = message.get('user')
        user_name = self._get_user_name(user_id)
        print(f"Found knowledge message from user {user_id} ({user_name})")

        # Process each #KNOWLEDGE block in the message
        for content_block in matches:
            content_block = content_block.strip()
            if not content_block:
                continue

            ts = float(message['ts'])
            metadata = {
                #"channel_name": channel_name,
                #"members": members,
                "date_str": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%d"),
                "user_name": self._get_user_name(message.get('user', 'UNKNOWN')),
                "mentions": self._get_mentioned_user_names(text),
                "keywords": self._get_llm_keywords(content_block, top_n=5),
                "summary": self._get_llm_summary(content_block),
            }
            self._export_to_file(content_block, metadata)


    def _handle_edit_flags(self, channel_id: str, messages: List[dict]):
        """
        Look for #EDIT tags in thread replies and reprocess their parent messages.
        React to the #EDIT message with a ✅ emoji.
        """
        for message in messages:
            text = message.get("text", "")

            if "#EDIT" not in text:
                continue

            ts = message.get("ts")
            thread_ts = message.get("thread_ts")

            # Ensure it's a reply, not a freestanding brainfart
            if not thread_ts or thread_ts == ts:
                print("Found #EDIT but it's a reply. Ignoring.")
                continue

            print(f"Found #EDIT in thread, fetching parent message ts={thread_ts}")
            
            try:
                parent_response = self.client.conversations_replies(
                    channel=channel_id,
                    ts=thread_ts,
                    limit=1
                )
                messages_in_thread = parent_response.get("messages", [])
                if not messages_in_thread:
                    print("Parent message not found. Slack’s being helpful again.")
                    continue

                parent_message = messages_in_thread[0]
                print("Reprocessing parent message...")
                self._process_message(channel_id, parent_message)

                # Add reaction to acknowledge the edit
                self._react_to_message(channel_id, ts, "mortar_board")

            except SlackApiError as e:
                print(f"Error handling #EDIT message: {e.response['error']}")


    def _react_to_message(self, channel_id: str, ts: str, emoji: str):
        """
        Reacts to a message with the specified emoji.
        """
        try:
            self.client.reactions_add(
                channel=channel_id,
                name=emoji,
                timestamp=ts
            )
            print(f"Reacted to message at ts={ts} with :{emoji}:")
        except SlackApiError as e:
            if e.response["error"] == "already_reacted":
                print(f"Already reacted to ts={ts}. Good to know.")
            else:
                print(f"Failed to react to message: {e.response['error']}")


def run_preflight_checks(test_set):
    with open(os.devnull, 'w') as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            retcode = pytest.main(["-m", test_set, "-q", "--tb=no", "--disable-warnings"])
        finally:
            sys.stdout, sys.stderr = old_out, old_err
    return retcode


def parse_args():
    """
    Parse command-line arguments for the bot script.
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run the Slack KnowledgeBot with optional startup checks."
    )

    parser.add_argument(
        "--no-checks",
        action="store_true",
        help="Skip startup checks before running the bot."
    )

    parser.add_argument(
        "--state-file",
        default="data/state.json",
        help="Path to persistent state file"
    )

    return parser.parse_args()


def main():
    """
    Loads configuration, optionally runs startup checks, and kicks off channel processing.
    """

    args = parse_args()

    load_dotenv()

    if not args.no_checks and run_preflight_checks("slack_preflight"):
        print("Slack configuration pre-flight checks failed – aborting. Run 'pytest -m slack_preflight' for details.")
        return
    
    # only perform Ollama pre-flight tests if we have enabled LLM functionality in the .env
    if str(os.environ.get("ENABLE_LLM")).strip().lower() in ("1", "true", "yes", "on"):
        if not args.no_checks and run_preflight_checks("ollama_preflight"):
            print("Ollama configuration pre-flight checks failed – aborting. Run 'pytest -m ollama_preflight' for details.")
            return

    try:
        bot = KnowledgeBot()
        for slack_channel_id in bot.channel_metadata.keys():
            bot.process_channel(channel_id=slack_channel_id)
        bot.save_bot_state()
        print("\nProcessing complete.")
    except Exception as e:
        print(e)



if __name__ == "__main__":
    main()
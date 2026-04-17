# agent/tools.py
# Tool definitions for the ML Engineer Agent using OpenHands SDK

import os
from openhands.sdk import Tool
from openhands.tools.terminal import TerminalTool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool


def get_tools() -> list[Tool]:
    """
    Returns the tool set for the MIRA agent.

    TerminalTool    → runs Python/bash (auto-detects tmux or subprocess)
    FileEditorTool  → reads datasets, writes output JSON reports
    TaskTrackerTool → marks run complete, terminates the agent loop

    NOTE: OpenHands SDK requires a Linux environment.
          Run this agent inside WSL on Windows:
          wsl bash -c "cd /mnt/c/... && python3 -m agent.main"
    """
    return [
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
        Tool(name=TaskTrackerTool.name),
    ]
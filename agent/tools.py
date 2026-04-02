# agent/tools.py
# Tool definitions for the ML Engineer Agent using OpenHands SDK

import os
from openhands.sdk import Tool
from openhands.tools.terminal import TerminalTool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool


def get_tools() -> list[Tool]:
    """
    Returns the tool set for the ML Engineer Agent.

    TerminalTool   → runs Python/bash commands to profile data,
                     train models, compute metrics
    FileEditorTool → reads datasets, writes output reports
    TaskTrackerTool→ tracks phase completion, prevents runaway loops
    """
    return [
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
        Tool(name=TaskTrackerTool.name),
    ]
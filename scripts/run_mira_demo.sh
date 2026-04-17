#!/bin/bash
# scripts/run_mira_demo.sh
# MIRA Multi-Agent Demo — 4 pane tmux layout

SESSION="mira"
REPO="/mnt/c/Users/shiba/mis6349-shibani-11-p1"
VENV="source ~/mis6349-venv/bin/activate"

# Kill existing session if running
tmux kill-session -t $SESSION 2>/dev/null

# Create new session
tmux new-session -d -s $SESSION -x 220 -y 55

# Create 4 panes
tmux split-window -h -t $SESSION
tmux split-window -v -t $SESSION:0.0
tmux split-window -v -t $SESSION:0.2

# Pane 0 — Top Left — Data Analyst
tmux send-keys -t $SESSION:0.0 "cd $REPO && $VENV && clear" Enter
tmux send-keys -t $SESSION:0.0 "echo '╔══════════════════════════════════════════╗'" Enter
tmux send-keys -t $SESSION:0.0 "echo '║   🔍 AGENT 1: DATA ANALYST               ║'" Enter
tmux send-keys -t $SESSION:0.0 "echo '║   Phase: Data Exploration                ║'" Enter
tmux send-keys -t $SESSION:0.0 "echo '╚══════════════════════════════════════════╝'" Enter
tmux send-keys -t $SESSION:0.0 "echo ''" Enter
tmux send-keys -t $SESSION:0.0 "echo 'Status: Waiting to start...'" Enter

# Pane 1 — Bottom Left — ML Engineer
tmux send-keys -t $SESSION:0.1 "cd $REPO && $VENV && clear" Enter
tmux send-keys -t $SESSION:0.1 "echo '╔══════════════════════════════════════════╗'" Enter
tmux send-keys -t $SESSION:0.1 "echo '║   🤖 AGENT 2: ML ENGINEER                ║'" Enter
tmux send-keys -t $SESSION:0.1 "echo '║   Phase: Model Building                  ║'" Enter
tmux send-keys -t $SESSION:0.1 "echo '╚══════════════════════════════════════════╝'" Enter
tmux send-keys -t $SESSION:0.1 "echo ''" Enter
tmux send-keys -t $SESSION:0.1 "echo 'Status: Waiting for Agent 1...'" Enter

# Pane 2 — Top Right — ML Test Engineer
tmux send-keys -t $SESSION:0.2 "cd $REPO && $VENV && clear" Enter
tmux send-keys -t $SESSION:0.2 "echo '╔══════════════════════════════════════════╗'" Enter
tmux send-keys -t $SESSION:0.2 "echo '║   🧪 AGENT 3: ML TEST ENGINEER           ║'" Enter
tmux send-keys -t $SESSION:0.2 "echo '║   Phase: Model Testing                   ║'" Enter
tmux send-keys -t $SESSION:0.2 "echo '╚══════════════════════════════════════════╝'" Enter
tmux send-keys -t $SESSION:0.2 "echo ''" Enter
tmux send-keys -t $SESSION:0.2 "echo 'Status: Waiting for Agent 2...'" Enter

# Pane 3 — Bottom Right — Data Scientist
tmux send-keys -t $SESSION:0.3 "cd $REPO && $VENV && clear" Enter
tmux send-keys -t $SESSION:0.3 "echo '╔══════════════════════════════════════════╗'" Enter
tmux send-keys -t $SESSION:0.3 "echo '║   📊 AGENT 4: DATA SCIENTIST             ║'" Enter
tmux send-keys -t $SESSION:0.3 "echo '║   Phase: Recommendation + Evals          ║'" Enter
tmux send-keys -t $SESSION:0.3 "echo '╚══════════════════════════════════════════╝'" Enter
tmux send-keys -t $SESSION:0.3 "echo ''" Enter
tmux send-keys -t $SESSION:0.3 "echo 'Status: Waiting for Agent 3...'" Enter

# Attach
tmux attach -t $SESSION
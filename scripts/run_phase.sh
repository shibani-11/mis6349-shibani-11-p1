#!/bin/bash
# scripts/run_phase.sh
# Run a single MIRA phase
# Usage: bash scripts/run_phase.sh <phase>

PHASE=$1
REPO="/mnt/c/Users/shiba/mis6349-shibani-11-p1"
PYTHON="$HOME/mis6349-venv/bin/python3"

cd $REPO

case $PHASE in
  "data_exploration")
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║   🔍 DATA ANALYST AGENT STARTING         ║"
    echo "╚══════════════════════════════════════════╝"
    $PYTHON -m agent.main data_exploration
    ;;
  "model_building")
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║   🤖 ML ENGINEER AGENT STARTING          ║"
    echo "╚══════════════════════════════════════════╝"
    $PYTHON -m agent.main model_building
    ;;
  "model_testing")
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║   🧪 ML TEST ENGINEER AGENT STARTING     ║"
    echo "╚══════════════════════════════════════════╝"
    $PYTHON -m agent.main model_testing
    ;;
  "recommendation")
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║   📊 DATA SCIENTIST AGENT STARTING       ║"
    echo "╚══════════════════════════════════════════╝"
    $PYTHON -m agent.main recommendation
    ;;
  *)
    echo "Usage: bash scripts/run_phase.sh <phase>"
    echo "Phases: data_exploration | model_building | model_testing | recommendation"
    ;;
esac
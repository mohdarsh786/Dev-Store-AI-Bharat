#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║          DevStore AI Bharat — Backend Setup Script              ║
# ║          FastAPI + AWS Bedrock + Pinecone + Neon                 ║
# ╚══════════════════════════════════════════════════════════════════╝
set -e

# ANSI color codes
RESET="\033[0m"
BOLD="\033[1m"
CYAN="\033[96m"
GREEN="\033[92m"
YELLOW="\033[93m"
RED="\033[91m"
DIM="\033[2m"
BLUE="\033[94m"

clear

echo -e "${CYAN}${BOLD}"
echo " ██████╗ ███████╗██╗   ██╗    ███████╗████████╗ ██████╗ ██████╗ ███████╗"
echo " ██╔══██╗██╔════╝██║   ██║    ██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝"
echo " ██║  ██║█████╗  ██║   ██║    ███████╗   ██║   ██║   ██║██████╔╝█████╗  "
echo " ██║  ██║██╔══╝  ╚██╗ ██╔╝    ╚════██║   ██║   ██║   ██║██╔══██╗██╔══╝  "
echo " ██████╔╝███████╗ ╚████╔╝     ███████║   ██║   ╚██████╔╝██║  ██║███████╗"
echo " ╚═════╝ ╚══════╝  ╚═══╝      ╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝"
echo -e "${RESET}"
echo -e "${DIM}                   AI for Bharat — Developer Marketplace${RESET}"
echo -e "${DIM}                   ─────────────────────────────────────${RESET}"
echo ""

# ─── Resolve script directory ─────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Step 1: Python check ────────────────────────────────────────────────────
echo -e " ${BLUE}[1/5]${RESET} Checking Python..."
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo -e "\n ${RED}[ERROR]${RESET} Python not found."
    echo -e "        Install from ${CYAN}https://python.org${RESET} (v3.11+) and re-run."
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &>/dev/null; then
    PYTHON_CMD="python"
fi

PYTHON_VER=$($PYTHON_CMD --version)
echo -e "       ${GREEN}${PYTHON_VER}${RESET} detected. ✓"
echo ""

# ─── Step 2: .env check ───────────────────────────────────────────────────────
echo -e " ${BLUE}[2/5]${RESET} Checking environment configuration..."
if [ ! -f ".env" ]; then
    echo ""
    echo -e " ${YELLOW}╔══════════════════════════════════════════════════════════════════╗${RESET}"
    echo -e " ${YELLOW}║  ⚠  WARNING: .env not found!                                    ║${RESET}"
    echo -e " ${YELLOW}║                                                                  ║${RESET}"
    echo -e " ${YELLOW}║  Required backend config:                                        ║${RESET}"
    echo -e " ${YELLOW}║    DATABASE_URL=postgresql://user:pass@host/db                   ║${RESET}"
    echo -e " ${YELLOW}║    PINECONE_API_KEY=your-pinecone-api-key                        ║${RESET}"
    echo -e " ${YELLOW}║    AWS_ACCESS_KEY_ID=your-aws-access-key                         ║${RESET}"
    echo -e " ${YELLOW}║    AWS_SECRET_ACCESS_KEY=your-aws-secret-key                     ║${RESET}"
    echo -e " ${YELLOW}║    BEDROCK_MODEL_ID=your-bedrock-model-arn                       ║${RESET}"
    echo -e " ${YELLOW}║                                                                  ║${RESET}"
    echo -e " ${YELLOW}║  Copy .env.example to .env and configure.                       ║${RESET}"
    echo -e " ${YELLOW}╚══════════════════════════════════════════════════════════════════╝${RESET}"
    echo ""
    echo -e " ${DIM}Server will start but AI features will be disabled.${RESET}"
    echo ""
    read -r -p "  Continue anyway? [y/N] " choice
    case "$choice" in
        y|Y) echo "" ;;
        *)   echo -e "\n ${RED}Aborted.${RESET} Create .env and re-run." && exit 1 ;;
    esac
else
    echo -e "       .env found. Configuration ${GREEN}loaded${RESET}. ✓"
    echo ""
fi

# ─── Step 3: Virtual environment ─────────────────────────────────────────────
echo -e " ${BLUE}[3/5]${RESET} Setting up virtual environment..."
if [ ! -d "venv" ]; then
    echo -e "       Creating virtual environment with ${CYAN}${PYTHON_CMD}${RESET}..."
    $PYTHON_CMD -m venv venv
    echo -e "       Virtual environment created. ✓"
else
    echo -e "       Virtual environment exists. Skipping creation. ✓"
fi
echo ""

# ─── Step 4: Install dependencies ────────────────────────────────────────────
echo -e " ${BLUE}[4/5]${RESET} Installing Python dependencies..."
source venv/bin/activate
echo -e "       Virtual environment activated. Installing packages..."
echo ""
pip install -r requirements.txt
echo ""
echo -e "       Dependencies installed. ✓"
echo ""

# ─── Step 5: Launch ──────────────────────────────────────────────────────────
echo -e " ${BLUE}[5/5]${RESET} Launching FastAPI dev server..."
echo ""
echo -e " ${GREEN}${BOLD}╔══════════════════════════════════════════════════════════════════╗${RESET}"
echo -e " ${GREEN}${BOLD}║                                                                  ║${RESET}"
echo -e " ${GREEN}${BOLD}║   ✅  DevStore Backend Initialized — AI for Bharat               ║${RESET}"
echo -e " ${GREEN}${BOLD}║                                                                  ║${RESET}"
echo -e " ${GREEN}${BOLD}║   Backend API:  ${CYAN}http://localhost:8000${GREEN}                          ║${RESET}"
echo -e " ${GREEN}${BOLD}║   API Docs:     ${CYAN}http://localhost:8000/docs${GREEN}                     ║${RESET}"
echo -e " ${GREEN}${BOLD}║   Health Check: ${CYAN}http://localhost:8000/api/v1/health${GREEN}            ║${RESET}"
echo -e " ${GREEN}${BOLD}║                                                                  ║${RESET}"
echo -e " ${GREEN}${BOLD}║   Frontend (start separately):                                   ║${RESET}"
echo -e " ${GREEN}${BOLD}║     ${DIM}cd ../frontend${GREEN}                                             ║${RESET}"
echo -e " ${GREEN}${BOLD}║     ${DIM}npm install${GREEN}                                                ║${RESET}"
echo -e " ${GREEN}${BOLD}║     ${DIM}npm run dev${GREEN}                                                ║${RESET}"
echo -e " ${GREEN}${BOLD}║                                                                  ║${RESET}"
echo -e " ${GREEN}${BOLD}║   Press Ctrl+C to stop the server                                ║${RESET}"
echo -e " ${GREEN}${BOLD}╚══════════════════════════════════════════════════════════════════╝${RESET}"
echo ""

uvicorn main:app --reload --port 8000
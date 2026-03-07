#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║          DevStore AI Bharat — Frontend Setup Script             ║
# ║          Next.js 14 + AWS Bedrock + OpenSearch                  ║
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

# ─── Step 1: Node.js check ───────────────────────────────────────────────────
echo -e " ${BLUE}[1/4]${RESET} Checking Node.js..."
if ! command -v node &>/dev/null; then
    echo -e "\n ${RED}[ERROR]${RESET} Node.js not found."
    echo -e "        Install from ${CYAN}https://nodejs.org${RESET} (v18+) and re-run."
    exit 1
fi
NODE_VER=$(node --version)
echo -e "       Node.js ${GREEN}${NODE_VER}${RESET} detected. ✓"
echo ""

# ─── Step 2: .env.local check ────────────────────────────────────────────────
echo -e " ${BLUE}[2/4]${RESET} Checking environment configuration..."
if [ ! -f ".env.local" ]; then
    echo ""
    echo -e " ${YELLOW}╔══════════════════════════════════════════════════════════════════╗${RESET}"
    echo -e " ${YELLOW}║  ⚠  WARNING: .env.local not found!                              ║${RESET}"
    echo -e " ${YELLOW}║                                                                  ║${RESET}"
    echo -e " ${YELLOW}║  Required frontend config:                                       ║${RESET}"
    echo -e " ${YELLOW}║    BACKEND_URL=http://localhost:8000                             ║${RESET}"
    echo -e " ${YELLOW}║                                                                  ║${RESET}"
    echo -e " ${YELLOW}║  See README.md § 3 'Security Protocols' for all vars.           ║${RESET}"
    echo -e " ${YELLOW}╚══════════════════════════════════════════════════════════════════╝${RESET}"
    echo ""
    echo -e " ${DIM}App will start in demo mode (backend offline fallback).${RESET}"
    echo ""
    read -r -p "  Continue anyway? [y/N] " choice
    case "$choice" in
        y|Y) echo "" ;;
        *)   echo -e "\n ${RED}Aborted.${RESET} Create .env.local and re-run." && exit 1 ;;
    esac
else
    echo -e "       .env.local found. Secrets are ${GREEN}server-side only${RESET}. ✓"
    echo ""
fi

# ─── Step 3: npm install ─────────────────────────────────────────────────────
echo -e " ${BLUE}[3/4]${RESET} Installing dependencies..."
if [ ! -d "node_modules" ]; then
    echo -e "       node_modules not found. Running ${CYAN}npm install${RESET}..."
    echo ""
    npm install
    echo ""
    echo -e "       Dependencies installed. ✓"
else
    echo -e "       node_modules exists. Skipping install. ✓"
fi
echo ""

# ─── Step 4: Launch ──────────────────────────────────────────────────────────
echo -e " ${BLUE}[4/4]${RESET} Launching Next.js dev server..."
echo ""
echo -e " ${GREEN}${BOLD}╔══════════════════════════════════════════════════════════════════╗${RESET}"
echo -e " ${GREEN}${BOLD}║                                                                  ║${RESET}"
echo -e " ${GREEN}${BOLD}║   ✅  DevStore Initialized — AI for Bharat                       ║${RESET}"
echo -e " ${GREEN}${BOLD}║                                                                  ║${RESET}"
echo -e " ${GREEN}${BOLD}║   Frontend:   ${CYAN}http://localhost:3000${GREEN}                             ║${RESET}"
echo -e " ${GREEN}${BOLD}║   API Routes: ${CYAN}http://localhost:3000/api/*${GREEN}                       ║${RESET}"
echo -e " ${GREEN}${BOLD}║                                                                  ║${RESET}"
echo -e " ${GREEN}${BOLD}║   Backend (start separately):                                    ║${RESET}"
echo -e " ${GREEN}${BOLD}║     ${DIM}cd ../backend${GREEN}                                               ║${RESET}"
    echo -e " ${GREEN}${BOLD}║     ${DIM}uvicorn api_gateway:app --reload --port 8000${GREEN}                ║${RESET}"
echo -e " ${GREEN}${BOLD}║                                                                  ║${RESET}"
echo -e " ${GREEN}${BOLD}║   Press Ctrl+C to stop the server                                ║${RESET}"
echo -e " ${GREEN}${BOLD}╚══════════════════════════════════════════════════════════════════╝${RESET}"
echo ""

npm run dev

#!/bin/zsh
set -e

cd "$(dirname "$0")"

echo "Mimo ASR local UI"
echo "================="
echo "This window runs the local service at http://127.0.0.1:7860"
echo "Close this terminal window or press Ctrl+C to stop the service and release port 7860."
echo

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BOOTSTRAP="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BOOTSTRAP="python"
else
  echo "Python was not found. Please install Python 3.9 or newer, then run this file again."
  read "?Press Enter to close..."
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "Creating virtual environment in .venv ..."
  "$PYTHON_BOOTSTRAP" -m venv .venv
fi

PYTHON=".venv/bin/python"

echo "Installing project dependencies into .venv ..."
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r requirements.txt

if [ ! -f ".env" ]; then
  if [ -f ".env.example" ]; then
    cp ".env.example" ".env"
  else
    printf "MIMO_API_KEY=\nHF_TOKEN=\n" > ".env"
  fi
  echo
  echo "Created .env. You can fill API keys in the web UI settings."
fi

mkdir -p output

echo
echo "Opening http://127.0.0.1:7860 ..."

MIMO_ASR_OPEN_BROWSER=1 "$PYTHON" app.py

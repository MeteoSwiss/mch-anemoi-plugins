#!/usr/bin/bash

# install uv if not already installed
if ! command -v uv > /dev/null; then
    echo "🔧 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "✅ uv already installed."
fi

# sync (aka install) the virtual environment
echo "🔧 Syncing python virtual environment with uv..."
uv sync --all-extras #&> /dev/null
source .venv/bin/activate

clone_dir=./.venv/share/eccodes-cosmo-resources
export ECCODES_DEFINITION_PATH=$(realpath $clone_dir)/definitions
echo "🔧 ECCODES_DEFINITION_PATH set to ${clone_dir}/definitions"

# set up dev environment variables
echo
echo "✅ Environment ready!"

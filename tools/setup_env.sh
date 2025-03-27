#!/usr/bin/bash

# install uv if not already installed
if ! command -v uv > /dev/null; then
    echo "🔧 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "✅ uv already installed."
fi

ecbuild --version

# sync (aka install) the virtual environment
echo "🔧 Syncing python virtual environment with uv..."
uv sync --all-extras #&> /dev/null
source .venv/bin/activate


export ECCODES_DEFINITION_PATH=$(realpath $clone_dir)/definitions
echo "🔧 ECCODES_DEFINITION_PATH set to ${clone_dir}/definitions"

# patch eccodes definitions
echo "🔧 Patching eccodes definitions..."
TEMPLATES=("4.0" "4.1" "4.6" "4.8" "4.10" "4.11" "4.41" "4.42")
BASE_URL="https://raw.githubusercontent.com/ecmwf/eccodes/refs/heads/develop/definitions/grib2/templates"
for version in "${TEMPLATES[@]}"; do
    curl -s "${BASE_URL}/template.${version}.def" -o "$ECCODES_DEFINITION_PATH/grib2/template.${version}.def"
done


    
# set up dev environment variables
export PYTHONBREAKPOINT="ipdb.set_trace"
export FDB_ROOT=/scratch/mch/trajond/fdb-root-realtime
export FDB5_HOME=/scratch/mch/trajond/spack-view
export FDB_HOME=/scratch/mch/trajond/spack-view
export FDB5_CONFIG_FILE=/scratch/mch/trajond/fdb-realtime-lcm/realtime_config.yaml
export FDB5_DIR=/scratch/mch/trajond/spack-view
export GRIB_DEFINITION_PATH=/scratch/mch/trajond/eccodes-cosmo-resources/definitions:/scratch/mch/trajond/eccodes/definitions

echo 
echo "✅ Environment ready!"




#!/bin/bash

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cast_api_dir="${PROJECT_ROOT}/src/data_access/clarius"

# Set library path based on system
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    export DYLD_LIBRARY_PATH="$cast_api_dir:$DYLD_LIBRARY_PATH"
else
    # Linux
    export LD_LIBRARY_PATH="$cast_api_dir:$LD_LIBRARY_PATH"
fi

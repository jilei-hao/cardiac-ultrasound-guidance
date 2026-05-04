#!/bin/bash

# Get project root directory (always relative to this script location)
if [ -n "$ZSH_VERSION" ]; then
    SCRIPT_PATH="${(%):-%x}"
else
    SCRIPT_PATH="${BASH_SOURCE[0]}"
fi
PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
cast_api_dir="${PROJECT_ROOT}/src/data_access/clarius"

# Validate project structure
if [ ! -d "$PROJECT_ROOT/src" ]; then
    echo "❌ Error: src/ directory not found at $PROJECT_ROOT"
    echo "Make sure you run this script from the project directory or its subdirectories"
    return 1
fi

if [ ! -d "$cast_api_dir" ]; then
    echo "❌ Error: Clarius directory not found at $cast_api_dir"
    return 1
fi

if [ ! -f "$cast_api_dir/libcast.dylib" ] && [ ! -f "$cast_api_dir/libcast.so" ]; then
    echo "❌ Error: libcast library not found in $cast_api_dir"
    echo "Run: bash scripts/download_cast_api.sh"
    return 1
fi

# Set library path based on system
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="$cast_api_dir:$DYLD_LIBRARY_PATH"
    echo "✓ macOS: DYLD_LIBRARY_PATH=$cast_api_dir"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export LD_LIBRARY_PATH="$cast_api_dir:$LD_LIBRARY_PATH"
    echo "✓ Linux: LD_LIBRARY_PATH=$cast_api_dir"
else
    echo "⚠ OS not recognized: $OSTYPE"
fi

echo "✓ Project root: $PROJECT_ROOT"
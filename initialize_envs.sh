#! /bin/bash
current_file_dir="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
# echo "$current_file_dir"

clarius_dylib_path=$(realpath $current_file_dir/src/data_access/clarius)
# echo "$clarius_dylib_path"

# Detect OS and set appropriate library path variable
if [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS
  LIB_PATH_VAR="DYLD_LIBRARY_PATH"
  current_path="$DYLD_LIBRARY_PATH"
else
  # Linux and others
  LIB_PATH_VAR="LD_LIBRARY_PATH"
  current_path="$LD_LIBRARY_PATH"
fi

# check if the path already in library path
if [[ ":$current_path:" != *":$clarius_dylib_path:"* ]]; then
  # if not, add it to the path
  if [[ "$LIB_PATH_VAR" == "DYLD_LIBRARY_PATH" ]]; then
    export DYLD_LIBRARY_PATH=$clarius_dylib_path:$DYLD_LIBRARY_PATH
  else
    export LD_LIBRARY_PATH=$clarius_dylib_path:$LD_LIBRARY_PATH
  fi
else
  echo "Path already in $LIB_PATH_VAR"
fi

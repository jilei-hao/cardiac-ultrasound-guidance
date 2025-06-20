#! /bin/bash
current_file_dir="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
# echo "$current_file_dir"

clarius_dylib_path=$(realpath $current_file_dir/src/data_access/clarius)
# echo "$clarius_dylib_path"

# check if the path already in DYLD_LIBRARY_PATH
if [[ ":$DYLD_LIBRARY_PATH:" != *":$clarius_dylib_path:"* ]]; then
  # if not, add it to the path
  export DYLD_LIBRARY_PATH=$clarius_dylib_path:$DYLD_LIBRARY_PATH
else
  echo "Path already in DYLD_LIBRARY_PATH"
fi

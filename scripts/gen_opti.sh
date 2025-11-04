#!/bin/bash

# List of new names to update the YAML key 'name' with
names=(
"test00000"
"test00001"
)

# Path to your YAML file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# echo $SCRIPT_DIR

exp_name="$SCRIPT_DIR/../configs/exp_name.yaml"
test_config="$SCRIPT_DIR/../configs/test_config.yaml"
# echo $yaml_file

test="$SCRIPT_DIR/../testings/test.py"
test_fixed="$SCRIPT_DIR/../testings/test_fixed.py"

#Iterate over each name and update the YAML key 'name' for each one
for new_name in "${names[@]}"; do
  echo "Updating 'exp_name' key to: $new_name"
  yq eval ".exp_name = \"$new_name\"" -i "$exp_name"
  # batcat $exp_name

  echo "Updating 'test_name' key to: $new_name"
  yq eval ".test_name = \"$new_name\"" -i "$test_config"
  # batcat $test_config

  ray210
  cd $SCRIPT_DIR/../testings

  python "$test"


done

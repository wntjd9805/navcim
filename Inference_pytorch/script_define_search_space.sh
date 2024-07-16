#!/bin/bash

# Assign fixed file path to a variable
FILE="${NAVCIM_DIR}/Inference_pytorch/search_space.txt"

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <mode>"
    exit 1
fi

# Assign mode argument to variable
MODE=$1

# Check if the specified file exists
if [ ! -f "$FILE" ]; then
    echo "Error: File does not exist."
    exit 1
fi

# Read the entire file content
content=$(cat "$FILE")

# Modify content based on the mode
case "$MODE" in
    wo_accuracy)
        # Update the values for main1
        content=$(cat <<EOF
sa_set = 64, 128, 256
pe_set = 16
tile_set = 32
adc_bit = 5
cell_bit = 2
EOF
)
        ;;
    w_accuracy)
        # Update the values for main3
        content=$(cat <<EOF
sa_set = 64, 128, 256
pe_set = 16
tile_set = 32
adc_bit = 4,5,6
cell_bit = 1,2,4
EOF
)
        ;;
    *)
        echo "Error: Unknown mode."
        exit 1
        ;;
esac

# Write the updated content back to the file
echo "$content" > "$FILE"

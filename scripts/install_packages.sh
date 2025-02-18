#!/bin/bash 
set -e
set -o pipefail

####################################################################################################
###############This file is used to install the required packages for the project.##################
####################################################################################################


# Obtain the directory containing the script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Define the log file path relative to the script directory
log_file="$SCRIPT_DIR/../logs/package_installation_log.log"

# Resolve the absolute path of the log file
log_file=$(realpath "$log_file")

# Create or clear the log file
> "$log_file"


# Default logging behavior: log only to the log file
log_to_terminal=false

# Parse command-line arguments
while getopts ":t" OPTION; 
do
  case ${OPTION} in
    t)
      log_to_terminal=true
      ;;
    \?)
      echo "Usage: $0 [-t]"
      echo "  -t    Log output to both terminal and log file"
      exit 1
      ;;
  esac
done


# Configure logging based on user input
if [ "$log_to_terminal" = true ]; then
    # Log to both terminal and log file
    exec > >(tee -a "$log_file") 2>&1
    echo "Logging to both terminal and log file."
else
    # Log only to log file
    exec > "$log_file" 2>&1
    echo "Logging only to log file."
fi

echo -e "\n"

# Installing basic packages
echo "Installing basic packages ..."
echo -e "\n"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  
pip install opencv-python matplotlib Pillow augly tensorboardX 
echo "Basic packages are installed successfully." 

echo -e "\n\n\n" >> "$log_file"

# Installing packages for object detection
echo "Installing packages for object detection..." 
echo -e "\n"
cd "$SCRIPT_DIR/../GroundingDINO" 
pip install -e . 

# Downloading the pre-trained model for object detection
mkdir -p weights
cd weights
wget -nc -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

echo -e "\n"
echo "Object detection packages installed successfully." 


echo -e "\n\n" >> "$log_file"

echo "Packages installed successfully. Check the log file for more details: $log_file"

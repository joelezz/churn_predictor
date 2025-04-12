#!/bin/bash
# ------------------------------------------------------------------
# Project Setup and Data Integration Script
# ------------------------------------------------------------------
# Step 1.1: Set Up Your Project
# Install necessary dependencies: Flask, scikit-learn, and pandas.
echo "Installing required Python packages..."
pip install Flask scikit-learn pandas

# Update the PATH environment variable so that tools are found.
echo "Updating PATH environment variable..."
export PATH="/home/coder/.local/bin:$PATH"

# Confirm the updated PATH (optional)
echo "Current PATH: $PATH"

echo "Project setup complete. Note: If the lab environment is rebooted, please re-run this script."

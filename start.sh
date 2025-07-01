#!/bin/bash

# Update and install any required packages
apt-get update && apt-get install -y python3-pip

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Run your bot
python3 forex_signal_bot.py

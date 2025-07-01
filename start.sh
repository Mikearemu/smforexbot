#!/bin/bash

# Install TA-Lib C Library
apt-get update && apt-get install -y build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xvzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
make install
cd ..

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

# Run the bot
python forex_signal_bot.py

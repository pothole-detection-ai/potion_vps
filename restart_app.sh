#!/bin/bash

# Set the working directory to the script's directory
cd /root/potion

# Stop any running instances of app.py
/usr/bin/pkill -f app.py

# Start app.py in the background
/usr/bin/nohup /usr/bin/python3 /root/potion/app.py &> /root/potion/logfile.log &

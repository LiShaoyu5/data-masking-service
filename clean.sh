ps aux | grep "python.*main.py worker" | grep -v grep | awk '{print $2}' | xargs kill

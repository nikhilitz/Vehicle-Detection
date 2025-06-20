# chat/query_state.py

import json
import os

STATE_FILE = os.path.join(os.path.dirname(__file__), "../chat_cache/query.json")

def update_query(data):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(data, f)

def get_current_query():
    if not os.path.exists(STATE_FILE):
        return None
    with open(STATE_FILE, "r") as f:
        return json.load(f)

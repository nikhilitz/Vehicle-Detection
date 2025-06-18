# tests/test_command_chat_parser.py

import sys
import os

# Add project root to system path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chat.chat_command_parser import parse_chat_command

#  List of test chat commands
test_cases = [
    "show black cars",
    "track the red car with number plate HR26DN1234",
    "follow grey SUV",
    "locate white vehicle UP14 AB 1234",
    "trace blue car with plate HR 26 DN 1234",
    "show the car with number plate KA 01 AA 0001",
    "find yellow auto with number MH-12-DE-9876",
    "I want to see green vehicle",
    "just show all vehicles",
    "black car"
]

# 🔍 Run parser on each test case
for i, command in enumerate(test_cases, 1):
    print(f"\n🔹 Test {i}: '{command}'")
    result = parse_chat_command(command)
    print("Parsed Output:", result)

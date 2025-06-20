import re

def parse_chat_command(command):
    """
    Parses a user-entered chat command and extracts:
    - action (e.g., 'track', 'show')
    - vehicle color (normalized)
    - number plate (standardized)

    Returns:
        dict: { action, color, plate }
    """

    command = command.lower().strip()

    # 🎨 Supported color keywords (with 'grey' normalized to 'gray')
    colors = [
        "black", "white", "gray", "grey", "red", "orange",
        "yellow", "green", "blue", "brown"
    ]

    # 🎯 Keywords that imply tracking
    track_keywords = ["track", "follow", "find", "locate", "trace"]

    # 🧠 Defaults
    action = "show"
    selected_color = None
    plate = None

    # Step 1: Detect action type
    if any(word in command for word in track_keywords):
        action = "track"

    # Step 2: Detect color
    for color in colors:
        if color in command:
            selected_color = "gray" if color == "grey" else color
            break

    # Step 3: Detect plate — works for:
    # 'RJ14 AB 1234', 'RJ-14-AB-1234', 'rj14ab1234', etc.
    plate_match = re.search(
        r"\b([a-z]{2})[\s\-]?(\d{1,2})[\s\-]?([a-z]{1,3})[\s\-]?(\d{3,4})\b",
        command,
        re.IGNORECASE
    )
    if plate_match:
        plate = ''.join(plate_match.groups()).upper()  # e.g. RJ14AB1234

    return {
        "action": action,
        "color": selected_color,
        "plate": plate
    }

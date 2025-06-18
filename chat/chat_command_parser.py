import re

def parse_chat_command(command):
    command = command.lower()

    # Expanded color list
    colors = [
        "black", "white", "gray", "grey", "red", "orange", "yellow",
        "green", "blue", "brown"
    ]

    # Expanded actions: synonyms of 'track'
    track_keywords = ["track", "follow", "find", "locate", "trace"]

    action = "show"
    selected_color = None
    plate = None

    #  Check for tracking intent
    if any(word in command for word in track_keywords):
        action = "track"

    #  Detect color
    for color in colors:
        if color in command:
            selected_color = "gray" if color == "grey" else color  # Normalize
            break

    #  Improved plate matching (handles spaces or dash-separated plates)
    plate_match = re.search(
        r"\b([a-z]{2})[\s-]?(\d{1,2})[\s-]?([a-z]{1,3})[\s-]?(\d{3,4})\b",
        command,
        re.IGNORECASE
    )
    if plate_match:
        plate = ''.join(plate_match.groups()).upper()

    return {
        "action": action,
        "color": selected_color,
        "plate": plate
    }

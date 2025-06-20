import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "vehicle_data.db")


def init_db():
    """
    Creates the `vehicle_detections` table if it doesn't already exist.
    This stores plate info, vehicle color, camera source, and timestamp.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT UNIQUE,         -- License plate (no duplicates)
                color TEXT,                -- Detected vehicle color
                camera TEXT,               -- Source camera name
                timestamp TEXT             -- Time of detection
            )
        """)
        conn.commit()


def insert_detection(plate, color, camera, timestamp=None):
    """
    Inserts or updates a vehicle detection into the database.

    - Skips "N/A" or empty plates
    - Converts all plates to uppercase, color to lowercase
    - If the plate already exists, updates its timestamp/color/camera

    Params:
    - plate (str): License plate text
    - color (str): Vehicle color
    - camera (str): Camera ID
    - timestamp (str): Optional custom time; defaults to current time
    """
    if not plate or plate.strip().upper() == "N/A":
        return  # Skip invalid entries

    processed_plate = plate.strip().upper()
    processed_color = color.strip().lower()
    timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO vehicle_detections (plate, color, camera, timestamp)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(plate) DO UPDATE SET
                    timestamp = excluded.timestamp,
                    color = excluded.color,
                    camera = excluded.camera
                WHERE plate = excluded.plate;
            """, (processed_plate, processed_color, camera, timestamp))
            conn.commit()

            # Simple success message (insert or update)
            print(f"✔️ Database: Plate {processed_plate} processed at {timestamp}.")

        except sqlite3.Error as e:
            print(f"❌ Database error while processing plate {processed_plate}: {e}")


def query_latest_match(color=None, plate=None):
    """
    Fetch the most recent vehicle match based on color and/or plate.

    Params:
    - color (str): Vehicle color (optional)
    - plate (str): License plate (optional)

    Returns:
    - dict with plate, color, camera, timestamp OR None if no match found
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        query = "SELECT plate, color, camera, timestamp FROM vehicle_detections"
        conditions = []
        params = []

        if plate:
            conditions.append("plate = ?")
            params.append(plate.strip().upper())

        if color:
            conditions.append("color = ?")
            params.append(color.strip().lower())

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp DESC LIMIT 1"

        cursor.execute(query, params)
        row = cursor.fetchone()

        if row:
            return {
                "plate": row[0],
                "color": row[1],
                "camera": row[2],
                "timestamp": row[3]
            }

        return None


def fetch_all_detections():
    """
    Returns all stored vehicle detections in reverse chronological order.

    Returns:
    - List of dictionaries with id, plate, color, camera, and timestamp
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, plate, color, camera, timestamp 
            FROM vehicle_detections 
            ORDER BY timestamp DESC
        """)
        rows = cursor.fetchall()

        return [
            {
                "id": row[0],
                "plate": row[1],
                "color": row[2],
                "camera": row[3],
                "timestamp": row[4]
            }
            for row in rows
        ]


def delete_all_detections():
    """
    Deletes all entries from the database (useful during testing).
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM vehicle_detections")
        conn.commit()
        print("🧹 All detections deleted from the database.")

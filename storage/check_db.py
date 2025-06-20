import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "vehicle_data.db")

def show_all_detections():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM vehicle_detections ORDER BY id DESC")
        rows = cursor.fetchall()

        print("📋 All Vehicle Detections:")
        for row in rows:
            print(f"ID: {row[0]}, Plate: {row[1]}, Color: {row[2]}, Camera: {row[3]}, Time: {row[4]}")

if __name__ == "__main__":
    show_all_detections()

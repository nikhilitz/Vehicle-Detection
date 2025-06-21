import streamlit as st
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chat.chat_command_parser import parse_chat_command
from chat.query_state import update_query, get_current_query
from storage.database import query_latest_match

st.set_page_config(page_title="AI Vehicle Monitoring", layout="centered")
st.title("🚗 Vehicle Monitoring System")


with st.form("chat_form"):
    chat_input = st.text_input(
        "💬 Enter tracking command:",
        placeholder="e.g., Track black car RJ14AB1234"
    )
    submitted = st.form_submit_button("Submit")

    if submitted:
        if chat_input.strip():
            parsed = parse_chat_command(chat_input)

            # 🧹 Normalize fields
            parsed["plate"] = parsed["plate"].upper() if parsed["plate"] else None
            parsed["color"] = parsed["color"].lower() if parsed["color"] else None

            # Save query state for live tracker
            update_query(parsed)

            st.success("✅ Command processed successfully!")
        else:
            st.warning("⚠️ Please enter a valid command before submitting.")


if submitted and chat_input.strip():
    st.markdown("---")
    st.header("📡 Latest Matching Vehicle Detection")

    query = get_current_query()

    if not query:
        st.info("ℹ️ No tracking data found.")
    else:
        st.subheader("🔍 Current Query")
        st.write(f"**Color:** `{query['color'] or 'Any'}`")
        st.write(f"**Plate:** `{query['plate'] or 'Any'}`")

        # Fetch latest match from database
        match = query_latest_match(color=query["color"], plate=query["plate"])

        if match:
            st.success("✅ Vehicle MATCH FOUND!")
            st.markdown(f"**Plate:** `{match['plate']}`")
            st.markdown(f"**Color:** `{match['color']}`")
            st.markdown(f"**Camera:** `{match['camera']}`")
            st.markdown(f"**Time:** `{match['timestamp']}`")

            if match.get("image_path") and os.path.exists(match["image_path"]):
                st.image(match["image_path"], caption="📸 Detected Vehicle Image", use_column_width=True)
            else:
                st.warning("⚠️ No image available for this match.")
        else:
            st.warning("🚫 No matching vehicle found yet. Waiting for detection...")

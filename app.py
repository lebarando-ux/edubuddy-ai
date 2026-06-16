# Run this app using: streamlit run app.py

import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load local environment variables (if running locally)
load_dotenv()


def configure_page():
    """Sets up the Streamlit page configuration and custom CSS."""
    st.set_page_config(
        page_title="EduBuddy - Your AI Private Teacher",
        page_icon="🎓",
        layout="centered"
    )

    st.markdown("""
    <style>
        .subtitle {
            font-size: 1.15rem;
            color: #4B5563;
            margin-top: -15px;
            margin-bottom: 25px;
            text-align: left;
            font-style: italic;
        }
    </style>
    """, unsafe_allow_html=True)


def get_api_key():
    """Retrieves the API key from Streamlit secrets, env variables, or user input."""
    api_key = os.getenv("GEMINI_API_KEY")

    # Safely check secrets as a fallback ONLY if no error is thrown
    try:
        if not api_key and "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass  # Ignore the error if the secrets file doesn't exist locally

    if not api_key or api_key == "your_gemini_api_key_here":
        st.sidebar.warning("⚠️ GEMINI_API_KEY is missing.")
        api_key_input = st.sidebar.text_input(
            "Enter Gemini API Key to unlock:", type="password")
        if api_key_input:
            return api_key_input
        else:
            st.info(
                "💡 To begin, please enter your Gemini API Key in the sidebar or update your `.env` file.")
            st.stop()

    return api_key


@st.cache_resource
def get_genai_client(api_key):
    """Initializes and caches the GenAI client."""
    return genai.Client(api_key=api_key)


def build_system_prompt(subject, complexity, style):
    """Constructs the dynamic system instruction based on user settings."""
    return f"""
    You are "EduBuddy", an empathetic, highly knowledgeable, and engaging private teacher and learning guide. 
    Your goal is to help the student understand concepts deeply, rather than just giving them direct answers.

    Follow these pedagogical principles:
    1. **Explain Conceptually**: Use analogies, simple real-world examples, and step-by-step breakdowns.
    2. **Use the Socratic Method**: If the teaching style is a Socratic Guide, do not immediately provide the answer. Instead, ask guiding questions or give hints.
    3. **Adapt to the Student**: Adjust your vocabulary, details, and complexity based on the student's selected level: {complexity}.
    4. **Be Encouraging**: Praise correct reasoning, validate effort, and maintain a patient, positive tone.
    5. **Format Clearly**: Use Markdown, bullet points, bold text for key terms, and code blocks for code or math equations to make information digestible.

    Current Subject/Topic Focus: {subject}
    Student's Learning Level: {complexity}
    Teaching Style/Personality: {style}

    Always stay in character as a helpful private tutor.
    """


def main():
    configure_page()

    # --- Header ---
    st.title("🎓 EduBuddy: Your AI Private Teacher")
    st.markdown('<p class="subtitle">Personalized Socratic guidance to help you learn, solve problems, and master concepts.</p>', unsafe_allow_html=True)

    # --- Authentication & Client Initialization ---
    api_key = get_api_key()
    try:
        client = get_genai_client(api_key)
    except Exception as e:
        st.error(f"Failed to initialize GenAI Client: {e}")
        st.stop()

    # --- Sidebar Configuration ---
    st.sidebar.title("🏫 Classroom Settings")
    subject = st.sidebar.selectbox(
        "Focus Subject/Topic:",
        ["General / Multidisciplinary", "Mathematics & Algebra", "Science (Physics, Chemistry, Biology)",
         "Computer Science & Programming", "History & Social Studies", "Languages & Essay Writing"]
    )
    complexity_level = st.sidebar.select_slider(
        "Explanation Depth:",
        options=["Explain Like I'm 5 (Analogy Heavy)", "High School Level",
                 "University Student", "Expert / Professional Level"],
        value="High School Level"
    )
    teaching_style = st.sidebar.radio(
        "Tutor Personality Style:",
        ["💡 Socratic Guide", "🤗 Patient & Encouraging",
            "🧠 Strict & Rigorous", "🎨 Creative Storyteller"]
    )

    if st.sidebar.button("🗑️ Reset Chat History"):
        st.session_state.messages = []
        st.rerun()

    # --- Session State Initialization ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Render Chat History ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Chat Input & Response Logic ---
    if user_input := st.chat_input("Ask EduBuddy a question or ask for help with a problem..."):

        # Display and store user message
        st.session_state.messages.append(
            {"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display and stream assistant response
        with st.chat_message("assistant"):
            try:
                # 1. Map history to SDK format
                sdk_history = [
                    types.Content(
                        role="user" if msg["role"] == "user" else "model",
                        parts=[types.Part.from_text(text=msg["content"])]
                    ) for msg in st.session_state.messages[:-1]
                ]

                # 2. Build dynamic system prompt
                system_prompt = build_system_prompt(
                    subject, complexity_level, teaching_style)

                # 3. Create chat session
                chat = client.chats.create(
                    model="gemini-2.5-flash",
                    history=sdk_history,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt
                    )
                )

                # 4. Stream the response directly to the UI
                response_stream = chat.send_message_stream(user_input)

                def stream_generator():
                    for chunk in response_stream:
                        if chunk.text:
                            yield chunk.text

                # st.write_stream creates the typewriter effect automatically
                full_response = st.write_stream(stream_generator())

                # 5. Save the completed response to state
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Error communicating with Gemini API: {str(e)}")
                st.session_state.messages.pop()  # Remove failed user message from history


if __name__ == "__main__":
    main()

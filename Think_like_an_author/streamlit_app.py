# streamlit_app.py
import streamlit as st
from workflow import AuthorWorkflow, run_conversation
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Configure Google Sheets
def connect_to_sheet():
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    client = gspread.authorize(creds)
    return client.open("AuthorMind_Feedback").sheet1

# App layout
st.set_page_config(layout="wide", page_title="AuthorMind AI", page_icon="📚")

# Title Section
col1, col2 = st.columns([3, 1])
with col1:
    st.title("AuthorMind AI")
    st.markdown("### Think Like Your Favorite Author")

# Main Content
main_col, side_col = st.columns([3, 1])

with main_col:
    # Chat Container
    chat_container = st.container(height=600, border=False)
    
    # Display Chat History
    for q, a in st.session_state.chat_history:
        with chat_container.chat_message("user"):
            st.write(q)
        with chat_container.chat_message("assistant"):
            st.write(a)
    
    # Input Section
    with st.container():
        topic = st.text_input("Enter Topic", key="topic_input")
        author = st.text_input("Enter Author Name", key="author_input")
        user_input = st.chat_input("Ask your question...")

with side_col:
    # About Section
    with st.expander("About the Project", expanded=True):
        st.markdown("""
        AuthorMind AI helps you think like your favorite authors by:
        - Analyzing their YouTube interviews & recent articles
        - Mimicking their communication style
        - Providing insights in their voice
            
        Example: Ask about "Wealth Building" from Robert Kiyosaki's perspective
        """)
    
    # Features Section
    with st.expander("Technical Features"):
        st.markdown("""
        - LLM: GPT-3.5 Turbo
        - Vector DB: FAISS
        - Embeddings: OpenAI text-embedding-3-small
        - Search: YouTube API + DuckDuckGo
        - Chat History: Last 5 exchanges
        """)
    
    # Feedback Section
    with st.form("feedback_form"):
        feedback = st.radio("Rate this response", ["👍 Like", "👎 Dislike"])
        comment = st.text_area("Additional comments")
        submitted = st.form_submit_button("Submit Feedback")
        
        if submitted:
            sheet = connect_to_sheet()
            sheet.append_row([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                st.session_state.get('current_topic', ''),
                st.session_state.get('current_author', ''),
                feedback,
                comment
            ])
            st.success("Thanks for your feedback!")

# Conversation Handling
if user_input and topic and author:
    # Initialize workflow
    if not st.session_state.conversation:
        st.session_state.conversation = AuthorWorkflow()
        st.session_state.current_topic = topic
        st.session_state.current_author = author
    
    # Run conversation
    with st.spinner(f"Thinking like {author}..."):
        response = run_conversation(
            workflow=st.session_state.conversation,
            user_input=user_input,
            topic=topic,
            author=author,
            chat_history=st.session_state.chat_history
        )
        
    # Update chat history
    st.session_state.chat_history.append((user_input, response))
    st.rerun()
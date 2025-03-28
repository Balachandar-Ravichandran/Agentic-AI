# author_mind.py
import streamlit as st
from dotenv import load_dotenv
import os
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Load LangGraph workflow from existing code
from your_existing_code import FinancialWorkflow, State, run_conversation

# Initialize secrets
load_dotenv()


# Configure Google Sheets
def connect_to_sheet():
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    client = gspread.authorize(creds)
    return client.open("AuthorMind_Feedback").sheet1

# Streamlit UI Configuration
st.set_page_config(page_title="AuthorMind AI", layout="wide")

# Initialize session state
if 'workflow' not in st.session_state:
    st.session_state.workflow = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# App layout
def main():
    # Title Section
    st.title("📚 AuthorMind AI")
    st.markdown("### Think Like Your Favorite Author")
    
    # Main columns
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat container
        chat_container = st.container(height=600)
        
        # Display chat history
        for q, a in st.session_state.chat_history:
            with chat_container.chat_message("user"):
                st.write(q)
            with chat_container.chat_message("assistant", avatar="📘"):
                st.write(a)
        
        # Input section
        with st.form("main_input"):
            topic = st.text_input("Topic of interest", key="topic")
            author = st.text_input("Author to emulate", key="author")
            question = st.text_input("Your question", key="question")
            
            if st.form_submit_button("Ask AuthorMind"):
                if not topic or not author or not question:
                    st.error("Please fill all fields")
                else:
                    process_question(topic, author, question)

    with col2:
        # About section
        with st.expander("About", expanded=True):
            st.markdown("""
            **AuthorMind AI** mimics authors' thinking patterns using:
            - YouTube interviews/lectures
            - Recent articles/interviews
            - AI-powered style emulation
            """)
        
        # Features section
        with st.expander("Technical Features"):
            st.markdown("""
            - **LLM**: GPT-3.5 Turbo
            - **Vector DB**: FAISS
            - **Search**: YouTube + DuckDuckGo
            - **Memory**: Last 5 exchanges
            """)
        
        # Feedback section
        with st.form("feedback"):
            feedback = st.radio("Rate response", ["👍", "👎"])
            comment = st.text_area("Comments")
            if st.form_submit_button("Submit Feedback"):
                submit_feedback(feedback, comment)

def process_question(topic, author, question):
    # Initialize workflow if needed
    if not st.session_state.workflow:
        st.session_state.workflow = FinancialWorkflow()
    
    # Create initial state
    state = {
        "topic": topic,
        "author": author,
        "question": question,
        "chat_history": st.session_state.chat_history
    }
    
    # Run conversation
    with st.spinner(f"🧠 Thinking like {author}..."):
        new_state = run_conversation(state)
    
    # Update chat history
    st.session_state.chat_history = new_state["chat_history"][-5:]
    st.rerun()

def submit_feedback(feedback, comment):
    sheet = connect_to_sheet()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([
        timestamp,
        st.session_state.get('topic', ''),
        st.session_state.get('author', ''),
        feedback,
        comment
    ])
    st.success("Feedback recorded!")

if __name__ == "__main__":
    main()
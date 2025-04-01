# streamlit_app.py
import streamlit as st
from workflow import AuthorWorkflow, State
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Initialize session state
if 'workflow' not in st.session_state:
    st.session_state.workflow = AuthorWorkflow()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Configure Google Sheets
def connect_to_sheet():
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    client = gspread.authorize(creds)
    return client.open("AuthorMind_Feedback").sheet1

def validate_sources(response, sources):
    """Checks which sources were actually referenced in the response"""
    highlights = []
    for source in sources.split("\n"):
        if source.lower() in response.lower():
            highlights.append(f"✅ {source}")
        else:
            highlights.append(f"⚠️ {source} (not directly referenced)")
    return "\n".join(highlights)

# App layout
st.set_page_config(layout="wide", page_title="AuthorMind AI", page_icon="📚")

# Title Section
st.title("📚 Author's Mind AI")
st.markdown("### Talk to Your Favorite Author")

# Configuration inputs at the very top
col1, col2 = st.columns(2)
with col1:
    topic = st.text_input("Topic of interest", key="topic_input")
with col2:
    author = st.text_input("Author to emulate", key="author_input")

# Main columns
main_col, side_col = st.columns([3, 1])

with main_col:
    # Chat container
    chat_container = st.container(height=500)
    
    # Display chat history with error handling
    for entry in st.session_state.chat_history:
        try:
            q, a, sources = entry
        except ValueError:
            # Handle legacy entries
            q, a = entry
            sources = "No sources available"
            
        with chat_container.chat_message("user"):
            st.write(q)
        with chat_container.chat_message("assistant", avatar="📘"):
            st.write(a)
            if 'debug_mode' in st.session_state and st.session_state.debug_mode:
                with st.expander("📚 Sources & Validation"):
                    st.markdown(f"**Reference materials:**\n{sources}")
                    validation = validate_sources(a, sources)
                    st.markdown("**Source Validation:**\n" + validation)

    # Question input at the bottom
    with st.form("question_form"):
        question = st.text_input("Your question", key="question_input")
        submitted = st.form_submit_button("Ask Author's Mind")

with side_col:
    # Debug mode toggle
    st.session_state.debug_mode = st.checkbox("🔧 Debug Mode", key="dbg_mode")
    
    # About section
    with st.expander("**About Author's Mind AI**", expanded=True):
        st.markdown("""
        Ever finished a book or podcast bursting with questions the author never addressed? 
        AuthorMind AI lets you converse directly with an author's digital clone!""")
    
    # Features section
    with st.expander("**Technical Features**"):
        st.markdown("""
        - **LLM**: mistral-saba-24b
        - **Embedding**:text-embedding-3-small
        - **Vector DB**: FAISS
        - **Search**: YouTube + DuckDuckGo""")
    
    # Feedback section
    with st.form("feedback"):
        feedback = st.radio("Rate response", ["👍", "👎"])
        comment = st.text_area("Comments")
        if st.form_submit_button("Submit Feedback"):
            sheet = connect_to_sheet()
            sheet.append_row([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                topic,
                author,
                feedback,
                comment
            ])
            st.success("Feedback recorded!")

    # Attribution
    st.markdown("""
    <div style='text-align: right; margin-top: 20px;'>
        Created by <a href="https://www.linkedin.com/in/balachandar-ravichandran-0918211b/">Balachandar Ravichandran</a>
    </div>
    """, unsafe_allow_html=True)

# Process questions
if submitted and topic and author and question:
    # Create initial state with sources
    state = {
        "topic": topic,
        "author": author,
        "question": question,
        "chat_history": st.session_state.chat_history,
        "context_sources": "",
        "objective_check": False,
        "response_summary": "",
        "validate_response": "",
        "fact_correction": "",
        "generate_final_response": "",
        "context_sources": ""
    }
    
    # Process through workflow
    with st.spinner(f"🧠 Analyzing {author}'s perspective..."):
        processed_state = st.session_state.workflow.process(state)
    
    # Update chat history with sources
    sources = processed_state.get("context_sources", "No sources available")
    st.session_state.chat_history.append(
        (question, 
         processed_state["generate_final_response"], 
         processed_state["context_sources"]))
    
    st.rerun()
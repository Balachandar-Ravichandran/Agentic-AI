# streamlit_app.py
import streamlit as st
from workflow import AuthorWorkflow, State
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os

# Initialize session state
if 'workflow' not in st.session_state:
    st.session_state.workflow = AuthorWorkflow()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

SPREADSHEET_ID = st.secrets["SPREADSHEET_ID"]

# Configure Google Sheets
def connect_to_sheet():
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    client = gspread.authorize(creds)
    return client.open_by_key(SPREADSHEET_ID).sheet1

# App layout
st.set_page_config(layout="wide", page_title="AuthorMind AI", page_icon="📚")

# Title Section
st.title("📚 Author's Mind AI")
st.markdown("### Talk to Your Favorite Author")

# Top Inputs
col1, col2 = st.columns(2)
with col1:
    topic = st.text_input("Topic of interest", placeholder="e.g. Real estate investing, Personal finance, Leadership",key="topic_input")
with col2:
    author = st.text_input("Author to emulate",placeholder="e.g. Robert Kiyosaki, Brené Brown, Malcolm Gladwell", key="author_input")

# Main columns
main_col, side_col = st.columns([3, 1])

with main_col:
    # Chat container
    chat_container = st.container(height=600)
    
    # Display chat history with error handling
    for q, a in st.session_state.chat_history:
            with chat_container.chat_message("user"):
                st.write(q)
            with chat_container.chat_message("assistant", avatar="📘"):
                st.write(a)

    # Question input at the bottom
    with st.form("question_form"):
        question = st.text_input("Your question", key="question_input")
        submitted = st.form_submit_button("Ask Author's Mind")
    
    # Add processing spinner right below the form
    if submitted and topic and author and question:
        
        try:
            sheet = connect_to_sheet()
            sheet.append_row([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                topic,
                author,
                question,
                "NO_FEEDBACK",  # Default value
                "NO_COMMENT"   # Default value
            ])
        except Exception as e:
            st.error(f"Error logging question: {str(e)}")

        # Create initial state
        state = {
            "topic": topic,
            "author": author,
            "question": question,
            "chat_history": st.session_state.chat_history,
            "objective_check": False,
            "response_summary": "",
            "validate_response": "",
            "fact_correction": "",
            "generate_final_response": ""
        }
        
        # Process through workflow - now directly below the button
        with st.spinner(f"🧠 Analyzing {author}'s perspective..."):
            processed_state = st.session_state.workflow.process(state)
        
        # Update chat history
        st.session_state.chat_history = processed_state["chat_history"]
        st.rerun()


with side_col:
    # About section
    with st.expander("**About Author's Mind AI**", expanded=True):
        st.markdown("""
        Ever finished a book or podcast bursting with questions the author never addressed?  
        **Author's Mind AI** lets you have a conversation with a digital clone of the author!  

        Imagine asking **Robert Kiyosaki** about **Investments**:  
        *"Is a luxury painting a true asset in today’s economy?"*  
        *"Would you buy a house as an investment right now?"*  

        **Author's Mind AI** analyzes an author’s latest interviews, books, and articles to generate responses in their signature style—blending  
        their historical wisdom with real-time insights. Get nuanced answers tailored to your unique scenarios,  
        as if the author had written a personalized chapter just for you.  

        **How Author's Mind AI Thinks**:  
        - YouTube interviews & lectures  
        - Recent articles & interviews sourced via DuckDuckGo Search  
        """)

    # Features section
    with st.expander("**Technical Details**"):
        st.markdown("""
        -**LLM**: mistral-saba-24b
        - **Embedding**:text-embedding-3-small
        - **Vector DB**: FAISS
        - **Search**: YouTube + DuckDuckGo
        - **Memory**: Last 5 exchanges
        - **Feedback**: Googlesheet
        """)

    # Feedback section
    with st.form("feedback"):
        feedback = st.radio("Rate Response Quality", ["👍", "👎"])
        comment = st.text_area("Optional Comments")
        if st.form_submit_button("Submit Feedback"):
            sheet = connect_to_sheet()
            sheet.append_row([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                topic,
                author,
                question,
                feedback,
                comment
            ])
            st.success("Thank you for your feedback!")

# Attribution
st.markdown("""
<div style='text-align: right; margin-top: 20px; color: #666;'>
    Created by <a href="https://www.linkedin.com/in/balachandar-ravichandran-0918211b/" style='color: #666;'>Balachandar Ravichandran</a>
</div>
""", unsafe_allow_html=True)
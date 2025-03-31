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

# App layout
st.set_page_config(layout="wide", page_title="AuthorMind AI", page_icon="📚")

# Title Section
st.title("📚 Author's Mind AI")
st.markdown("### Talk to Your Favorite Author")


# Main columns
main_col, side_col = st.columns([3, 1])

with main_col:
    # Chat container
    chat_container = st.container(height=600)
    
    # Display chat history
    for q, a in st.session_state.chat_history:
        with chat_container.chat_message("user"):
            st.write(q)
        with chat_container.chat_message("assistant", avatar="📘"):
            st.write(a)
    
    # Input form
    with st.form("main_input"):
        topic = st.text_input("Topic of interest")
        author = st.text_input("Author to emulate")
        question = st.text_input("Your question")
        
        submitted = st.form_submit_button("Ask Author's Perspective")

with side_col:
    # About section
    with st.expander("**About Author's Mind AI**", expanded=True):
        st.markdown("""
        Ever finished a book or podcast bursting with questions the author never addressed? 
        AuthorMind AI lets you converse directly with an author's digital clone! 
        
        Imagine asking **Robert Kiyosaki**:  
        *"Is a luxury painting a true asset in this economy?"*  
        *"Would you buy a house today as an investment?"*  
        
        Author's Mind AI analyzes recent author’s interviews, books, and articles to deliver responses in their signature style—combining 
        their historical wisdom with real-time data. Get nuanced answers to your niche scenarios, 
        as if the author crafted a personalized chapter just for your situation.               
                         
        **Author's Mind AI** thinking patterns using:         
        - YouTube interviews/lectures
        - Recent articles/interviews from DuckDuckGoSearch
        """)
    
    # Features section
    with st.expander("**Technical Features**"):
        st.markdown("""
        - **LLM**: GPT-3.5 Turbo
        - **Embedding**:text-embedding-3-small
        - **Vector DB**: FAISS
        - **Search**: YouTube + DuckDuckGo
        - **Memory**: Last 5 exchanges
        """)
    
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

 # Attribution with LinkedIn link
    st.markdown("""
    <div style='text-align: right; margin-top: 20px;'>
        Created by <a href="https://www.linkedin.com/in/balachandar-ravichandran-0918211b/">Balachandar Ravichandran</a>
    </div>
    """, unsafe_allow_html=True)

# Process questions
if submitted and topic and author and question:
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
    
    # Process through workflow
    with st.spinner(f"🧠 Analyzing {author}'s perspective..."):
        processed_state = st.session_state.workflow.process(state)
    
    # Update chat history
    st.session_state.chat_history = processed_state["chat_history"]
    st.rerun()
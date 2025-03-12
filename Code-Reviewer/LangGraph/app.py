import streamlit as st
from Code_Reviewer_End_to_End import CodeReviewWorkflow
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# Streamlit UI setup
st.set_page_config(page_title="CodeGuardian AI", layout="wide")
st.markdown("""
<style>
    .big-title { text-align: center; font-size: 2.5em !important; font-weight: bold; }
    .sidebar .sidebar-content { background-color: #f0f2f6; }
    .stButton>button { width: 100%; background-color: #4CAF50 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'results' not in st.session_state:
    st.session_state.results = None

# Sidebar configuration
with st.sidebar:
    st.markdown("## 🔧 Settings")
    llm_provider = st.selectbox("LLM Provider", ["Groq", "OpenAI"], index=0)
    
    if llm_provider == "Groq":
        model_name = st.selectbox("Model", ["mixtral-8x7b-32768", "llama2-70b-4096","qwen-2.5-32b"], index=0)
        api_key = st.text_input("Groq API Key", type="password")
    else:
        model_name = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4","gpt-4o"], index=0)
        api_key = st.text_input("OpenAI API Key", type="password")

# Main interface
st.markdown('<p class="big-title">🔍 CodeGuardian AI Review</p>', unsafe_allow_html=True)

with st.form("code_review_form"):
    col1, col2 = st.columns(2)
    with col1:
        objective = st.text_area("Code Objective", height=100)
        review_types = st.text_input("Review Types (comma-separated)")
    with col2:
        code = st.text_area("Paste Your Code", height=300)
    submitted = st.form_submit_button("🚀 Start Review")

if submitted and api_key:
    try:
        # Initialize LLM
        if llm_provider == "Groq":
            llm = ChatGroq(temperature=0.1, model_name=model_name, groq_api_key=api_key)
        else:
            llm = ChatOpenAI(temperature=0.1, model_name=model_name, openai_api_key=api_key)
        
        # Create and run workflow
        workflow = CodeReviewWorkflow(llm).compile()
        results = workflow.invoke({
            "input_objective": objective,
            "input_code": code,
            "input_instructions": [t.strip() for t in review_types.split(",")]
        })
        
        st.session_state.results = {
            "feedback": results.get("feedback_Collector", ""),
            "summary": results.get("final_summary", "")
        }
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Display results
if st.session_state.results:
    st.markdown("## 📝 Review Results")
    with st.expander("Detailed Feedback"):
        st.markdown(st.session_state.results["feedback"])
    with st.expander("Executive Summary"):
        st.markdown(st.session_state.results["summary"])
    st.download_button(
        "📥 Download Report",
        st.session_state.results["summary"],
        file_name="code_review.md"
    )
import os
import streamlit as st
from dotenv import load_dotenv
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Initialize YouTube API
youtube = build('youtube', 'v3', developerKey=os.getenv("YOUTUBE_API_KEY"))

# Initialize OpenAI Embedder
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# ---- Sidebar (Provider + Model Selection) ----
with st.sidebar:
    st.header("Settings")
    llm_provider = st.selectbox("Select LLM Provider", ["OpenAI", "Groq"])
    
    if llm_provider == "Groq":
        model_name = st.selectbox("Model", ["mixtral-8x7b-32768", "llama2-70b-4096", "qwen-2.5-32b"])
        api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY"))
    else:
        model_name = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4o"])
        api_key = st.text_input("OpenAI API Key", type="password")

# Initialize LLM
if llm_provider == "Groq":
    os.environ["GROQ_API_KEY"] = api_key
    from langchain_groq import ChatGroq
    llm = ChatGroq(model_name=model_name, temperature=0.7)  # ✅ Fixed parameter name
else:
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(model=model_name, temperature=0.7)

# ---- Centered Title ----
st.markdown(
    "<h1 style='text-align: center; font-size: 40px; font-weight: bold;'>Author Perspective AI</h1>",
    unsafe_allow_html=True
)

# ---- Search Function ----
class SearchResult:
    def __init__(self, search_result):
        self.video_id = search_result['id']['videoId']
        self.title = search_result['snippet']['title']
        self.description = search_result['snippet']['description']
        self.thumbnails = search_result['snippet']['thumbnails']['default']['url']
        self.transcript = self._get_transcript()

    def _get_transcript(self):
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(self.video_id)
            return " ".join([item['text'] for item in transcript_list])
        except Exception as e:
            st.warning(f"Error getting transcript for {self.video_id}: {e}")
            return ""

class SearchResponse:
    def __init__(self, search_response):
        self.prev_page_token = search_response.get('prevPageToken')
        self.next_page_token = search_response.get('nextPageToken')
        self.search_results = [SearchResult(item) for item in search_response.get('items', [])]

def search_yt(query, max_results=5, page_token=None):
    request = youtube.search().list(
        part="snippet",
        maxResults=max_results,
        pageToken=page_token,
        q=query,
        videoCaption='closedCaption',
        type='video',
    )
    return SearchResponse(request.execute())

# ---- FAISS Indexing ----
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def create_faiss_index(search_response):
    documents = []
    for result in search_response.search_results:
        # ✅ Add validation for non-empty transcripts
        if not result.transcript:
            continue
            
        metadata = {
            "video_id": result.video_id,
            "title": result.title
        }
        full_text = f"Title: {result.title}\nTranscript: {result.transcript}"
        
        # ✅ Handle empty text edge case
        if len(full_text.strip()) == 0:
            continue
            
        chunks = text_splitter.split_text(full_text)
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata=metadata))
    
    # ✅ Handle empty documents case
    if not documents:
        st.error("No valid transcripts found. Try different search terms.")
        return None
    
    return FAISS.from_documents(documents, embedder)

# Initialize FAISS index per session
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

# ---- Get User Input ----
topic = st.text_input("Enter the Topic to be discussed")
author = st.text_input("Enter the Author Name")

if st.button("Search"):
    if topic and author:
        with st.spinner("Searching YouTube..."):
            try:
                search_response = search_yt(f"{topic} {author}", max_results=3)
                if not search_response.search_results:
                    st.warning("No videos found. Try different search terms.")
                    st.session_state.faiss_index = None
                else:
                    st.session_state.faiss_index = create_faiss_index(search_response)
                    if st.session_state.faiss_index:  # ✅ Added null check
                        st.success(f"Index created with {len(st.session_state.faiss_index.index_to_docstore_id)} chunks!")
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
    else:
        st.warning("Please provide both topic and author")

# ---- RAG Pipeline ----
template = """
Act as an expert analyst of {author}'s work. Follow these rules:

1. Base answers ONLY on provided context from verified sources.
2. Match {author}'s distinctive style:
   - Philosophical tone
   - Historical framework for modern issues
3. If unable to answer, say "I need more sources."

Context provided:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# ---- Query Handling ----
query = st.text_input("Enter your question about the topic")  # ✅ Changed to text_input for better UX

if st.button("Get Answer"):
    if query and topic and author:
        if st.session_state.faiss_index and st.session_state.faiss_index.index.ntotal > 0:
            with st.spinner("Generating response..."):
                try:
                    rag_chain = (
                        {
                            "context": st.session_state.faiss_index.as_retriever(),
                            "question": RunnablePassthrough(),
                            "author": lambda _: author
                        }
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    response = rag_chain.invoke(query)
                    
                    st.subheader(f"Answer in {author}'s style:")
                    st.write(response)

                    # Display source reference
                    retrieved_docs = st.session_state.faiss_index.similarity_search(query)
                    sources = {doc.metadata["video_id"] for doc in retrieved_docs}  # ✅ Use set to avoid duplicates
                    st.write("\n**Source(s):**")
                    for src in sources:
                        st.markdown(f"- [{src}](https://www.youtube.com/watch?v={src})")
                        
                except Exception as e:
                    st.error(f"Error generating response: {e}")
        else:
            st.warning("FAISS index is empty. Possible reasons: \n1. No valid transcripts found \n2. Search not performed \n3. API limits exceeded")
    else:
        st.warning("Please fill all fields before submitting.")
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
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
# from sentence_transformers import SentenceTransformer
# from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Initialize APIs
youtube = build('youtube', 'v3', developerKey=os.getenv("YOUTUBE_API_KEY"))
embedder = OpenAIEmbeddings(model="text-embedding-3-small")
# embedder = HuggingFaceEmbeddings(
#     api_key=os.getenv("HF_TOKEN"),  # Add to .env as HF_API_KEY=your_token
#     model_name="sentence-transformers/all-MiniLM-L6-v2"  # Free tier default
# )
#model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",)
llm = ChatGroq(
    model_name="mixtral-8x7b-32768",
    temperature=0.7,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ---- UI Setup ----
st.markdown(
    "<h1 style='text-align: center; font-size: 40px; font-weight: bold;'>AuthorMind AI - Think Like an Author</h1>",
    unsafe_allow_html=True
)

# Split search inputs
col1, col2 = st.columns(2)
with col1:
    topic = st.text_input("Enter Topic (e.g., 'Artificial Intelligence')")
with col2:
    author = st.text_input("Enter Author Name (e.g., 'Yuval Noah Harari')")

# ---- Search Function ----
class SearchResult:
    def __init__(self, search_result):
        self.video_id = search_result['id']['videoId']
        self.title = search_result['snippet']['title']
        self.transcript = self._get_transcript()

    def _get_transcript(self):
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(self.video_id)
            return " ".join([item['text'] for item in transcript_list])
        except:
            return ""

def search_yt(query, max_results=3):
    request = youtube.search().list(
        part="snippet",
        maxResults=max_results,
        q=query,
        videoCaption='closedCaption',
        type='video',
    )
    return request.execute().get('items', [])

# ---- FAISS Indexing ----
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def create_faiss_index(items):
    documents = []
    for item in items:
        result = SearchResult(item)
        if not result.transcript:
            continue
            
        full_text = f"Title: {result.title}\nTranscript: {result.transcript}"
        chunks = text_splitter.split_text(full_text)
        
        for chunk in chunks:
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "video_id": result.video_id,
                    "title": result.title
                }
            ))
    
    return FAISS.from_documents(documents, embedder) if documents else None

# ---- App Flow ----
if st.button("Analyze Content"):
    if topic and author:
        with st.spinner(f"Searching for {author}'s views on {topic}..."):
            try:
                items = search_yt(f"{topic} {author}")
                if not items:
                    st.warning("No videos found with transcripts. Try different terms.")
                    st.session_state.faiss_index = None
                else:
                    st.session_state.faiss_index = create_faiss_index(items)
                    if st.session_state.faiss_index:
                        st.success("Content analyzed! Now ask your question below")
            except Exception as e:
                st.error(f"Error processing content: {str(e)}")
    else:
        st.warning("Please fill both topic and author fields")

# Only show question input if index exists
if 'faiss_index' in st.session_state and st.session_state.faiss_index:
    # ---- RAG Pipeline ----
    template = """
    Analyze and respond as {author} would. Rules:
    1. Use ONLY provided context
    2. Maintain {author}'s philosophical/historical style
    3. If context is insufficient, state "I need more information"
    
    Context: {context}
    
    Question: {question}
    
    {author}'s analysis:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": st.session_state.faiss_index.as_retriever(), 
         "question": RunnablePassthrough(),
         "author": lambda _: author}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # ---- Question Handling ----
    question = st.text_input(f"What would you like to ask about {author}'s perspective on {topic}?")
    
    if question:
        with st.spinner(f"Channeling {author}'s thinking..."):
            try:
                response = rag_chain.invoke(question)
                st.subheader(f"{author}'s Perspective")
                st.write(response)
                
                # Show sources (fixed unhashable error)
                docs = st.session_state.faiss_index.similarity_search(question)
                seen = set()
                st.markdown("**Reference Videos:**")
                for doc in docs:
                    vid = doc.metadata["video_id"]
                    if vid not in seen:
                        seen.add(vid)
                        st.markdown(f"- [{doc.metadata['title']}](https://youtu.be/{vid})")
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
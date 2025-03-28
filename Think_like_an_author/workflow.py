# workflow.py
from typing import TypedDict, Optional, List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from datetime import datetime
import os
import re

# Initialize APIs
embedder = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
youtube = build('youtube', 'v3', developerKey=os.getenv("YOUTUBE_API_KEY"))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

class State(TypedDict):
    topic: str
    author: str
    question: str
    chat_history: list
    generate_final_response: str

class AuthorWorkflow:
    def __init__(self):
        self.combined_index: Optional[FAISS] = None
        self.current_topic: str = ""
        self.current_author: str = ""

    def _search_yt(self, query: str, max_results: int = 3):
        try:
            request = youtube.search().list(
                part="snippet",
                maxResults=max_results,
                q=query,
                videoCaption='closedCaption',
                type='video',
            )
            return request.execute().get('items', [])
        except Exception as e:
            return []

    def _process_ddg_results(self, results: str, topic: str, author: str) -> List[Document]:
        try:
            return [
                Document(
                    page_content=result.strip(),
                    metadata={
                        "source": "web",
                        "search_query": f"{topic} {author}",
                        "timestamp": datetime.now().isoformat()
                    }
                ) for result in results.split("\n\n") if result.strip()
            ]
        except Exception:
            return []

    def build_index(self, topic: str, author: str):
        if self.combined_index and self.current_topic == topic and self.current_author == author:
            return
        
        self.current_topic = topic
        self.current_author = author
        
        # YouTube processing
        yt_items = self._search_yt(f"{topic} {author}")
        yt_docs = self._process_youtube_items(yt_items)
        
        # Web processing
        search = DuckDuckGoSearchRun()
        ddg_results = search.invoke(f"{topic} {author} recent articles/interviews")
        ddg_docs = self._process_ddg_results(ddg_results, topic, author)
        
        # Combine documents
        all_docs = yt_docs + ddg_docs
        if all_docs:
            self.combined_index = FAISS.from_documents(all_docs, embedder)

    def _process_youtube_items(self, items):
        documents = []
        for item in items:
            video_id = item['id']['videoId']
            title = item['snippet']['title']
            try:
                transcript = YouTubeTranscriptApi.get_transcript(
                    video_id, languages=['en', 'en-US']
                )
                full_text = f"{title}\n{' '.join([t['text'] for t in transcript])"
                documents.extend([
                    Document(
                        page_content=chunk,
                        metadata={"source": "youtube", "title": title}
                    ) for chunk in text_splitter.split_text(full_text)
                ])
            except Exception:
                continue
        return documents

    def generate_response(self, question: str, chat_history: list) -> str:
        if not self.combined_index:
            return "System not initialized properly"
        
        history_str = "\n".join([f"User: {q}\nAuthor: {a}" for q, a in chat_history[-3:]])
        
        rag_chain = (
            {"context": self.combined_index.as_retriever(search_kwargs={"k": 5}),
            "question": RunnablePassthrough(),
            "author": lambda _: self.current_author,
            "topic": lambda _: self.current_topic,
            "history": lambda _: history_str}
            | ChatPromptTemplate.from_template("""As {author}'s clone analyzing {topic}:
            
            Previous discussion:
            {history}
            
            Context: {context}
            Question: {question}
            
            Respond in {author}'s style with practical advice:""")
            | llm
            | StrOutputParser()
        )
        
        return rag_chain.invoke(question)

def run_conversation(workflow: AuthorWorkflow, user_input: str, topic: str, author: str, chat_history: list) -> str:
    # Initialize index
    workflow.build_index(topic, author)
    
    # Generate response
    response = workflow.generate_response(user_input, chat_history)
    
    # Format author-style response
    return f"As {author}, I would say:\n\n{response}"
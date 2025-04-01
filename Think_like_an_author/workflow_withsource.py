# workflow.py
from typing import TypedDict, Optional, List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, StateGraph,END
from datetime import datetime
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
import os
import re

# Initialize APIs
embedder = OpenAIEmbeddings(model="text-embedding-3-small")
#llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

llm = ChatGroq(
    model_name="mistral-saba-24b",
    temperature=0.7,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

youtube = build('youtube', 'v3', developerKey=os.getenv("YOUTUBE_API_KEY"))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

class State(TypedDict):
    topic: str
    author: str
    question: str
    objective_check: bool
    response_summary: str
    validate_response: str
    fact_correction: str
    generate_final_response: str
    chat_history: list
    context_sources: str

class AuthorWorkflow:
    def __init__(self):
        self.combined_index: Optional[FAISS] = None
        self.current_topic: str = ""
        self.current_author: str = ""
        self.workflow_graph = None

    class SearchResult:
        def __init__(self, search_result):
            self.video_id = search_result['id']['videoId']
            self.title = search_result['snippet']['title']
            self.transcript = self._get_transcript()

        def _get_transcript(self):
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(
                    self.video_id,
                    languages=['en', 'en-US', 'en-GB', 'en-CA']
                )
                return " ".join([item['text'] for item in transcript_list])
            except Exception as e:
                return ""

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
        except Exception:
            return []

    def _process_ddg_results(self, results: str, state: State) -> List[Document]:
        try:
            return [
                Document(
                    page_content=result.strip(),
                    metadata={
                        "source": "web",
                        "search_query": f"{state['topic']} {state['author']}",
                        "timestamp": datetime.now().isoformat()
                    }
                ) for result in results.split("\n\n") if result.strip()
            ]
        except Exception:
            return []

    def _process_youtube_items(self, items):
        documents = []
        for item in items:
            result = self.SearchResult(item)
            if result.transcript:
                full_text = f"Video: {result.title}\nTranscript: {result.transcript}"
                chunks = text_splitter.split_text(full_text)
                documents.extend([
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": "youtube",
                            "video_id": result.video_id,
                            "title": result.title
                        }
                    ) for chunk in chunks
                ])
        return documents

    def build_combined_index(self, state: State) -> State:
        try:
            if (self.combined_index and 
                self.current_topic != state["topic"] and 
                self.current_author != state["author"]):
                self.combined_index = None
                
            self.current_topic = state["topic"]
            self.current_author = state["author"]

            if not self.combined_index:
                # YouTube processing
                print(f"\n=== Building Index for {state['topic']} ===")
                yt_items = self._search_yt(f"{state['topic']} {state['author']}")
                yt_docs = self._process_youtube_items(yt_items)
                
                # DDG processing
                ddg_results = DuckDuckGoSearchRun().invoke(
                    f"{state['topic']} {state['author']} recent articles/interviews"
                )
                ddg_docs = self._process_ddg_results(ddg_results, state)
                
                all_docs = yt_docs + ddg_docs
                if all_docs:
                    self.combined_index = FAISS.from_documents(all_docs, embedder)
            
            return state
        except Exception as e:
            state["generate_final_response"] = f"Index build failed: {str(e)}"
            return state

    def generate_author_response(self, state: State) -> State:
        try:
            if not self.combined_index:
                raise ValueError("Index not initialized")
            
            print(f"\n=== Generating Response for: {state['question']} ===")

            # Get context documents with metadata
            retriever = self.combined_index.as_retriever(search_kwargs={"k": 5})
            docs = retriever.get_relevant_documents(state['question'])
            
            # Store raw sources for verification
            context_content = "\n\n".join([doc.page_content for doc in docs])
            state["context_sources"] = "\n".join([
                f"Source {i+1}: {doc.metadata.get('title', doc.metadata.get('source', 'Unknown'))}"
                for i, doc in enumerate(docs)
            ])


            history_str = "\n".join(
                [f"User: {q}\nAuthor: {a}" for q, a in state["chat_history"]]
            ) if state["chat_history"] else "No previous conversation"

            rag_template = """As {author}'s clone analyzing {topic}, respond to:
        
                **Question**: {question}

                **Conversation History**:
                {history}

                **Reference Materials**:
                {sources}

                **Response Requirements**:
                1. Open naturally without AI-like phrases
                2. Use {author}'s signature style (storytelling/analytical)
                3. Support claims with specific source references like:
                - "As I discussed in [Source 1]..."
                - "This aligns with my view in [Source 3]..."
                4. Acknowledge knowledge gaps clearly
                5. Keep under 400 words"""

            rag_chain = (
                { "context": lambda _: context_content,
                  "sources": lambda _: state["context_sources"],
                #"context": self.combined_index.as_retriever(search_kwargs={"k": 5}),
                 "question": RunnablePassthrough(),
                 "author": lambda _: state["author"],
                 "topic": lambda _: state["topic"],
                 "history": lambda _: history_str}
                | ChatPromptTemplate.from_template("""As {author}'s clone analyzing {topic}: consider:
                
                Chat History:
                {history}
                
                Context: {context}
                Question: {question}
                Reference Materials: {sources}
                
                Requirements:
                1. Use {author}'s signature style
                2. Reference historical and current context
                3. Provide actionable advice
                
                Analysis:""")
                | llm
                | StrOutputParser()
            )

            state["response_summary"] = rag_chain.invoke(state["question"])
            return state
        except Exception as e:
            state["response_summary"] = f"Error: {str(e)}"
            return state

    def validate_user_input(self, state: State) -> State:
        if not state.get("question", "").strip():
            return {**state, "objective_check": False}
        
        print(f"\n=== Validating user request {state['question']} and {state['topic']} ===")
        
        #validator_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        validator_llm = ChatGroq(
                model_name="mistral-saba-24b",
                temperature=0.7,
                groq_api_key=os.getenv("GROQ_API_KEY"))

        validation_prompt = f"""**Conversation Context Validation Task**

            [Author Profile]
            Name: {state['author']}
            Expertise: {state['topic']} 
            [Question]: {state['question']}
            
            [Current Question]
            {state['question']}

            
            [Validation Rules]
            1. PRIMARY RELEVANCE: Direct connection to {state['topic']} concepts
            2. CONTEXTUAL RELEVANCE: Follow-up to previous discussion 
            3. AUTHOR ALIGNMENT: Within {state['author']}'s expertise
            4. TANGENTIAL ALLOWANCE: Related financial/economic concepts
            5. REJECT: Unrelated subjects (romance, politics, etc.)

            [Decision Framework]
            - Accept if ANY of these apply:
            a) Directly references {state['topic']} concepts
            b) Follow-up to previous exchange
            c) Financial strategy question
            d) Economic trend analysis
            - Reject only if completely unrelated

            [Examples]
            Case 1: 
            History: Discussed asset classification
            Question: "How does this apply to rental properties?"
            Verdict: True (Contextual follow-up)

            Case 2:
            History: No previous conversation  
            Question: "Best crypto investments?"
            Verdict: False (Outside {state['author']}'s documented expertise)

            Case 3:
            History: Talked about market cycles
            Question: "How do interest rates affect this?"
            Verdict: True (Economic analysis)

            [Current Validation]
            Respond ONLY with 'True' or 'False'"""

        
        response = validator_llm.invoke(validation_prompt).content.strip().lower()
        is_valid = "true" in response
        print(is_valid)
        return {**state, "objective_check":is_valid}
    


    def validate_facts(self, state: State) -> State:
        
        print(f"\n=== Validating facts for '{state['question']}' ===")

        feedback_prompt = f"""**Validation Task: Summary Quality Check**

            [Author] {state['author']}
            [Topic] {state['topic']}
            [User Question] {state['question']}

            [Quality Criteria]
            1. Directly answers the specific question asked
            2. Uses {state['author']}'s signature communication style
            3. Contains concrete examples/data from source material
            4. Acknowledges limitations when information is missing

            [Summary to Validate]
            {state['response_summary']}

            [Validation Rules]
            - Respond "VALID" if all criteria are met
            - Respond "SEARCH FOR CONTENT: [Topic Area] - [Author] - [Specific Need]" if:
                * Missing key question aspects
                * Contains generic/non-specific information
                * Lacks author-style analysis

            [Examples]
            Good Response-> VALID
            Needs Improvement-> SEARCH FOR CONTENT: real estate investing - Robert Kiyosaki - 2024 market trends"""

        response = llm.invoke(feedback_prompt).content.strip()
        
        if "VALID" in response.upper():
            return {**state, "validate_response": "VALID"}
        else:
            search_terms = re.sub(r'[^a-zA-Z0-9\s\-,:]', '', response.split(":")[-1].strip())
            return {**state, "validate_response": f"SEARCH FOR: {search_terms}"}

    def consolidate_summary(self, state: State) -> State:
        try:
            print("consolidate_summary")
            print("validate_response", state["validate_response"])
            print("objective_check", state["objective_check"])
            
            # Handle failed objective check
            if not state.get("objective_check", True):
                state["generate_final_response"] = "❌ Question not relevant to the topic"
                # Still record the interaction in history
                state["chat_history"].append(
                    (state["question"], 
                     state["generate_final_response"],
                     state["context_sources"])                 
                )
                return state

            # Format chat history for context
            history_str = "\n".join(
                [f"User: {q}\nAuthor: {a}" for q, a in state.get("chat_history", [])]
            ) or "No previous conversation"

            # Determine context sources
            primary_analysis = state.get("response_summary", "")
            needs_verification = "VALID" not in state.get("validate_response", "").upper()
            external_data = state.get("fact_correction", "") if needs_verification else ""

            # Build context string conditionally
            context_sources = f"[Primary Analysis]\n{primary_analysis}"
            if external_data and "No additional information" not in external_data:
                context_sources += f"\n\n[External Verification]\n{external_data}"

            # Create dynamic prompt with history
            prompt = f"""**Role**: You are {state['author']}, responding in their distinct voice to address "{state['topic']}".

            **Objective**: Speak as if {state['author']} is having a personal conversation, using their authentic tone and mannerisms.

            **Context**:
            - Previous discussion: {history_str}
            - Current question: "{state['question']}"
            - Reference materials: {context_sources}

            **Response Style Rules**:
            1. *Natural Openings**: Begin conversationally without formulaic phrases. Examples:
                - *"This brings to mind an important principle..."*  
                - *"Consider how this relates to..."*  
                - *"There's an interesting parallel here..."*  
                (Vary based on context - no required starter phrases)
            2. **Authentic Voice**: Maintain {state['author']}'s signature:
                - Communication style (storytelling/analytical/philosophical)  
                - Characteristic tone and cadence  
                - Core beliefs and values 
            3. **Organic Flow**:
                - Start with what feels most natural to the conversation  
                - Support arguments with relevant examples (current/historical/personal)  
                - Address potential misunderstandings through dialogue  
                - Conclude with actionable wisdom 
            4. Modern context: {f"Weave in recent developments" if external_data else "Focus on timeless principles"}
            5. Keep it human:
            - Vary sentence structure like spontaneous speech
            - Use rhetorical questions when appropriate
            - Allow natural transitions between ideas
            - Limit to 3-4 concise paragraphs (under 400 words)
            """
            
            # Generate and store final response
            state["generate_final_response"] = llm.invoke(prompt).content
            
            # Update chat history with current interaction
            state["chat_history"].append(
                (state["question"], state["generate_final_response"])
            )
            
            # Optional: Trim history to last N interactions
            max_history = 5  # Keep last 5 exchanges
            state["chat_history"] = state["chat_history"][-max_history:]
            
            return state

        except Exception as e:
            print(f"Consolidation error: {str(e)}")
            state["generate_final_response"] = "Error generating final analysis"
            # Record error in history
            state["chat_history"].append(
                (state["question"], "System Error: Could not generate response")
            )
            return state

    def initialize_workflow(self):
        builder = StateGraph(State)
        
        builder.add_node("get_topic", lambda state: state)
        builder.add_node("build_index", self.build_combined_index)
        builder.add_node("validate_input", self.validate_user_input)
        builder.add_node("generate_response", self.generate_author_response)
        builder.add_node("validate_facts", self.validate_facts)
        builder.add_node("finalize_response", self.consolidate_summary)

        builder.set_entry_point("get_topic")
        builder.add_edge("get_topic", "build_index")
        builder.add_edge("build_index", "validate_input")
        
        builder.add_conditional_edges(
            "validate_input",
            lambda state: "Accepted" if state["objective_check"] else "Rejected",
            {"Accepted": "generate_response", "Rejected": "finalize_response"}
        )
        
        builder.add_edge("generate_response", "validate_facts")
        builder.add_edge("validate_facts", "finalize_response")
        builder.add_edge("finalize_response", END)

        self.workflow_graph = builder.compile()

    def process(self, state: State) -> State:
        if not self.workflow_graph:
            self.initialize_workflow()
        return self.workflow_graph.invoke(state)
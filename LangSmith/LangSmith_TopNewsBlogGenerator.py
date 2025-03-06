import os
from dotenv import load_dotenv
load_dotenv()


os.environ["xai_api_key"]=os.getenv("X_AI")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"


from langsmith import traceable
from langchain_xai import ChatXAI
from langchain_groq import ChatGroq

from IPython.display import Image, display

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph,END
from langgraph.prebuilt import tools_condition, ToolNode

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing_extensions import TypedDict


#XAI for News
News_llm = ChatXAI(model="grok-2")

#OpenSource llm for Summary generator
#Grog for Summary generator & Validate the User input
llm=ChatGroq(model="qwen-2.5-32b")

# System message
sys_msg = SystemMessage(content="You News Blog/Article Generator assistance - help with Top 5 news for requested Country/Area")


# Graph state
class State(TypedDict, total=False):
    country: str
    news_column: str
    input_status: str
    news: str
    summary_news: str

#@traceable
## get user input
def get_user_input(state: State):
    country = input("Please enter the Country/Area for which you want to get the news: ")
    column_name = input("Please enter the news column (e.g., Politics, Sports, Business): ")
    print("Country/Area: ", country)
    print("News Column: ", column_name)
    return {"country": country, "news_column": column_name}


#@traceable
def get_news(state: State):
    """
    Get news from the News LLM
    """
    # Get the country from the state
    country = state["country"]
    # Get the column name from the state
    column = state["news_column"]
    # Get the news from the News LLM
    msg = News_llm.invoke(f"Get me Top 5 news for {country} news for {column},  check for factual and should not be baised/Sensitve news")
    # Return the news
    return {"news": msg.content}

#@traceable
def get_summary(state: State):
    msg = llm.invoke(f"Summarize the news and have the headings/ Topics along with summary: {state['news']}")
    return {"summary_news": msg.content}  # Use lowercase "summary_news"


#@traceable
def validate_user_input(state: State):
    """
    Check if the user input is valid
    """
    # Get the country & column from the state and validate it using Groq llm
    country = state["country"]
    column_name = state["news_column"]
    
    msg = llm.invoke(

        f"Validate if '{country}' is a real country and '{column_name}' is a valid news category. "
        f"Respond ONLY with 'True' (if valid) or 'False' (if invalid). Do not include any extra words."

    )
    print(msg.content)

    is_valid = msg.content.strip().lower() == "true"
    return {"input_status": is_valid}


#@traceable
def route_input(state: State):
    """Route back to joke generator or end based upon feedback from the evaluator"""

    if state["input_status"] == True:
        print("Accepted")
        return "Accepted"
    elif state["input_status"] == False:
        print("Rejected")
        return "Rejected"
    

# Graph

def default_graph():
    Newsbuilder = StateGraph(State)

    Newsbuilder.add_node("get_user_input", get_user_input)
    Newsbuilder.add_node("get_news", get_news)
    Newsbuilder.add_node("get_summary", get_summary)
    Newsbuilder.add_node("validate_user_input", validate_user_input)


    Newsbuilder.add_edge(START, "get_user_input")
    Newsbuilder.add_edge("get_user_input", "validate_user_input")
    #Newsbuilder.add_edge("validate_user_input", "get_news")
    Newsbuilder.add_edge("get_news", "get_summary")
    Newsbuilder.add_edge("get_summary", END)



    Newsbuilder.add_conditional_edges(
        "validate_user_input",
        route_input,
        {
            "Accepted": "get_news",
            "Rejected": "get_user_input"
        }
    )


    graph = Newsbuilder.compile()
    return graph



# from IPython.display import Image, display
# display(Image(graph.get_graph().draw_mermaid_png()))

graph = default_graph()
#state = graph.invoke({"country": "", "news_column": ""}) 





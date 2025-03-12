import os
from dotenv import load_dotenv
from typing import Annotated, List, TypedDict
from pydantic import BaseModel, Field
import operator
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# Schema definitions
class AgentTask(BaseModel):
    name: str = Field(description="Type of code review agent")
    description: str = Field(description="Review description")

class Reviews(BaseModel):
    reviews: List[AgentTask] = Field(description="List of review tasks")

class State(TypedDict, total=False):
    input_objective: str
    input_code: str
    input_instructions: List[str]
    objective: str
    code: str
    user_instructions_list: List[str]
    objective_check: bool
    agent_tasks: List[AgentTask]
    feedback_Generator: Annotated[List, operator.add]
    feedback_Collector: str
    final_summary: str

# Remove hardcoded LLM initialization and environment variables
class CodeReviewWorkflow:
    def __init__(self, llm):
        self.llm = llm
        self.planner = self.llm.with_structured_output(Reviews)
        self.workflow = StateGraph(State)
        self._setup_graph()

    def _setup_graph(self):
        # Add nodes
        self.workflow.add_node("get_input", self.get_user_input)
        self.workflow.add_node("validate", self.validate_objective)
        self.workflow.add_node("plan_tasks", self.master_planner)
        self.workflow.add_node("execute_task", self.execute_agent_review_task)
        self.workflow.add_node("synthesize", self.synthesize_feedback)
        self.workflow.add_node("generate_summary", self.generate_summary)

        # Set up edges
        self.workflow.set_entry_point("get_input")
        self.workflow.add_edge("get_input", "validate")
        
        # Conditional routing
        self.workflow.add_conditional_edges(
            "validate",
            self.route_based_on_validation,
            {"accepted": "plan_tasks", "rejected": "generate_summary"}
        )
        
        self.workflow.add_edge("plan_tasks", "execute_task")
        self.workflow.add_edge("execute_task", "synthesize")
        self.workflow.add_edge("synthesize", "generate_summary")
        self.workflow.add_edge("generate_summary", END)

    # Modified state management functions
    def get_user_input(self, state: State):
        # Get inputs from state instead of CLI
        state["objective"] = state["input_objective"]
        state["user_instructions_list"] = state["input_instructions"]
        state["code"] = state["input_code"]
        return state

    def validate_objective(self, state: State):
        response = self.llm.invoke(
            f"Verify if code objective matches implementation:\n"
            f"Objective: {state['objective']}\nCode:\n{state['code']}\n"
            "Respond ONLY with 'True' or 'False'"
        )
        state["objective_check"] = response.content.strip().lower() == "true"
        return state

    def route_based_on_validation(self, state: State):
        return "accepted" if state["objective_check"] else "rejected"

    def master_planner(self, state: State):
        result = self.planner.invoke([
            SystemMessage(content="Plan code review tasks based on user instructions."),
            HumanMessage(content=f"Review types: {state['user_instructions_list']}")
        ])
        state["agent_tasks"] = result.reviews
        return state

    def execute_agent_review_task(self, state: State):
        agent_feedback = []
        for task in state["agent_tasks"]:
            response = self.llm.invoke([
                SystemMessage(content=f"You are a {task.name} expert. Perform detailed code review."),
                HumanMessage(content=f"Task: {task.description}\nCode:\n{state['code']}")
            ])
            agent_feedback.append({"agent": task.name, "feedback": response.content})
        return {"feedback_Generator": agent_feedback}

    def synthesize_feedback(self, state: State):
        formatted = [
            f"=== {entry['agent'].upper()} REVIEW ===\n{entry['feedback']}\n{'-'*40}"
            for entry in state["feedback_Generator"]
        ]
        state["feedback_Collector"] = "\n\n".join(formatted)
        return state

    def generate_summary(self, state: State):
        if not state["objective_check"]:
            state["final_summary"] = "❌ Objective mismatch between code and stated goals"
            return state
        
        general_summary = self.llm.invoke(f"""
            Generate final code review summary:
            1. OVERALL ASSESSMENT
            2. KEY FINDINGS
            3. RECOMMENDATIONS
            
            Context:
            Objective: {state['objective']}
            Code:\n{state['code']}
            Feedback:\n{state['feedback_Collector']}
        """).content
        
        state["final_summary"] = (
            f"🛠️ AGENT FEEDBACK\n\n{state['feedback_Collector']}\n\n"
            f"🔍 SUMMARY\n\n{general_summary}"
        )
        return state

    def compile(self):
        return self.workflow.compile()


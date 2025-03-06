### News Generator using LangGraph , XAI and Groq

Overview
This project is a Top headlines news generation and summarization pipeline built using LangGraph. 
It takes user input (country and news category), validates it using an LLM, fetches the latest news, and generates a concise summary. The system ensures factual accuracy and unbiased reporting.

The workflow follows a state-based approach where:

User input is taken for country & news category.
Validation is performed using an LLM.
If valid, news articles are fetched using a different LLM.
Summarization is done to provide a structured output.
The process re-prompts users if the input is invalid.


Features
✅ LLM-Powered Validation – Ensures the input country & news category are real and valid.
✅ State-Based Execution – Uses LangGraph for structured workflow automation.
✅ News Retrieval – Fetches top 5 unbiased news articles for the given category.
✅ Summarization – Generates concise, readable news summaries.
✅ Error Handling – Automatically re-prompts users for incorrect input.
✅ LangSmith Integration – Enables tracing and debugging of the workflow.


Technologies Used
🛠 LangGraph – For managing state-based workflows.
🛠 LangSmith – For debugging and tracing execution.
🛠 ChatXAI – Used for news retrieval.
🛠 ChatGroq (Qwen-2.5-32b) – Used for input validation & summarization.

# Top 5 Headline News Generator using LangGraph , XAI and Groq


## Overview
This project is a **Headline news generation** built using **LangGraph**. It takes user input (country and news category), validates it using an **LLM**, fetches the latest news, and generates a concise summary. The system ensures factual accuracy and unbiased reporting.

The workflow follows a **state-based approach**:
1. **User Input:** The user provides a country and a news category.
2. **Validation:** The input is validated using an **LLM**.
3. **News Retrieval:** If valid, the system fetches the top 5 news articles.
4. **Summarization:** A concise news summary is generated.
5. **Error Handling:** If the input is invalid, the system prompts the user again.

---

## Features
✅ **LLM-Powered Validation** – Ensures that the provided country & news category are real and valid.  
✅ **State-Based Execution** – Uses **LangGraph** for structured workflow automation.  
✅ **News Retrieval** – Fetches **top 5 unbiased news articles** for the given category.  
✅ **Summarization** – Generates **concise, readable news summaries**.  
✅ **Error Handling** – Automatically re-prompts users for incorrect input.  
✅ **LangSmith Integration** – Enables **tracing and debugging** of the workflow.  

---

## Technologies Used
- **[LangGraph](https://github.com/langchain-ai/langgraph)** – For managing **state-based workflows**.  
- **[LangSmith](https://smith.langchain.com/)** – For debugging and tracing execution.  
- **[ChatXAI](https://x.ai/)** – Used for **news retrieval**.  
- **[ChatGroq](https://groq.com/)** – Used for **input validation & summarization**.  
- **[LangChain](https://python.langchain.com/)** – Core framework for **LLM interactions**.  


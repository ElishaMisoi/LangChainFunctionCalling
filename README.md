# LangChain Function Calling with Azure OpenAI GPT Model

This repository contains example code and resources for the blog post  
**[“When LLMs Stop Talking and Start Doing: A Guide to LangChain Function Calling with Python and Azure AI Foundry.”](https://medium.com/@elishamisoi/when-llms-stop-talking-and-start-doing-a-guide-to-langchain-function-calling-with-python-and-azure-f3ebc660dbf2)**

The project demonstrates how to use **LangChain** with an **Azure OpenAI GPT deployment** to extend LLMs beyond conversation and into **actionable tasks**.

## Features

- Python integration with **LangChain** and Azure OpenAI SDK
- Example **function calling demos**:
  - Fetching the latest news
  - Checking live weather
- Clear pattern to extend into **enterprise use cases** (e.g., HR portals, timesheet submissions, ERP workflows, IT helpdesk tasks)

## Getting Started

Full setup instructions are included in the blog post.

### Running the API locally

1. Create a virtual environment and install requirements (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill your Azure OpenAI values.

3. Start the server with uvicorn (from repo root):

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

4. Health check: GET http://127.0.0.1:8000/healthz

5. Chat endpoint example:

POST http://127.0.0.1:8000/chat
Content-Type: application/json

{
"input": "Hello",
"session_id": "default"
}

---

Running in VS Code

- Select the Python interpreter for the project (pick the `.venv` you created) from the bottom-right status bar or `Ctrl+Shift+P` -> `Python: Select Interpreter`.
- Open the Run view (left sidebar) and choose "Python: Uvicorn (app.main)" then press the green play button. This uses `.vscode/launch.json` and the project's `.env` file.

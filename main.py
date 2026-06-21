import os
import json
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from ddgs import DDGS
from dotenv import load_dotenv
import math
import uvicorn
from googlesearch import search
import time
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# -------------------------------------------------------------------
# 1. TOOLS (identical to your original script)
# -------------------------------------------------------------------
def search_web(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                snippets = [r.get("body", "") for r in results if r.get("body")]
                if snippets:
                    return "\n".join(snippets)
            return "No relevant search results found."
    except Exception as e:
        return f"Search error: {str(e)}"

# def search_web(query: str) -> str:
#     try:
#         # Use advanced=True to get title, url, description (snippets)
#         results = list(search(query, num_results=3, advanced=True))
#         if not results:
#             return "No relevant search results found."

#         snippets = [res.description for res in results if res.description]
#         print(snippets)
#         if snippets:
#             return "\n".join(snippets)
#         else:
#             return "No text snippets found in the search results."

#     except Exception as e:
#         return f"Search error: {str(e)}"

# def search_web(query: str) -> str:
#     logger.info(f"🔍 Searching for: {query}")
#     try:
#         results = list(search(query, num_results=3, advanced=True))
#         logger.info(f"✅ Got {len(results)} results")
#         if not results:
#             return "No relevant search results found."
#         snippets = [res.description for res in results if res.description]
#         logger.info(f"📄 Snippets: {snippets}")
#         if snippets:
#             return "\n".join(snippets)
#         else:
#             return "No text snippets found in the search results."
#     except Exception as e:
#         logger.error(f"❌ Search error: {e}")
#         return f"Search error: {str(e)}"
    

def calculate(expression: str) -> str:
    try:
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed_names.update({"abs": abs, "round": round})
        code = compile(expression, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                return f"Error: '{name}' is not allowed."
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"

def get_current_time() -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def save_note(content: str) -> str:
    try:
        with open("agent_notes.txt", "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {content}\n")
        return f"Note saved: '{content}'"
    except Exception as e:
        return f"Failed to save note: {str(e)}"

def read_notes() -> str:
    try:
        if not os.path.exists("agent_notes.txt"):
            return "No notes found."
        with open("agent_notes.txt", "r", encoding="utf-8") as f:
            content = f.read()
            return content if content.strip() else "No notes yet."
    except Exception as e:
        return f"Error reading notes: {str(e)}"

# -------------------------------------------------------------------
# 2. AGENT CLASS – USES GEMINI
# -------------------------------------------------------------------
class AdvancedAgent:
    def __init__(self, gemini_api_key: str, model: str = "gemini-2.5-flash"):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model)
        self.conversation_memory = []          # for short‑term memory
        self._tool_functions = {
            "search_web": search_web,
            "calculate": calculate,
            "get_current_time": get_current_time,
            "save_note": save_note,
            "read_notes": read_notes,
        }

    def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return Gemini‑compatible function declarations."""
        return [
            {
                "name": "search_web",
                "description": "Search the web for current information.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {"query": {"type": "STRING"}},
                    "required": ["query"],
                },
            },
            {
                "name": "calculate",
                "description": "Perform mathematical calculations.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {"expression": {"type": "STRING"}},
                    "required": ["expression"],
                },
            },
            {
                "name": "get_current_time",
                "description": "Get the current date and time.",
                "parameters": {"type": "OBJECT", "properties": {}},
            },
            {
                "name": "save_note",
                "description": "Save a note to long‑term memory.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {"content": {"type": "STRING"}},
                    "required": ["content"],
                },
            },
            {
                "name": "read_notes",
                "description": "Read all previously saved notes.",
                "parameters": {"type": "OBJECT", "properties": {}},
            },
        ]

    def _execute_tool(self, name: str, args: Dict) -> str:
        func = self._tool_functions.get(name)
        if not func:
            return f"Error: Unknown tool '{name}'"
        return func(**args)

    def _format_memory(self) -> str:
        if not self.conversation_memory:
            return "(No previous conversation)"
        lines = []
        for i, exchange in enumerate(self.conversation_memory[-6:], 1):
            lines.append(f"{i}. User: {exchange['user']}\n   Assistant: {exchange['assistant']}")
        return "\n".join(lines)

    def run(self, user_query: str) -> str:
        # Build the conversation history (system + last 6 exchanges)
        system_prompt = (
            "You are an English language tutor for Georgian speaking users. "
            "Always respond in English. Help the user learn English in a clear, friendly way. "
            "If the user writes in Georgian, understand it and respond in English – you can explain "
            "the meaning, provide translations, correct mistakes, or give simple grammar/vocabulary tips. "
            "You also have access to several tools (search, calculator, notes, current time). "
            "Use them when needed. Always think step by step and be concise but helpful.\n\n"
            f"Previous conversation (short‑term memory):\n{self._format_memory()}\n\n"
        )

        # Build message list in Gemini format (history)
        history = []
        for mem in self.conversation_memory[-6:]:
            history.append({"role": "user", "parts": [mem["user"]]})
            history.append({"role": "model", "parts": [mem["assistant"]]})

        # Start a chat session
        chat = self.model.start_chat(history=history)

        # Send the user query along with the system prompt (as first message)
        # Gemini doesn't support a system message directly; prepend it as a user message.
        # We'll combine system prompt + user query for better context.
        full_message = f"{system_prompt}\nUser: {user_query}"
        response = chat.send_message(full_message, tools=self._get_tool_definitions())

        # Handle tool calls (up to 5 iterations)
        for _ in range(5):
            if not response.candidates or not response.candidates[0].content.parts:
                break

            # Check if there is a function call in the response
            function_calls = []
            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    function_calls.append(part.function_call)

            if not function_calls:
                # No tool calls – this is the final answer
                final_answer = response.text
                self.conversation_memory.append({"user": user_query, "assistant": final_answer})
                return final_answer

            # Execute all requested tools
            for fc in function_calls:
                func_name = fc.name
                func_args = {k: v for k, v in fc.args.items()}
                result = self._execute_tool(func_name, func_args)
                # Send the tool result back to Gemini
                response = chat.send_message(
                    f"Tool result for '{func_name}': {result}",
                    tools=self._get_tool_definitions(),
                )

        # If we exit the loop without a final answer
        final = "Max iterations reached. Unable to complete."
        self.conversation_memory.append({"user": user_query, "assistant": final})
        return final

# -------------------------------------------------------------------
# 3. FASTAPI SERVER
# -------------------------------------------------------------------
app = FastAPI(title="English Tutor Agent API (Gemini)")

# Allow requests from your Next.js frontend (adjust origin in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://english-tutor-ai-app.netlify.app/"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY missing in .env")
agent = AdvancedAgent(api_key)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        answer = agent.run(request.message)
        return ChatResponse(response=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
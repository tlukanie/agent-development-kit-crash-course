"""
Comparison: Agent with Runner vs Agent without Runner
"""
from fastapi import FastAPI
from pydantic import BaseModel
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# Same agent for both approaches
agent = Agent(
    model='gemini-2.0-flash',
    name='comparison_agent',
    description='Agent for comparing Runner vs Direct usage',
    instruction='You are a helpful assistant. Remember previous conversations when using sessions.',
)

app = FastAPI(title="Runner vs Direct Comparison")

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    has_memory: bool
    approach: str

# APPROACH 1: Direct Agent (Stateless)
@app.post("/direct", response_model=ChatResponse)
async def chat_direct(request: ChatRequest):
    """Direct agent call - NO memory between requests"""
    try:
        message_content = types.Content(
            role="user", 
            parts=[types.Part(text=request.message)]
        )
        
        response = await agent.run_async([message_content])
        response_text = response.parts[0].text if response and response.parts else "No response"
        
        return ChatResponse(
            response=response_text,
            has_memory=False,
            approach="direct_agent"
        )
        
    except Exception as e:
        return ChatResponse(
            response=f"Error: {str(e)}",
            has_memory=False,
            approach="direct_agent"
        )

# APPROACH 2: With Runner (Stateful)
session_service = InMemorySessionService()
runner = Runner(
    agent=agent,
    app_name="ComparisonApp",
    session_service=session_service
)

@app.post("/runner", response_model=ChatResponse)
async def chat_with_runner(request: ChatRequest):
    """Runner-based call - HAS memory between requests"""
    try:
        user_id = "test_user"
        session_id = request.session_id
        
        # Create session if needed
        existing_session = session_service.get_session("ComparisonApp", user_id, session_id)
        if existing_session is None:
            session_service.create_session("ComparisonApp", user_id, session_id, state={})
        
        message_content = types.Content(
            role="user", 
            parts=[types.Part(text=request.message)]
        )
        
        final_response = None
        async for event in runner.run_async(user_id, session_id, message_content):
            if event.is_final_response() and event.content and event.content.parts:
                final_response = event.content.parts[0].text
                break
        
        return ChatResponse(
            response=final_response or "No response",
            has_memory=True,
            approach="runner_based"
        )
        
    except Exception as e:
        return ChatResponse(
            response=f"Error: {str(e)}",
            has_memory=True,
            approach="runner_based"
        )

@app.get("/")
async def info():
    return {
        "message": "Compare /direct vs /runner endpoints",
        "test_instructions": [
            "1. POST to /direct with: {'message': 'My name is John'}",
            "2. POST to /direct with: {'message': 'What is my name?'}",
            "3. POST to /runner with: {'message': 'My name is Jane', 'session_id': 'test1'}",
            "4. POST to /runner with: {'message': 'What is my name?', 'session_id': 'test1'}",
            "Notice: Direct forgets, Runner remembers!"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
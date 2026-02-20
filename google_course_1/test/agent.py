from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from fastapi import FastAPI
from pydantic import BaseModel
from google.genai import types
from dotenv import load_dotenv
import uvicorn

# Load environment variables from .env file
load_dotenv()

# Define your agent
root_agent = Agent(
    model='gemini-2.0-flash-001',
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
)

# Create FastAPI app
app = FastAPI(title="Google ADK Agent API", description="FastAPI server for Google ADK agent")

# Initialize session service and runner (correct way)
session_service = InMemorySessionService()
runner = Runner(
    agent=root_agent,
    app_name="FastAPI_Agent",
    session_service=session_service,
)

# Define request/response models
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str

# Health check endpoint
@app.get("/")
async def health_check():
    return {"status": "healthy", "agent": root_agent.name}

# List apps endpoint (for compatibility with ADK web interface)
@app.get("/list-apps")
async def list_apps():
    return {
        "apps": [
            {
                "name": root_agent.name,
                "description": root_agent.description,
                "model": root_agent.model
            }
        ]
    }

# Main chat endpoint that handles the /run functionality
@app.post("/run", response_model=ChatResponse)
async def run_agent(request: ChatRequest):
    try:
        # Create session if it doesn't exist
        user_id = "default_user"
        session_id = request.session_id
        
        # Try to get existing session
        existing_session = session_service.get_session(
            app_name="FastAPI_Agent",
            user_id=user_id,
            session_id=session_id
        )
        
        # If session doesn't exist, create it
        if existing_session is None:
            session_service.create_session(
                app_name="FastAPI_Agent",
                user_id=user_id,
                session_id=session_id,
                state={}
            )
        
        # Create message content
        new_message = types.Content(
            role="user", 
            parts=[types.Part(text=request.message)]
        )
        
        # Run the agent and collect response
        final_response = None
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_response = event.content.parts[0].text
                break
        
        if final_response is None:
            final_response = "No response generated"
        
        return ChatResponse(
            response=final_response,
            session_id=request.session_id
        )
        
    except Exception as e:
        return ChatResponse(
            response=f"Error: {str(e)}",
            session_id=request.session_id
        )

# Alternative chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    return await run_agent(request)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8060)

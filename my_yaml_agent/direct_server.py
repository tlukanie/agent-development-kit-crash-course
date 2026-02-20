"""
FastAPI server using ADK Agent directly (no Runner)
"""
from fastapi import FastAPI
from pydantic import BaseModel
from google.adk.agents import Agent
from google.genai import types
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Create agent directly
agent = Agent(
    model='gemini-2.0-flash',
    name='direct_agent',
    description='A helpful assistant without Runner wrapper.',
    instruction='Answer user questions to the best of your knowledge',
)

# Create FastAPI app
app = FastAPI(title="Direct Agent API", description="FastAPI without Runner")

# Request/Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.get("/")
async def health_check():
    return {"status": "healthy", "agent": agent.name, "uses_runner": False}

@app.post("/chat", response_model=ChatResponse)
async def chat_direct(request: ChatRequest):
    try:
        # Create message content directly
        message_content = types.Content(
            role="user", 
            parts=[types.Part(text=request.message)]
        )
        
        # Call agent directly (no Runner needed!)
        response = await agent.run_async([message_content])
        
        # Extract response text
        if response and response.parts:
            response_text = response.parts[0].text
        else:
            response_text = "No response generated"
        
        return ChatResponse(response=response_text)
        
    except Exception as e:
        return ChatResponse(response=f"Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
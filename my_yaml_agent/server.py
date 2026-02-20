"""
YAML-based FastAPI server for ADK Agent
"""
from fastapi import FastAPI
from pydantic import BaseModel
from google.genai import types
import uvicorn
from pathlib import Path

# Import our YAML agent loader
from agent import YAMLAgentLoader

# Load configuration
config_file = Path(__file__).parent / "agent_config.yaml"
loader = YAMLAgentLoader(config_file)

# Create agent and runner from YAML config
agent = loader.create_agent()
runner = loader.create_runner(agent)
server_config = loader.get_server_config()

# Create FastAPI app with config from YAML
app = FastAPI(
    title=server_config.get('title', 'YAML Agent API'),
    description=server_config.get('description', 'ADK Agent configured via YAML')
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.get("/")
async def health_check():
    return {
        "status": "healthy", 
        "agent": agent.name,
        "configured_via": "YAML"
    }

@app.get("/config")
async def get_config():
    """Return current agent configuration from YAML"""
    return {
        "agent_name": agent.name,
        "model": agent.model,
        "description": agent.description,
        "tools_enabled": len(agent.tools) if hasattr(agent, 'tools') else 0,
        "config_source": "agent_config.yaml"
    }

@app.post("/run", response_model=ChatResponse)
async def run_agent(request: ChatRequest):
    try:
        user_id = "default_user"
        session_id = request.session_id
        
        # Session management
        existing_session = runner.session_service.get_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id
        )
        
        if existing_session is None:
            runner.session_service.create_session(
                app_name=runner.app_name,
                user_id=user_id,
                session_id=session_id,
                state={}
            )
        
        # Create message and run agent
        new_message = types.Content(
            role="user", 
            parts=[types.Part(text=request.message)]
        )
        
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

if __name__ == "__main__":
    # Use server config from YAML
    host = server_config.get('host', '0.0.0.0')
    port = server_config.get('port', 8000)
    
    print(f"Starting YAML-configured agent: {agent.name}")
    print(f"Server config loaded from: agent_config.yaml")
    
    uvicorn.run(app, host=host, port=port)
import yaml
import os
from pathlib import Path
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from dotenv import load_dotenv
from datetime import datetime

class YAMLAgentLoader:
    """Load and create ADK agents from YAML configuration files."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        """Load YAML configuration from file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _setup_environment(self):
        """Setup environment variables and configurations."""
        load_dotenv()
        
        # Set up environment variables from config
        env_config = self.config.get('environment', {})
        api_key_var = env_config.get('api_key_env_var', 'GOOGLE_API_KEY')
        
        if not os.getenv(api_key_var):
            print(f"Warning: {api_key_var} environment variable not set")
    
    def _create_tools(self) -> list:
        """Create tools based on YAML configuration."""
        tools = []
        tool_configs = self.config.get('agent', {}).get('tools', [])
        
        for tool_config in tool_configs:
            if tool_config.get('enabled', False):
                tool_name = tool_config.get('name')
                if tool_name == 'get_current_time':
                    tools.append(self._get_current_time_tool())
                # Add more tools as needed
                
        return tools
    
    def _get_current_time_tool(self):
        """Create a current time tool."""
        def get_current_time() -> dict:
            """Get the current time in ISO format."""
            return {
                "current_time": datetime.now().isoformat(),
                "timezone": "UTC"
            }
        return get_current_time
    
    def create_agent(self) -> Agent:
        """Create an ADK Agent from YAML configuration."""
        self._setup_environment()
        
        agent_config = self.config.get('agent', {})
        tools = self._create_tools()
        
        agent = Agent(
            name=agent_config.get('name', 'yaml_agent'),
            model=agent_config.get('model', 'gemini-2.0-flash'),
            description=agent_config.get('description', 'YAML configured agent'),
            instruction=agent_config.get('instruction', 'You are a helpful assistant.'),
            tools=tools
        )
        
        return agent
    
    def create_runner(self, agent: Agent = None) -> Runner:
        """Create a Runner with the configured agent."""
        if agent is None:
            agent = self.create_agent()
            
        session_service = InMemorySessionService()
        
        runner = Runner(
            agent=agent,
            app_name=self.config.get('agent', {}).get('name', 'yaml_agent'),
            session_service=session_service
        )
        
        return runner
    
    def get_server_config(self) -> dict:
        """Get server configuration from YAML."""
        return self.config.get('server', {
            'host': '0.0.0.0',
            'port': 8000,
            'title': 'YAML Configured Agent',
            'description': 'Agent configured via YAML'
        })

# Create the agent using YAML configuration
def load_agent_from_yaml(config_path: str = "agent_config.yaml"):
    """Convenience function to load agent from YAML."""
    loader = YAMLAgentLoader(config_path)
    return loader.create_agent()

# For compatibility with ADK discovery
if __name__ != "__main__":
    # Load the agent when imported
    config_file = Path(__file__).parent / "agent_config.yaml"
    if config_file.exists():
        loader = YAMLAgentLoader(config_file)
        root_agent = loader.create_agent()
    else:
        # Fallback to basic agent if no YAML config
        from google.adk.agents import Agent
        root_agent = Agent(
            model='gemini-2.0-flash',
            name='fallback_agent',
            description='Fallback agent when YAML config is missing.',
            instruction='Answer user questions to the best of your knowledge',
        )

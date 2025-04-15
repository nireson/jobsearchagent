import os
import re
from dotenv import load_dotenv, find_dotenv

def load_env_file():
    """Load environment variables from .env file."""
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("# Browser Agent API Keys\n")
            f.write("OPENAI_API_KEY=\n")
            f.write("OPENAI_MODEL=gpt-4o\n")
    
    # Load the environment variables
    load_dotenv(override=True)

def save_env_file(env_vars):
    """
    Save environment variables to .env file without adding quotes.
    
    Args:
        env_vars: Dictionary of environment variables to save
    """
    # Determine the .env file path
    dotenv_path = find_dotenv()
    if not dotenv_path:
        dotenv_path = '.env'
        with open(dotenv_path, 'w') as f:
            f.write("# AOTT AI Research Agent Configuration\n")
    
    # Read existing variables that weren't in the form
    existing_vars = {}
    if os.path.exists(dotenv_path):
        with open(dotenv_path, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    # Skip keys that are in the form (they'll be updated)
                    if key not in env_vars:
                        existing_vars[key] = value
    
    # Write the .env file from scratch
    with open(dotenv_path, 'w') as f:
        f.write("# AOTT AI Research Agent Configuration\n\n")
        
        # Write form variables first
        for key, value in env_vars.items():
            # Clean the value - strip any quotes
            value = value.strip().strip('"\'')
            f.write(f"{key}={value}\n")
        
        # Write existing variables that weren't updated
        for key, value in existing_vars.items():
            f.write(f"{key}={value}\n")
    
    # Reload the environment variables
    load_dotenv(override=True)
    
    # Print debug information
    print("Updated .env file with values:")
    with open(dotenv_path, 'r') as f:
        print(f.read())

def get_env_var(key, default=None):
    """
    Get an environment variable with a default fallback.
    
    Args:
        key: The environment variable key
        default: Default value if the key doesn't exist
        
    Returns:
        The environment variable value or default
    """
    value = os.environ.get(key, default)
    
    # Strip quotes if present
    if isinstance(value, str):
        value = value.strip('"\'')
        
    return value

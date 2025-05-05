import os
import json
import asyncio
import threading
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_from_directory
from flask_socketio import SocketIO, emit
import uuid
import requests
from typing import Dict, List, Optional
import traceback
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig

# Import utility modules
from utils.agent import BrowserAgent
from utils.document import save_result_file
from utils.env_manager import load_env_file, save_env_file

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- Constants ---
PROMPTS_FILE = 'prompts.json'
# MAX_ACTIONS_DEFAULT = 5 # Removed

# Dictionary to store running tasks
running_tasks = {}

# Load environment variables from .env file at startup
load_env_file()

# Load environment variables from .env file
load_dotenv()

# Initialize the Browser Agent (using the imported class)
browser_agent = BrowserAgent()

# --- Prompt Management Helpers ---

def load_prompts():
    """Load prompts from the JSON file."""
    if not os.path.exists(PROMPTS_FILE):
        return []
    try:
        with open(PROMPTS_FILE, 'r') as f:
            content = f.read()
            if not content.strip(): # Check if content is just whitespace
                return []
            # Parse the content string directly
            return json.loads(content)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading prompts: {e}")
        return []

def save_prompts(prompts):
    """Save prompts to the JSON file."""
    try:
        with open(PROMPTS_FILE, 'w') as f:
            json.dump(prompts, f, indent=4)
    except IOError as e:
        print(f"Error saving prompts: {e}")

# --- End Prompt Management Helpers ---

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection."""
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    print('Client disconnected')

@socketio.on_error()
def handle_error(e):
    """Handle WebSocket errors."""
    print(f'WebSocket error: {str(e)}')
    # Returning None or an empty response is usually expected
    return None

@socketio.on_error_default
def default_error_handler(e):
    """Default error handler for WebSocket events."""
    print(f'WebSocket default error: {str(e)}')
    # Returning None or an empty response is usually expected
    return None

def get_openai_models(api_key: str) -> List[Dict[str, str]]:
    """Fetch available models from OpenAI API."""
    if not api_key:
        return []
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get("https://api.openai.com/v1/models", headers=headers)
        if response.status_code == 200:
            models = response.json()["data"]
            # Include all OpenAI chat models (GPT, o1, o3)
            chat_models = [
                {"id": model["id"], "name": model["id"]}
                for model in models
                if model["id"].startswith(("gpt-3.5", "gpt-4", "o1-", "o3-"))
            ]
            return sorted(chat_models, key=lambda x: x["id"])
        return []
    except Exception as e:
        print(f"Error fetching OpenAI models: {str(e)}")
        return []

def get_anthropic_models(api_key: str) -> List[Dict[str, str]]:
    """Fetch available models from Anthropic API."""
    if not api_key:
        return []
    
    try:
        # Using the correct headers as specified in the documentation
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        
        # Get models from the API
        response = requests.get(
            "https://api.anthropic.com/v1/models",
            headers=headers
        )
        
        print(f"Anthropic API Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                models_data = response.json()
                print(f"Anthropic API Raw Response: {json.dumps(models_data, indent=2)}")
                
                # According to the documentation, response has a "data" field with a list of model objects
                available_models = []
                
                if "data" in models_data and isinstance(models_data["data"], list):
                    for model in models_data["data"]:
                        if "id" in model and isinstance(model["id"], str):
                            # Format model name for display
                            display_name = model["id"].replace("-", " ").title()
                            available_models.append({
                                "id": model["id"],
                                "name": display_name
                            })
                    
                    print(f"Found {len(available_models)} Anthropic models")
                    return sorted(available_models, key=lambda x: x["id"], reverse=True)
                else:
                    print(f"Unexpected API response structure: {models_data}")
            except json.JSONDecodeError:
                print(f"Invalid JSON response: {response.text}")
            except Exception as e:
                print(f"Error parsing Anthropic models: {str(e)}")
        else:
            print(f"Anthropic API Error: {response.status_code} - {response.text}")
        
        return []
    except Exception as e:
        print(f"Error fetching Anthropic models: {str(e)}")
        return []

def get_ollama_models(api_url: str) -> List[Dict[str, str]]:
    """Fetch available models from Ollama API using multiple endpoints.
    
    According to Ollama API documentation, we can get models from:
    - /api/tags - Lists locally available models
    - /api/ps - Lists currently running models
    """
    if not api_url:
        api_url = "http://localhost:11434"
    
    # Ensure the URL has a scheme
    if not api_url.startswith(("http://", "https://")):
        api_url = f"http://{api_url}"
    
    # Store unique models by their ID
    all_models = {}
    
    try:
        # 1. Try the /api/tags endpoint first (locally available models)
        tags_url = f"{api_url}/api/tags"
        print(f"Fetching Ollama models from: {tags_url}")
        
        try:
            tags_response = requests.get(tags_url, timeout=5)
            print(f"Ollama /api/tags Status Code: {tags_response.status_code}")
            
            if tags_response.status_code == 200:
                tags_data = tags_response.json()
                
                if "models" in tags_data and isinstance(tags_data["models"], list):
                    for model in tags_data["models"]:
                        if "name" in model:
                            model_id = model["name"]
                            # Format model name for display (remove namespace if present)
                            if "/" in model_id:
                                display_name = model_id.split("/")[-1].replace(":", " ").title()
                            else:
                                display_name = model_id.replace(":", " ").title()
                            
                            # Add to our models dictionary
                            all_models[model_id] = {
                                "id": model_id,
                                "name": display_name
                            }
                    
                    print(f"Found {len(all_models)} models from /api/tags")
        except Exception as e:
            print(f"Error fetching from /api/tags: {str(e)}")
        
        # 2. Try the /api/ps endpoint (running models)
        ps_url = f"{api_url}/api/ps"
        print(f"Fetching running Ollama models from: {ps_url}")
        
        try:
            ps_response = requests.get(ps_url, timeout=5)
            print(f"Ollama /api/ps Status Code: {ps_response.status_code}")
            
            if ps_response.status_code == 200:
                ps_data = ps_response.json()
                
                if "models" in ps_data and isinstance(ps_data["models"], list):
                    for model in ps_data["models"]:
                        if "name" in model:
                            model_id = model["name"]
                            display_name = model_id
                            
                            # Use more detailed formatting if available
                            if "model" in model:
                                model_id = model["model"]
                            
                            # Format display name
                            if "/" in display_name:
                                display_name = display_name.split("/")[-1]
                            display_name = display_name.replace(":", " ").title()
                            
                            # Add to our models dictionary
                            all_models[model_id] = {
                                "id": model_id,
                                "name": display_name
                            }
                    
                    print(f"Found {len(all_models)} models after checking /api/ps")
        except Exception as e:
            print(f"Error fetching from /api/ps: {str(e)}")
        
        # Convert the dictionary to a list and sort by ID
        available_models = list(all_models.values())
        sorted_models = sorted(available_models, key=lambda x: x["id"])
        
        # If we found any models, return them
        if sorted_models:
            return sorted_models
        
        # Otherwise, return default models
        print("No models found, returning defaults")
        return [
            {"id": "llama3", "name": "Llama 3"},
            {"id": "llama3:8b", "name": "Llama 3 8B"},
            {"id": "llama3:70b", "name": "Llama 3 70B"},
            {"id": "mistral", "name": "Mistral"},
            {"id": "gemma:7b", "name": "Gemma 7B"}
        ]
    
    except Exception as e:
        print(f"Error fetching Ollama models: {str(e)}")
        # Return default models in case of error
        return [
            {"id": "llama3", "name": "Llama 3"},
            {"id": "llama3:8b", "name": "Llama 3 8B"},
            {"id": "llama3:70b", "name": "Llama 3 70B"},
            {"id": "mistral", "name": "Mistral"},
            {"id": "gemma:7b", "name": "Gemma 7B"}
        ]

@app.route('/')
def index():
    """Homepage with the prompt input form"""
    return render_template('index.html')

@app.route('/prompts')
def prompts_page():
    """Page to display and manage saved prompts."""
    return render_template('prompts.html')

@app.route('/settings')
def settings():
    """API key management page"""
    # Get current environment variables
    env_vars = {}
    settings = {}
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    env_vars[key] = value
                    # Map environment variables to settings
                    if key == 'OPENAI_API_KEY':
                        settings['openai_api_key'] = value
                    elif key == 'ANTHROPIC_API_KEY':
                        settings['anthropic_api_key'] = value
                    elif key == 'OPENAI_MODEL':
                        settings['openai_model'] = value
                    elif key == 'ANTHROPIC_MODEL':
                        settings['anthropic_model'] = value
                    elif key == 'MODEL_PROVIDER':
                        settings['model_provider'] = value
                    elif key == 'OLLAMA_API_URL':
                        settings['ollama_api_url'] = value
                    elif key == 'OLLAMA_MODEL':
                        settings['ollama_model'] = value
                    # Add CHROME_PATH and others if needed for display
                    elif key == 'CHROME_PATH':
                        settings['chrome_path'] = value
                    elif key == 'TIMEOUT':
                        settings['timeout'] = value
                    elif key == 'DEBUG':
                         settings['debug'] = value
                    # Check for Tavily API Key
                    elif key == 'TAVILY_API_KEY':
                         settings['tavily_api_key'] = value

    # Set default values if not present
    if 'ollama_api_url' not in settings:
        settings['ollama_api_url'] = 'http://localhost:11434'
    if 'model_provider' not in settings:
        settings['model_provider'] = 'openai' # Default provider

    # Fetch available models based on current provider keys/URLs
    openai_models = get_openai_models(settings.get('openai_api_key', ''))
    anthropic_models = get_anthropic_models(settings.get('anthropic_api_key', ''))
    ollama_models = get_ollama_models(settings.get('ollama_api_url', ''))

    # Debug logging
    print("OpenAI Models:", openai_models)
    print("Anthropic Models:", anthropic_models)
    print("Ollama Models:", ollama_models)
    print("Settings:", settings)

    # Check if Tavily key is present and has a value
    tavily_key_present = bool(settings.get('tavily_api_key', '').strip())

    return render_template('settings.html',
                         env_vars=env_vars,
                         settings=settings,
                         openai_models=openai_models,
                         anthropic_models=anthropic_models,
                         ollama_models=ollama_models,
                         tavily_key_present=tavily_key_present)

@app.route('/save_settings', methods=['POST'])
def save_settings():
    """Save API keys and other settings to .env file"""
    # Get current environment variables to preserve existing values
    current_env_vars = {}
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    current_env_vars[key] = value

    # Start with current environment variables
    env_vars = current_env_vars.copy()

    # Update specific values from the form
    env_vars['MODEL_PROVIDER'] = request.form.get('model_provider', 'openai')
    env_vars['OPENAI_API_KEY'] = request.form.get('openai_api_key', '')
    env_vars['OPENAI_MODEL'] = request.form.get('openai_model', 'gpt-3.5-turbo')
    env_vars['ANTHROPIC_API_KEY'] = request.form.get('anthropic_api_key', '')
    env_vars['ANTHROPIC_MODEL'] = request.form.get('anthropic_model', 'claude-3-sonnet-20240229')

    # Get and clean the Ollama API URL
    ollama_api_url = request.form.get('ollama_api_url', 'http://localhost:11434').rstrip('/')
    if ollama_api_url and not ollama_api_url.startswith(('http://', 'https://')):
        ollama_api_url = f'http://{ollama_api_url}'
    env_vars['OLLAMA_API_URL'] = ollama_api_url
    env_vars['OLLAMA_MODEL'] = request.form.get('ollama_model', 'llama3')

    # Handle other settings from the form (using `key_` prefix as before)
    # Preserve existing .env structure if keys exist there
    env_vars['TAVILY_API_KEY'] = request.form.get('tavily_api_key', env_vars.get('TAVILY_API_KEY', ''))
    env_vars['TIMEOUT'] = request.form.get('key_TIMEOUT', env_vars.get('TIMEOUT', '300'))
    env_vars['CHROME_PATH'] = request.form.get('key_CHROME_PATH', env_vars.get('CHROME_PATH', ''))
    env_vars['DEBUG'] = request.form.get('key_DEBUG', env_vars.get('DEBUG', 'False'))
    
    # Debug print
    print("Saving settings:", env_vars)
    
    # Save to .env file
    save_env_file(env_vars)
    
    # Reload environment variables into the current process
    load_dotenv(override=True)
    
    # Reinitialize the agent with potentially new settings
    browser_agent.reinitialize()
    
    return redirect(url_for('settings'))

@app.route('/results')
def results():
    """Results browser page"""
    results_dir = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    result_files = []
    for file in os.listdir(results_dir):
        # Include both .docx and .xlsx files
        if file.lower().endswith(('.docx', '.xlsx')):
            file_path = os.path.join(results_dir, file)
            try:
                file_info = {
                    'filename': file,
                    'path': file_path,
                    'size': os.path.getsize(file_path),
                    'created': datetime.fromtimestamp(os.path.getctime(file_path))
                }
                result_files.append(file_info)
            except OSError as e:
                print(f"Error accessing file {file_path}: {e}")
    
    # Sort by creation date (newest first)
    result_files.sort(key=lambda x: x['created'], reverse=True)
    
    return render_template('results.html', result_files=result_files)

@app.route('/run_task', methods=['POST'])
def run_task():
    """Start a browser task based on the prompt"""
    data = request.get_json()
    print(f"[DEBUG /run_task] Received data: {data}") # Log received data
    task_prompt = data.get('prompt', '')
    format_prompt = data.get('formatPrompt', '')
    selected_model = data.get('model', '')
    max_steps = data.get('maxSteps', 30)
    agent_type = data.get('agentType', 'browser') # Default to browser agent
    print(f"[DEBUG /run_task] Assigned agent_type: {agent_type}") # Log assigned agent_type
    output_format = data.get('outputFormat', 'docx') # Default to docx
    print(f"[DEBUG /run_task] Assigned output_format: {output_format}") # Log assigned output_format
    run_headless = data.get('runHeadless', False) # Get new parameter, default False (visible window)
    
    if not task_prompt:
        return jsonify({'status': 'error', 'message': 'Prompt cannot be empty'})
    
    # Validate max_steps
    try:
        max_steps = int(max_steps)
        if max_steps < 5:
            max_steps = 5
        elif max_steps > 100:
            max_steps = 100
    except (ValueError, TypeError):
        max_steps = 30  # Default if invalid

    # Validate run_headless (ensure boolean)
    if not isinstance(run_headless, bool):
        run_headless = False # Default to False if not a boolean
    print(f"[DEBUG /run_task] Validated run_headless: {run_headless}")

    task_specific_model = None
    # Handle model selection if provided
    if selected_model:
        env_vars = {}
        current_provider = None
        if os.path.exists('.env'):
             with open('.env', 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        env_vars[key] = value
                        if key == 'MODEL_PROVIDER':
                            current_provider = value

        # Check if the selected model requires a temporary override
        provider_model_key = f"{current_provider.upper()}_MODEL" if current_provider else None
        if provider_model_key and env_vars.get(provider_model_key) != selected_model:
            log_message = f"Temporarily using model: {selected_model} (instead of {env_vars.get(provider_model_key, 'default')})"
            print(log_message)
            task_specific_model = selected_model # Pass this to the agent runner

    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    
    # Create a wrapper function that properly handles the async function
    def run_async_task_in_thread(current_task_id, prompt, fmt_prompt, steps, headless_mode, model_override, agent_type_to_use, format_to_use):
        print(f"[DEBUG run_async_task_in_thread] Received agent_type_to_use: {agent_type_to_use}") # Log received agent_type in thread
        print(f"[DEBUG run_async_task_in_thread] Received format_to_use: {format_to_use}") # Log received format in thread
        print(f"[DEBUG run_async_task_in_thread] Received headless_mode: {headless_mode}") # Log headless mode
        async_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(async_loop)
        try:
            # Pass the task-specific model and agent_type
            async_loop.run_until_complete(start_task_with_socketio(
                current_task_id, # Use passed task_id
                prompt,          # Use passed prompt
                fmt_prompt,      # Use passed format_prompt
                steps,           # Use passed max_steps
                task_specific_model=model_override, # Use passed model
                agent_type=agent_type_to_use, # Use passed agent_type
                output_format=format_to_use, # Use passed output format
                run_headless=headless_mode # Pass headless mode
            ))
        except Exception as e:
            print(f"Error running task {current_task_id} in thread: {str(e)}")
            traceback.print_exc()
        finally:
            async_loop.close()
            # Note: Resetting model is handled within start_task_with_socketio's finally block

    # Start task in a background thread
    thread_args = (
        task_id, 
        task_prompt, 
        format_prompt, 
        max_steps, 
        run_headless, # Pass validated value
        task_specific_model, 
        agent_type,
        output_format
    )
    print(f"[DEBUG /run_task] Passing args to thread: {thread_args}") # Log args being passed
    thread = threading.Thread(target=run_async_task_in_thread, args=thread_args)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'success', 
        'message': 'Task started', 
        'task_id': task_id
    })

@app.route('/cancel_task', methods=['POST'])
def cancel_task():
    """Cancel a running browser task"""
    data = request.get_json()
    task_id = data.get('task_id')
    
    if task_id in running_tasks:
        print(f"Cancellation requested for task: {task_id}")
        # Mark the task for cancellation
        running_tasks[task_id]['stop'] = True

        # Give the task a moment to recognize the stop signal
        time.sleep(0.5)

        # Emit a cancelled event via Socket.IO
        # Let the finally block in start_task_with_socketio handle the final cleanup and emit
        socketio.emit('task_cancelling', {
            'task_id': task_id,
            'message': 'Task cancellation initiated...'
        })

        return jsonify({
            'status': 'success',
            'message': 'Task cancellation initiated successfully'
        })

    return jsonify({
        'status': 'error',
        'message': 'Task not found or already completed'
    })

@app.route('/download_result/<filename>')
def download_result(filename):
    """
    Download a specific result file.
    
    Args:
        filename: The name of the file to download.
        
    Returns:
        Response: A Flask response object with the file as an attachment.
        
    Raises:
        404: If the file is not found.
        403: If the file path is invalid.
    """
    results_dir = os.path.join(os.getcwd(), 'results')
    # Securely join the path and resolve it to prevent path traversal
    try:
        safe_path = os.path.abspath(os.path.join(results_dir, filename))

        # Ensure the resolved path is still within the results directory
        if not safe_path.startswith(os.path.abspath(results_dir)):
             print(f"Forbidden: Path traversal attempt denied for {filename}")
             return jsonify({'status': 'error', 'message': 'Invalid file path'}), 403

        # Check if file exists using the safe path
        if not os.path.isfile(safe_path):
             print(f"File not found: {safe_path}")
             return jsonify({'status': 'error', 'message': 'File not found'}), 404

        # Send the file as an attachment
        return send_from_directory(
            results_dir, # Use original directory for sending
            filename,    # Use original filename for sending
            as_attachment=True,
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )

    except Exception as e:
        print(f"Error downloading file '{filename}': {str(e)}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'Error downloading file: {str(e)}'}), 500

@app.route('/diagnostics')
def diagnostics():
    """Diagnostic page to check system settings"""
    # Check Chrome path from environment
    chrome_path = os.environ.get("CHROME_PATH", "").strip('"\'')
    chrome_exists = False
    chrome_is_file = False
    directory_contents = []
    error_message = None
    
    if chrome_path:
        try:
            chrome_exists = os.path.exists(chrome_path)
            if chrome_exists:
                chrome_is_file = os.path.isfile(chrome_path)
                if not chrome_is_file:
                    error_message = "CHROME_PATH exists but is not a file (it should point to the executable)."
                else:
                     # Check if executable (basic check on Windows)
                     if os.name == 'nt' and not chrome_path.lower().endswith('.exe'):
                         error_message = "Warning: CHROME_PATH on Windows does not end with .exe"
            else:
                 error_message = "CHROME_PATH does not exist."

            # Get directory listing of the parent directory
            parent_dir = os.path.dirname(chrome_path)
            if os.path.isdir(parent_dir):
                directory_contents = os.listdir(parent_dir)[:20] # Limit listing
            else:
                 directory_contents = [f"Parent directory '{parent_dir}' not found."]

        except Exception as e:
            error_message = f"Error checking CHROME_PATH: {str(e)}"
            directory_contents = [f"Error listing directory: {str(e)}"]
    else:
        error_message = "CHROME_PATH environment variable is not set."

    # Get relevant environment variables from os.environ
    relevant_env_vars = {}
    keys_to_check = ['MODEL_PROVIDER', 'OPENAI_API_KEY', 'OPENAI_MODEL',
                     'ANTHROPIC_API_KEY', 'ANTHROPIC_MODEL',
                     'OLLAMA_API_URL', 'OLLAMA_MODEL',
                     'CHROME_PATH', 'TIMEOUT', 'DEBUG']
    for key in keys_to_check:
        relevant_env_vars[key] = os.environ.get(key, 'Not Set')
        # Mask API keys
        if 'API_KEY' in key and relevant_env_vars[key] != 'Not Set':
            relevant_env_vars[key] = relevant_env_vars[key][:4] + '...' + relevant_env_vars[key][-4:]

    # Check if .env file exists and read its content
    env_file_exists = os.path.exists('.env')
    env_file_content = "Not found or empty."
    if env_file_exists:
        try:
            with open('.env', 'r') as f:
                env_file_content = f.read()
            if not env_file_content.strip():
                 env_file_content = ".env file exists but is empty."
        except Exception as e:
            env_file_content = f"Error reading .env file: {str(e)}"
    
    # Return diagnostic information
    return jsonify({
        'chrome_path': {
             'value': chrome_path,
             'exists': chrome_exists,
             'is_file': chrome_is_file,
             'error': error_message,
             'parent_dir_contents': directory_contents
        },
        'environment_variables': relevant_env_vars,
        'dotenv_file': {
             'exists': env_file_exists,
             'content_preview': env_file_content[:500] + ('...' if len(env_file_content) > 500 else '') # Preview content
        }
    })

@app.route('/delete_result/<filename>', methods=['POST'])
def delete_result(filename):
    """Delete a result file"""
    results_dir = os.path.join(os.getcwd(), 'results')
    try:
        # Securely join path and resolve
        safe_path = os.path.abspath(os.path.join(results_dir, filename))

        # Verify path is within the results directory
        if not safe_path.startswith(os.path.abspath(results_dir)):
            print(f"Forbidden: Path traversal attempt denied for deletion of {filename}")
            return jsonify({'status': 'error', 'message': 'Invalid file path'}), 403

        # Check if file exists and is a file before attempting deletion
        if os.path.isfile(safe_path):
            os.remove(safe_path)
            print(f"Deleted result file: {safe_path}")
            return jsonify({'status': 'success', 'message': 'File deleted successfully'})
        else:
             print(f"File not found for deletion: {safe_path}")
             return jsonify({'status': 'error', 'message': 'File not found'}), 404

    except Exception as e:
        print(f"Error deleting file '{filename}': {str(e)}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'Error deleting file: {str(e)}'}), 500

@app.route('/restart_app', methods=['POST'])
def restart_app():
    """Restart the Flask application and browser agent"""
    try:
        print("Restart requested...")
        # Start a background thread to handle the actual restart
        restart_thread = threading.Thread(target=perform_restart)
        restart_thread.daemon = True
        restart_thread.start()

        return jsonify({
            'status': 'success',
            'message': 'Application restart initiated'
        })
    except Exception as e:
        print(f"Error initiating restart: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Error initiating restart: {str(e)}'
        })

def perform_restart():
    """Perform the actual restart operations in a background thread"""
    print("Performing restart in background thread...")
    try:
        # Signal all running tasks to stop
        task_ids = list(running_tasks.keys())
        print(f"Signalling stop for tasks: {task_ids}")
        for task_id in task_ids:
            if task_id in running_tasks:
                running_tasks[task_id]['stop'] = True
                # Emit cancellation event via Socket.IO (might be handled in finally block too)
                socketio.emit('task_cancelling', {
                    'task_id': task_id,
                    'message': 'Task cancelled due to application restart'
                })

        # Wait briefly for tasks to acknowledge stop signal
        time.sleep(1.5) # Slightly longer wait

        # Force clear any remaining tasks (should be stopped, but just in case)
        running_tasks.clear()
        print("Running tasks cleared.")

        # Clean up browser resources before reinitializing
        print("Cleaning up browser resources...")
        try:
            # Call cleanup method on the agent instance
             browser_agent.cleanup_resources()
             print("Browser resources cleaned up.")
        except Exception as e:
            print(f"Warning: Error cleaning up browser resources: {str(e)}")

        # Reinitialize the browser agent (loads new settings)
        print("Reinitializing browser agent...")
        browser_agent.reinitialize()
        print("Browser agent reinitialized.")

        # Emit a restart complete event
        socketio.emit('app_restarted', {
            'status': 'success',
            'message': 'Application restarted successfully'
        })
        print("Application restart completed successfully.")

    except Exception as e:
        print(f"Error during application restart process: {str(e)}")
        traceback.print_exc()
        # Emit error event
        socketio.emit('app_restart_error', {
            'status': 'error',
            'message': f'Error during application restart: {str(e)}'
        })

@app.route('/task_status', methods=['GET'])
def get_task_status():
    """Get the status of all running tasks or a specific task"""
    task_id = request.args.get('task_id')

    # If a specific task ID is provided, return that task's detailed status
    if task_id:
        if task_id in running_tasks:
            task_info = running_tasks[task_id]
            elapsed_time = time.time() - task_info['start_time']
            return jsonify({
                'status': 'running',
                'task_id': task_id,
                'prompt': task_info.get('prompt', 'N/A'),
                'format_prompt': task_info.get('format_prompt', ''),
                'elapsed_time': round(elapsed_time, 2),
                'logs': task_info.get('logs', []),
                'started_at': task_info.get('start_time', 0)
            })
        else:
             return jsonify({'status': 'not_found', 'task_id': task_id}), 404

    # If no task ID, return summary of all active tasks
    active_tasks = {}
    for tid, task_info in running_tasks.items():
        active_tasks[tid] = {
            'prompt': task_info.get('prompt', 'N/A'),
            'format_prompt': task_info.get('format_prompt', ''),
            'elapsed_time': round(time.time() - task_info.get('start_time', 0), 2),
            'started_at': task_info.get('start_time', 0)
        }

    return jsonify({
        'has_active_tasks': len(active_tasks) > 0,
        'active_tasks': active_tasks
    })

@app.route('/end_task', methods=['POST'])
def end_task():
    """End a running browser task but preserve partial results"""
    data = request.get_json()
    task_id = data.get('task_id')

    if task_id in running_tasks:
        print(f"Graceful end requested for task: {task_id}")
        # Mark the task for graceful termination
        running_tasks[task_id]['end'] = True

        # Give the task a moment to recognize the end signal
        time.sleep(0.5)

        # Emit an event to notify the client
        socketio.emit('task_ending', {
            'task_id': task_id,
            'message': 'Task ending requested - preserving current progress...'
        })

        return jsonify({
            'status': 'success',
            'message': 'Task ending requested'
        })

    return jsonify({
        'status': 'error',
        'message': 'Task not found or already completed'
    })

@app.route('/refresh_ollama_models', methods=['POST'])
def refresh_ollama_models():
    """Refresh the list of available Ollama models."""
    data = request.get_json()

    # Get API URL from request or use default from environment
    api_url = data.get('api_url') if data and 'api_url' in data else None
    if not api_url:
        api_url = os.environ.get('OLLAMA_API_URL', 'http://localhost:11434')

    try:
        # Get models using the provided URL
        models = get_ollama_models(api_url)
        print(f"Refreshed Ollama models from {api_url}: Found {len(models)}")
        return jsonify({
            'status': 'success',
            'models': models,
            'count': len(models)
        })
    except Exception as e:
        print(f"Error refreshing Ollama models from {api_url}: {str(e)}")
        return jsonify({
             'status': 'error',
             'message': f'Failed to refresh models: {str(e)}',
             'models': [],
             'count': 0
        }), 500

@app.route('/available_models')
def available_models():
    """Get available models for the UI selection dropdown based on current settings"""
    try:
        # Determine the current provider from environment
        model_provider = os.environ.get('MODEL_PROVIDER', 'openai')
        current_model_id = None
        current_model_name = None
        models = []

        print(f"Fetching available models for provider: {model_provider}")

        if model_provider == 'openai':
            current_model_id = os.environ.get('OPENAI_MODEL', 'gpt-4o')
            api_key = os.environ.get('OPENAI_API_KEY', '')
            if api_key:
                models = get_openai_models(api_key)
                print(f"Fetched {len(models)} OpenAI models")
            if not models:
                models = [
                    {"id": "gpt-4o", "name": "GPT-4o"},
                    {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"},
                    {"id": "gpt-4", "name": "GPT-4"},
                    {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"}
                ]
                print("Using default OpenAI models")

        elif model_provider == 'anthropic':
            current_model_id = os.environ.get('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229')
            api_key = os.environ.get('ANTHROPIC_API_KEY', '')
            if api_key:
                models = get_anthropic_models(api_key)
                print(f"Fetched {len(models)} Anthropic models")
            if not models:
                models = [
                    {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus"},
                    {"id": "claude-3-sonnet-20240229", "name": "Claude 3 Sonnet"},
                    {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku"},
                    {"id": "claude-2.1", "name": "Claude 2.1"}
                ]
                print("Using default Anthropic models")

        elif model_provider == 'ollama':
            current_model_id = os.environ.get('OLLAMA_MODEL', 'llama3')
            api_url = os.environ.get('OLLAMA_API_URL', '')
            models = get_ollama_models(api_url) # Fetches or returns defaults
            print(f"Fetched {len(models)} Ollama models (or defaults)")

        else: # Default to OpenAI if provider is unknown
             model_provider = 'openai'
             current_model_id = os.environ.get('OPENAI_MODEL', 'gpt-4o')
             models = [
                    {"id": "gpt-4o", "name": "GPT-4o"},
                    {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"},
                    {"id": "gpt-4", "name": "GPT-4"},
                    {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"}
             ]
             print("Unknown provider, using default OpenAI models")

        # Format the current model name for display
        if current_model_id:
            current_model_name = current_model_id.replace('-', ' ').replace(':', ' ').title()
            current_model = {"id": current_model_id, "name": current_model_name}
        else:
             current_model = None # Should not happen if defaults are set

        # Ensure the current model is in the list, add if missing
        if current_model:
            current_model_in_list = any(m['id'] == current_model['id'] for m in models)
            if not current_model_in_list:
                models.insert(0, current_model)

        return jsonify({
            'current_model': current_model,
            'available_models': models,
            'provider': model_provider
        })

    except Exception as e:
        print(f"Error in available_models route: {str(e)}")
        traceback.print_exc()
        # Return a safe default response
        return jsonify({
            'current_model': {"id": "error-model", "name": "Error"},
            'available_models': [{"id": "error-model", "name": "Error Fetching Models"}],
            'provider': "unknown",
            'error': str(e)
        }), 500

async def start_task_with_socketio(task_id, task_prompt, format_prompt, max_steps, task_specific_model, agent_type, output_format, run_headless):
    print(f"[DEBUG start_task_with_socketio] Received agent_type: {agent_type}") # Log received agent_type
    print(f"[DEBUG start_task_with_socketio] Received output_format: {output_format}") # Log received output format
    print(f"[DEBUG start_task_with_socketio] Received run_headless: {run_headless}") # Log headless mode
    """Run the agent task and emit socket events for progress updates."""
    global running_tasks

    def log_callback(message, level='info'):
        # Log to console for debugging
        print(f"[{task_id} - {level.upper()}] {message}")

        # Ensure task still exists before logging/emitting
        if task_id not in running_tasks:
            print(f"[{task_id}] Task ended or cancelled, skipping log: {message}")
            return None # Task gone, do nothing

        # Always store log
        running_tasks[task_id]['logs'].append(f"[{level.upper()}] {message}")

        # ----- Stuck Detection Logic -----
        is_stuck = False
        reason = ""
        stuck_check_enabled = True # Can be disabled for debugging

        if stuck_check_enabled:
            import re
            step_match = re.search(r'Step (\d+):', message)
            action_match = re.search(r'(clicking|typing|navigating|searching|reading|looking)', message.lower())

            # Track recent actions (keep last N)
            action_history_limit = 10
            running_tasks[task_id]['last_actions'].append(message)
            if len(running_tasks[task_id]['last_actions']) > action_history_limit:
                running_tasks[task_id]['last_actions'].pop(0)

            if step_match:
                step_number = step_match.group(1)
                current_step = running_tasks[task_id]['current_step']
                max_repeats = 5 # Max times to repeat a step before intervention

                if step_number != current_step:
                    # New step detected
                    running_tasks[task_id]['current_step'] = step_number
                    running_tasks[task_id]['step_tracker'][step_number] = 1
                    running_tasks[task_id]['last_step_change_time'] = time.time()
                    running_tasks[task_id]['error_count_this_step'] = 0 # Reset error count for new step
                else:
                    # Same step repeated
                    running_tasks[task_id]['step_tracker'][step_number] = running_tasks[task_id]['step_tracker'].get(step_number, 0) + 1
                    if running_tasks[task_id]['step_tracker'][step_number] >= max_repeats:
                        is_stuck = True
                        reason = f"repeated Step {step_number} {max_repeats} times"

            # Check for repeating the exact same log message (strong indicator of loop)
            if len(running_tasks[task_id]['last_actions']) >= 3:
                last_3 = running_tasks[task_id]['last_actions'][-3:]
                if last_3[0] == last_3[1] == last_3[2]:
                    is_stuck = True
                    reason = "repeated the exact same action 3 times"

            # Check for frequent error messages within the current step
            error_keywords = ["failed", "error", "cannot", "unable to", "not found", "doesn't exist", "timeout"]
            if any(keyword in message.lower() for keyword in error_keywords) and action_match:
                 running_tasks[task_id]['error_count_this_step'] += 1
                 max_errors_per_step = 3
                 if running_tasks[task_id]['error_count_this_step'] >= max_errors_per_step:
                     is_stuck = True
                     reason = f"encountered {max_errors_per_step} errors on the current step"


            # ----- Intervention Logic -----
            if is_stuck:
                min_time_between_interventions = 30 # seconds
                time_since_last_intervention = time.time() - running_tasks[task_id]['last_intervention_time']

                if time_since_last_intervention > min_time_between_interventions:
                    step_num = running_tasks[task_id]['current_step'] or "current"
                    running_tasks[task_id]['interventions_attempted'] += 1
                    urgency = "NOW" if running_tasks[task_id]['interventions_attempted'] <= 1 else "IMMEDIATELY"

                    intervention_log = f"Agent appears stuck on Step {step_num} ({reason}). Attempting intervention #{running_tasks[task_id]['interventions_attempted']}..."
                    print(f"Task {task_id}: {intervention_log}")
                    running_tasks[task_id]['logs'].append(intervention_log)

                    intervention_message = f"""
SYSTEM MESSAGE (Intervention #{running_tasks[task_id]['interventions_attempted']}):
You seem stuck on Step {step_num} because you have {reason}.
Please try one of these options {urgency}:
1. Move on to the next logical step (e.g., Step {int(step_num) + 1 if step_num.isdigit() else 'next'}).
2. Attempt the current step using a completely different method or action.
3. If interacting with a specific element fails, try an alternative element or navigation method.
4. Briefly summarize why you are stuck and what you will try next.
5. If absolutely necessary, skip this specific sub-task and proceed with the main objective.

DO NOT repeat the failing action. Change your approach.
"""
                    # Emit logs via socket
                    socketio.emit('task_log', {'task_id': task_id, 'message': intervention_log, 'timestamp': time.time(), 'level': 'warning'})
                    socketio.emit('task_log', {'task_id': task_id, 'message': intervention_message, 'timestamp': time.time(), 'level': 'system'})

                    # Update intervention time and potentially reset step counter
                    running_tasks[task_id]['last_intervention_time'] = time.time()
                    if step_num in running_tasks[task_id]['step_tracker']:
                         running_tasks[task_id]['step_tracker'][step_num] = 0 # Reset counter after intervention
                         running_tasks[task_id]['error_count_this_step'] = 0

                    # Return the intervention message to be possibly used by the agent (implementation specific)
                    # In browser-use, log_callback doesn't directly feed back, but logs are visible
                    # If a different agent library were used, this could be returned to influence it.
                    # For now, it's primarily for logging and observation.

        # ----- Normal Log Emission -----
        # Emit the original log message via socket
        socketio.emit('task_log', {
            'task_id': task_id,
            'message': message, # Send original message
            'timestamp': time.time(),
            'level': level
        })

        # Return None for normal logs, indicating no special agent action needed from callback
        return None

    # --- Main Task Execution ---
    result = None
    try:
        # Create task entry in the global dictionary
        running_tasks[task_id] = {
            'prompt': task_prompt,
            'format_prompt': format_prompt,
            'start_time': time.time(),
            'stop': False, # Flag for hard cancellation
            'end': False,  # Flag for graceful termination
            'logs': [],
            'step_tracker': {}, # Track step repetitions {step_num: count}
            'current_step': None,
            'last_step_change_time': time.time(),
            'error_count_this_step': 0,
            'last_intervention_time': 0, # Track intervention timing
            'interventions_attempted': 0,
            'last_actions': [], # Track recent actions
            'partial_results': None, # Store partial results if needed
            'task_specific_model': task_specific_model, # Store model used
            'agent_type': agent_type, # Store agent type
            'final_status': 'running', # Add a field to track the intended final state
            'output_format': output_format, # Store the requested output format
            'run_headless': run_headless, # Store headless preference
        }
        print(f"Task {task_id} created. Prompt: '{task_prompt[:50]}...' Model: {task_specific_model or 'default'}")

        # Use the custom log callback
        log_handler = log_callback

        log_handler("Starting agent task execution...") # Adjusted log message

        print(f"[DEBUG start_task_with_socketio] Calling browser_agent.run_task with agent_type: {agent_type}") # Log type just before call
        # Configure and run the agent task using the imported BrowserAgent instance
        result = await browser_agent.run_task(
            task_description=task_prompt,
                log_callback=log_handler, 
            stop_check=lambda: task_id not in running_tasks or running_tasks[task_id]['stop'],
            end_check=lambda: task_id in running_tasks and running_tasks[task_id]['end'],
            format_prompt=format_prompt,
            max_steps=max_steps,
            task_specific_model=task_specific_model, # Pass temporary model override
            agent_type=agent_type, # Pass agent type
            run_headless=run_headless # Pass headless mode
        )

        # --- Determine Outcome (if no exception occurred) ---
        if task_id in running_tasks: # Check if task wasn't cancelled/removed during run
            if running_tasks[task_id]['end']: # Graceful end requested
                running_tasks[task_id]['final_status'] = 'ended'
                log_handler(f"Task {task_id} ended gracefully by request.")
            elif running_tasks[task_id]['stop']: # Stop was set but didn't raise CancelledError (unlikely but possible)
                 running_tasks[task_id]['final_status'] = 'cancelled'
                 log_handler(f"Task {task_id} stopped by request (post-run check).", level='warning')
            else: # Normal completion
                running_tasks[task_id]['final_status'] = 'completed'
                log_handler(f"Task {task_id} completed normally.")

    except asyncio.CancelledError:
        # --- Task Cancellation Handling --- 
        log_handler("Task run was cancelled (CancelledError caught).")
        if task_id in running_tasks:
            # Check if cancellation was triggered by the 'end' flag
            if running_tasks[task_id]['end']:
                running_tasks[task_id]['final_status'] = 'ended'
                log_handler(f"Task {task_id} ended by request (CancelledError handled).")
                result = "Task Ended" # Set appropriate result placeholder
            else:
                running_tasks[task_id]['final_status'] = 'cancelled'
                log_handler(f"Task {task_id} cancelled by request (CancelledError handled).")
                result = "Task Cancelled" 
        else:
            # Task already removed, assume cancelled
            log_handler(f"Task {task_id} cancelled (task already removed).", level='warning')
            result = "Task Cancelled" 

    except Exception as e:
        # --- Task Execution Error Handling ---
        error_message = f"Critical error during task {task_id} execution: {str(e)}"
        print(error_message) # Ensure critical errors are printed
        traceback.print_exc()
        log_handler(error_message, level='critical') # Use critical level
        if task_id in running_tasks:
            running_tasks[task_id]['final_status'] = 'error'
            running_tasks[task_id]['error_message'] = error_message # Store error message
        result = f"Critical Error: {str(e)}" # Set result to error message

    finally:
        # --- Task Cleanup & Final Status Emission --- 
        print(f"Executing finally block for task {task_id}")
        final_status = 'unknown'
        final_message = "Task finished with unknown status."
        saved_filename = None

        if task_id in running_tasks:
            task_info = running_tasks[task_id]
            elapsed_time = round(time.time() - task_info.get('start_time', time.time()), 2)
            final_status = task_info.get('final_status', 'unknown')

            # --- Save Results (Only if Completed or Ended) ---
            if final_status in ['completed', 'ended']:
                log_handler(f"Task {task_id} {final_status}. Saving results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"result_{task_id[:8]}_{timestamp}.docx"
        results_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(results_dir, exist_ok=True)

        try:
                    # Use the 'result' variable captured from the try block, or partial results if available
            # Get the requested format for saving
            requested_format = task_info.get('output_format', 'docx') 
            result_content_to_save = result or task_info.get('partial_results', "No result captured.")
            result_path = save_result_file(
                result_content=result_content_to_save,
                filename=base_filename,
                output_format=requested_format, # Pass format to saving function
                search_query=task_info.get('prompt', ''),
                format_instructions=task_info.get('format_prompt', ''),
                execution_logs=task_info.get('logs', [])
            )
        except Exception as save_err:
            error_message = f"Error saving document for task {task_id}: {str(save_err)}"
            log_handler(error_message, level='error')
            traceback.print_exc()
            final_status = 'error' # Downgrade status if saving failed
            final_message = error_message
            # Ensure error message is stored if saving fails and task still exists
            if task_id in running_tasks: 
                running_tasks[task_id]['error_message'] = error_message 

        if not result_path or not os.path.exists(result_path):
            raise ValueError(f"Failed to save document or file not found at {result_path}")
            
            saved_filename = base_filename # Store filename for emit
            log_handler(f"Result saved to {saved_filename}")
            final_message = result or "Partial results saved." if final_status == 'ended' else "Task completed successfully."


        # --- Set Final Messages for Other Statuses ---
        elif final_status == 'cancelled':
            final_message = f"Task {task_id} was cancelled after {elapsed_time}s."
        elif final_status == 'error':
                final_message = task_info.get('error_message', f"Task {task_id} failed after {elapsed_time}s.")
        else: # Handle unknown or unexpected status
                final_message = f"Task {task_id} finished with status '{final_status}' after {elapsed_time}s."
        
        print(final_message) # Log final message to console

            # --- Model Reset (before removing task info) --- 
        if task_info.get('task_specific_model'):
                try:
                    browser_agent.reset_model()
                    print(f"Reset agent model after task {task_id}.")
                except Exception as reset_err:
                     print(f"Warning: Error resetting agent model for task {task_id}: {reset_err}")

            # --- Remove Task BEFORE Emitting Final Event --- 
            # This prevents race conditions with background status checks
        print(f"Removing task {task_id} from running tasks before final emit.")
        running_tasks.pop(task_id, None)

            # --- Emit Final Socket Event --- 
        if final_status in ['completed', 'ended']:
            print(f"Emitting task_complete for {task_id}")
            socketio.emit('task_complete', {
            'task_id': task_id,
            'success': True,
            'result': final_message, 
            'filename': saved_filename,
            'elapsed_time': elapsed_time,
            'status': final_status # Send 'completed' or 'ended'
        })
        elif final_status == 'cancelled':
            print(f"Emitting task_cancelled for {task_id}")
            socketio.emit('task_cancelled', {
                'task_id': task_id,
                'message': final_message,
                'elapsed_time': elapsed_time
            })
        elif final_status == 'error':
            print(f"Emitting task_error for {task_id}")
            socketio.emit('task_error', {
                'task_id': task_id,
                'error': final_message,
                'elapsed_time': elapsed_time
            })
        else: # Fallback for unknown status
            print(f"Emitting task_error (unknown status) for {task_id}")
            socketio.emit('task_error', {
                'task_id': task_id,
                'error': f"Task finished with unknown status: {final_status}",
                'elapsed_time': elapsed_time
            })
        

            print(f"Final message for task {task_id}: {final_message}") # Log final message



# --- API Endpoints for Prompts ---

@app.route('/api/prompts', methods=['GET'])
def get_prompts():
    """API endpoint to get all saved prompts."""
    prompts = load_prompts()
    return jsonify(prompts)

@app.route('/api/prompts', methods=['POST'])
def add_prompt():
    """API endpoint to add a new prompt."""
    data = request.get_json()
    if not data or 'text' not in data or not data['text'].strip():
        return jsonify({'status': 'error', 'message': 'Prompt text cannot be empty'}), 400
    
    prompts = load_prompts()
    new_prompt = {
        'id': str(uuid.uuid4()),
        'text': data['text'].strip()
    }
    prompts.append(new_prompt)
    save_prompts(prompts)
    return jsonify({'status': 'success', 'prompt': new_prompt}), 201

@app.route('/api/prompts/search', methods=['GET'])
def search_prompts():
    """API endpoint to search saved prompts."""
    query = request.args.get('query', '').lower()
    prompts = load_prompts()
    if not query:
        return jsonify(prompts)
    
    filtered_prompts = [p for p in prompts if query in p.get('text', '').lower()]
    return jsonify(filtered_prompts)

@app.route('/api/prompts/<prompt_id>', methods=['DELETE'])
def delete_prompt(prompt_id):
    """API endpoint to delete a specific prompt."""
    prompts = load_prompts()
    initial_length = len(prompts)
    prompts = [p for p in prompts if p.get('id') != prompt_id]
    
    if len(prompts) < initial_length:
        save_prompts(prompts)
        return jsonify({'status': 'success', 'message': 'Prompt deleted'})
    else:
        return jsonify({'status': 'error', 'message': 'Prompt not found'}), 404

# --- End API Endpoints for Prompts ---

if __name__ == '__main__':
    # Ensure results directory exists
    results_path = 'results'
    if not os.path.exists(results_path):
        print(f"Creating results directory at: {os.path.abspath(results_path)}")
        os.makedirs(results_path)

    # Determine host based on DEBUG environment variable
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    host = '127.0.0.1' if debug_mode else '0.0.0.0'
    port = 5000

    print(f"Starting Flask app with Socket.IO...")
    print(f" * Host: {host}:{port}")
    print(f" * Debug Mode: {debug_mode}")
    print(f" * Allow Unsafe Werkzeug: True") # Necessary for threading mode sometimes
    print(f" * Use Reloader: False") # Important for SocketIO stability
    
    # Run the Flask app with Socket.IO
    # use_reloader=False is important to prevent issues with threading/async modes
    socketio.run(app,
                host=host,
                port=port,
                debug=debug_mode,
                allow_unsafe_werkzeug=True,
                use_reloader=False) # MUST be False for stability

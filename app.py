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

# Import utility modules
from utils.agent import BrowserAgent
from utils.document import save_result_as_docx
from utils.env_manager import load_env_file, save_env_file

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*")

# Dictionary to store running tasks
running_tasks = {}

# Load environment variables from .env file at startup
load_env_file()

# Initialize the Browser Agent
browser_agent = BrowserAgent()

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
    
    # Set default values if not present
    if 'ollama_api_url' not in settings:
        settings['ollama_api_url'] = 'http://localhost:11434'
    
    # Fetch available models
    openai_models = get_openai_models(settings.get('openai_api_key', ''))
    anthropic_models = get_anthropic_models(settings.get('anthropic_api_key', ''))
    ollama_models = get_ollama_models(settings.get('ollama_api_url', ''))
    
    # Debug logging
    print("OpenAI Models:", openai_models)
    print("Anthropic Models:", anthropic_models)
    print("Ollama Models:", ollama_models)
    print("Settings:", settings)
    
    return render_template('settings.html', 
                         env_vars=env_vars, 
                         settings=settings,
                         openai_models=openai_models,
                         anthropic_models=anthropic_models,
                         ollama_models=ollama_models)

@app.route('/save_settings', methods=['POST'])
def save_settings():
    """Save API keys to .env file"""
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
    
    # Handle model provider selection
    model_provider = request.form.get('model_provider', 'openai')
    env_vars['MODEL_PROVIDER'] = model_provider
    
    # Update API keys and models without clearing the other provider
    if model_provider == 'openai':
        env_vars['OPENAI_API_KEY'] = request.form.get('openai_api_key', '')
        env_vars['OPENAI_MODEL'] = request.form.get('openai_model', 'gpt-3.5-turbo')
    elif model_provider == 'anthropic':
        env_vars['ANTHROPIC_API_KEY'] = request.form.get('anthropic_api_key', '')
        env_vars['ANTHROPIC_MODEL'] = request.form.get('anthropic_model', 'claude-3-sonnet-20240229')
    elif model_provider == 'ollama':
        # Get and clean the Ollama API URL (remove trailing slashes)
        ollama_api_url = request.form.get('ollama_api_url', 'http://localhost:11434').rstrip('/')
        
        # Ensure URL has a scheme
        if not ollama_api_url.startswith(('http://', 'https://')):
            ollama_api_url = f'http://{ollama_api_url}'
            
        env_vars['OLLAMA_API_URL'] = ollama_api_url
        env_vars['OLLAMA_MODEL'] = request.form.get('ollama_model', 'llama3')
    
    # Handle other settings
    env_vars['TIMEOUT'] = request.form.get('key_TIMEOUT', '300')
    env_vars['CHROME_PATH'] = request.form.get('key_CHROME_PATH', '')
    env_vars['DEBUG'] = request.form.get('key_DEBUG', 'False')
    
    # Debug print
    print("Saving settings:", env_vars)
    
    # Save to .env file
    save_env_file(env_vars)
    
    # Reload environment variables
    load_env_file()
    
    # Reinitialize the agent with new API keys
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
        if file.endswith('.docx'):
            file_path = os.path.join(results_dir, file)
            file_info = {
                'filename': file,
                'path': file_path,
                'size': os.path.getsize(file_path),
                'created': datetime.fromtimestamp(os.path.getctime(file_path))
            }
            result_files.append(file_info)
    
    # Sort by creation date (newest first)
    result_files.sort(key=lambda x: x['created'], reverse=True)
    
    return render_template('results.html', result_files=result_files)

@app.route('/run_task', methods=['POST'])
def run_task():
    """Start a browser task based on the prompt"""
    data = request.get_json()
    task_prompt = data.get('prompt', '')
    format_prompt = data.get('formatPrompt', '')
    
    if not task_prompt:
        return jsonify({'status': 'error', 'message': 'Prompt cannot be empty'})
    
    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    
    # Create a wrapper function that properly handles the async function
    def run_async_task_in_thread():
        async_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(async_loop)
        try:
            async_loop.run_until_complete(start_task_with_socketio(task_id, task_prompt, format_prompt))
        except Exception as e:
            print(f"Error running task in thread: {str(e)}")
            traceback.print_exc()
        finally:
            async_loop.close()
    
    # Start task in a background thread
    thread = threading.Thread(target=run_async_task_in_thread)
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
        # Mark the task for cancellation
        running_tasks[task_id]['stop'] = True
        
        # Wait a short time to ensure cancellation message is processed
        time.sleep(0.5)
        
        # Remove the task from running tasks to reset state
        running_tasks.pop(task_id, None)
        
        # Emit a cancelled event via Socket.IO
        socketio.emit('task_cancelled', {
            'task_id': task_id,
            'message': 'Task cancelled successfully'
        })
        
        return jsonify({
            'status': 'success', 
            'message': 'Task cancelled successfully'
        })
    
    return jsonify({
        'status': 'error', 
        'message': 'Task not found'
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
    try:
        results_dir = os.path.join(os.getcwd(), 'results')
        file_path = os.path.join(results_dir, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': 'File not found'}), 404
            
        # Check if file is in the results directory
        if not os.path.abspath(file_path).startswith(os.path.abspath(results_dir)):
            return jsonify({'status': 'error', 'message': 'Invalid file path'}), 403
            
        # Send the file as an attachment
        return send_from_directory(
            results_dir,
            filename,
            as_attachment=True,
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/diagnostics')
def diagnostics():
    """Diagnostic page to check system settings"""
    # Check Chrome path
    chrome_path = os.environ.get("CHROME_PATH", "").strip('"\'')
    chrome_exists = False
    chrome_is_file = False
    directory_contents = []
    
    if chrome_path:
        chrome_exists = os.path.exists(chrome_path)
        chrome_is_file = os.path.isfile(chrome_path)
        
        # Get directory listing
        try:
            parent_dir = os.path.dirname(chrome_path)
            if os.path.exists(parent_dir):
                directory_contents = os.listdir(parent_dir)[:10]  # Limit to 10 files
        except Exception as e:
            directory_contents = [f"Error: {str(e)}"]
    
    # Get all environment variables
    env_vars = {}
    for key, value in os.environ.items():
        # Only include relevant variables, not system ones
        if key.startswith(('OPENAI_', 'CHROME_', 'DEBUG', 'TIMEOUT')):
            env_vars[key] = value
    
    # Check if .env file exists and read it
    env_file_exists = os.path.exists('.env')
    env_file_content = ""
    if env_file_exists:
        try:
            with open('.env', 'r') as f:
                env_file_content = f.read()
        except Exception as e:
            env_file_content = f"Error reading .env file: {str(e)}"
    
    # Return diagnostic information
    return jsonify({
        'chrome_path': chrome_path,
        'chrome_exists': chrome_exists,
        'chrome_is_file': chrome_is_file,
        'directory_contents': directory_contents,
        'env_vars': env_vars,
        'env_file_exists': env_file_exists,
        'env_file_content': env_file_content
    })

@app.route('/delete_result/<filename>', methods=['POST'])
def delete_result(filename):
    """Delete a result file"""
    results_dir = os.path.join(os.getcwd(), 'results')
    file_path = os.path.join(results_dir, filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        return jsonify({'status': 'error', 'message': 'File not found'})
    
    # Check if file is in the results directory
    if not os.path.abspath(file_path).startswith(os.path.abspath(results_dir)):
        return jsonify({'status': 'error', 'message': 'Invalid file path'})
    
    try:
        # Delete file
        os.remove(file_path)
        return jsonify({'status': 'success', 'message': 'File deleted successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/restart_app', methods=['POST'])
def restart_app():
    """Restart the Flask application and browser agent"""
    try:
        # Start a background thread to handle the actual restart
        # This prevents blocking the HTTP response
        restart_thread = threading.Thread(target=perform_restart)
        restart_thread.daemon = True
        restart_thread.start()
        
        # Return success immediately
        return jsonify({
            'status': 'success',
            'message': 'Application restart initiated'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error initiating restart: {str(e)}'
        })

def perform_restart():
    """Perform the actual restart operations in a background thread"""
    try:
        # Mark all running tasks for cancellation
        task_ids = list(running_tasks.keys())
        for task_id in task_ids:
            if task_id in running_tasks:
                running_tasks[task_id]['stop'] = True
                
                # Emit cancellation event via Socket.IO
                socketio.emit('task_cancelled', {
                    'task_id': task_id,
                    'message': 'Task cancelled due to application restart'
                })
        
        # Wait a moment for tasks to be cancelled properly
        # This is now safe since we're in a background thread
        time.sleep(1)
        
        # Clear any remaining running tasks
        running_tasks.clear()
        
        # Clean up browser resources before reinitializing
        try:
            # Close any active browser instances
            browser_agent.cleanup_resources()
        except Exception as e:
            print(f"Warning: Error cleaning up browser resources: {str(e)}")
        
        # Reinitialize the browser agent
        browser_agent.reinitialize()
        
        # Emit a restart complete event
        socketio.emit('app_restarted', {
            'status': 'success',
            'message': 'Application restarted successfully'
        })
        
        print("Application restart completed successfully")
        
    except Exception as e:
        print(f"Error during application restart: {str(e)}")
        # Emit error event
        socketio.emit('app_restart_error', {
            'status': 'error',
            'message': f'Error during application restart: {str(e)}'
        })

@app.route('/task_status', methods=['GET'])
def get_task_status():
    """Get the status of all running tasks or a specific task"""
    task_id = request.args.get('task_id')
    
    # If a specific task ID is provided, return that task's status
    if task_id and task_id in running_tasks:
        elapsed_time = time.time() - running_tasks[task_id]['start_time']
        return jsonify({
            'status': 'running',
            'task_id': task_id,
            'prompt': running_tasks[task_id]['prompt'],
            'format_prompt': running_tasks[task_id]['format_prompt'],
            'elapsed_time': elapsed_time,
            'logs': running_tasks[task_id]['logs'],
            'started_at': running_tasks[task_id]['start_time']
        })
    
    # If no task ID or task not found, return all active tasks
    active_tasks = {}
    for tid, task_info in running_tasks.items():
        active_tasks[tid] = {
            'prompt': task_info['prompt'],
            'format_prompt': task_info.get('format_prompt', ''),
            'elapsed_time': time.time() - task_info['start_time'],
            'started_at': task_info['start_time']
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
        # Mark the task for graceful termination
        running_tasks[task_id]['end'] = True
        
        # Wait a short time to ensure the end message is processed
        time.sleep(0.5)
        
        # Emit an event to notify the client
        socketio.emit('task_ending', {
            'task_id': task_id,
            'message': 'Task ending - preserving current progress'
        })
        
        return jsonify({
            'status': 'success', 
            'message': 'Task ending requested'
        })
    
    return jsonify({
        'status': 'error', 
        'message': 'Task not found'
    })

@app.route('/refresh_ollama_models', methods=['POST'])
def refresh_ollama_models():
    """Refresh the list of available Ollama models."""
    data = request.get_json()
    
    # Get API URL from request or use default
    api_url = data.get('api_url') if data and 'api_url' in data else None
    if not api_url:
        api_url = os.environ.get('OLLAMA_API_URL', 'http://localhost:11434')
    
    # Get models using the provided URL
    models = get_ollama_models(api_url)
    
    return jsonify({
        'status': 'success',
        'models': models,
        'count': len(models)
    })

async def start_task_with_socketio(task_id, task_prompt, format_prompt=''):
    """Run the browser task and emit socket events for progress updates."""
    global running_tasks
    
    def log_callback(message):
        # Log to console for debugging
        print(f"[{task_id}] {message}")
        
        # Update the task log
        if task_id in running_tasks:
            running_tasks[task_id]['logs'].append(message)
            
            # Track recent actions for pattern detection (keep last 10)
            running_tasks[task_id]['last_actions'].append(message)
            if len(running_tasks[task_id]['last_actions']) > 10:
                running_tasks[task_id]['last_actions'].pop(0)
            
            # Detect step numbers in the log message
            import re
            step_match = re.search(r'Step (\d+):', message)
            action_match = re.search(r'(clicking|typing|navigating|searching|reading|looking)', message.lower())
            
            # Check for repetitive actions that might indicate being stuck
            is_stuck = False
            reason = ""
            
            if step_match:
                step_number = step_match.group(1)
                current_step = running_tasks[task_id]['current_step']
                
                # If this is a new step, reset the counter
                if step_number != current_step:
                    running_tasks[task_id]['current_step'] = step_number
                    running_tasks[task_id]['step_tracker'][step_number] = 1
                    running_tasks[task_id]['last_step_change_time'] = time.time()
                else:
                    # Increment counter for the repeated step
                    if step_number in running_tasks[task_id]['step_tracker']:
                        running_tasks[task_id]['step_tracker'][step_number] += 1
                    else:
                        running_tasks[task_id]['step_tracker'][step_number] = 1
                    
                    # Check if step has been repeated more than 5 times
                    if running_tasks[task_id]['step_tracker'][step_number] >= 5:
                        is_stuck = True
                        reason = f"repeated Step {step_number} more than 5 times"
            
            # Check for repeating the exact same action multiple times
            if len(running_tasks[task_id]['last_actions']) >= 3:
                last_3_actions = running_tasks[task_id]['last_actions'][-3:]
                if last_3_actions[0] == last_3_actions[1] == last_3_actions[2]:
                    is_stuck = True
                    reason = "repeated the exact same action 3 times in a row"
            
            # Check for error messages that might indicate being stuck
            error_keywords = ["failed", "error", "cannot", "unable to", "not found", "doesn't exist", "timeout"]
            if any(keyword in message.lower() for keyword in error_keywords) and action_match:
                # If we see an error with an action, increment the counter for potential stuck state
                running_tasks[task_id]['unstuck_attempts'] += 0.5
                if running_tasks[task_id]['unstuck_attempts'] >= 3:
                    is_stuck = True
                    reason = "encountered multiple errors while trying to perform actions"
            
            # If we determine the agent is stuck, send an intervention
            if is_stuck:
                # Check if it's been at least 30 seconds since the last move-on prompt
                time_since_change = time.time() - running_tasks[task_id]['last_step_change_time']
                if time_since_change > 30:
                    # Get the current step number
                    step_number = running_tasks[task_id]['current_step'] or "unknown"
                    
                    # Send a prompt to move on
                    move_on_message = f"Agent appears to be stuck on Step {step_number} ({reason}). Prompting it to move on..."
                    print(f"Task {task_id}: {move_on_message}")
                    running_tasks[task_id]['logs'].append(move_on_message)
                    
                    # Reset the step change time to avoid spamming move-on prompts
                    running_tasks[task_id]['last_step_change_time'] = time.time()
                    running_tasks[task_id]['unstuck_attempts'] += 1
                    
                    # Reset step counter for this step
                    if step_number in running_tasks[task_id]['step_tracker']:
                        running_tasks[task_id]['step_tracker'][step_number] = 0
                    
                    # Modify the task prompt to include instructions to move on
                    original_prompt = running_tasks[task_id]['prompt']
                    if not running_tasks[task_id]['modified_prompt']:
                        # First time getting stuck, append instructions
                        unstuck_instruction = f"\n\nIMPORTANT: If you find yourself stuck or repeating the same step multiple times, please try a different approach or move on to the next step. Don't get trapped in loops."
                        running_tasks[task_id]['modified_prompt'] = original_prompt + unstuck_instruction
                    
                    # Create a direct intervention message for the current step
                    # Make the message more urgent based on how many times we've been stuck
                    urgency = "NOW" if running_tasks[task_id]['unstuck_attempts'] <= 1 else "IMMEDIATELY"
                    
                    intervention_message = f"""
SYSTEM MESSAGE: You appear to be stuck on Step {step_number} because you have {reason}. Please try one of the following:
1. Move on to Step {int(step_number) + 1 if step_number.isdigit() else "the next step"}
2. Try a completely different approach to complete this step
3. If the issue is with a UI element, try finding an alternative way to accomplish the task
4. Skip this part of the task if it's not essential

DO NOT continue repeating the same actions. Change your approach {urgency}.
"""
                    
                    # Emit the move-on prompt via socket
                    socketio.emit('task_log', {
                        'task_id': task_id,
                        'message': move_on_message,
                        'timestamp': time.time()
                    })
                    
                    # Emit the intervention message as a separate log to show it prominently
                    socketio.emit('task_log', {
                        'task_id': task_id,
                        'message': intervention_message,
                        'timestamp': time.time()
                    })
                    
                    # Return the intervention message to be sent to the agent
                    return intervention_message
            
            # Emit the log via socket
            socketio.emit('task_log', {
                'task_id': task_id,
                'message': message,
                'timestamp': time.time()
            })
            
            # Return None for normal logs, no special action needed
            return None
    
    try:
        # Create task entry
        running_tasks[task_id] = {
            'prompt': task_prompt,
            'format_prompt': format_prompt,
            'start_time': time.time(),
            'stop': False,
            'end': False,  # Flag for graceful termination
            'logs': [],
            'step_tracker': {},  # Track step repetitions
            'current_step': None,
            'last_step_change_time': time.time(),
            'modified_prompt': None,  # Store a modified prompt if we need to add unstuck instructions
            'unstuck_attempts': 0,    # Count how many times we've tried to get unstuck
            'last_actions': [],       # Track recent actions to detect repetitive patterns
            'partial_results': None   # Store partial results
        }
        
        # Custom log handler that emits socket events
        log_handler = log_callback
        
        try:
            log_handler("Starting browser task...")
            # Run the agent task - use the current event loop
            result = await browser_agent.run_task(
                running_tasks[task_id]['modified_prompt'] or running_tasks[task_id]['prompt'], 
                log_callback=log_handler, 
                stop_check=lambda: task_id not in running_tasks or running_tasks[task_id]['stop'],
                end_check=lambda: task_id in running_tasks and running_tasks[task_id]['end'],
                format_prompt=running_tasks[task_id]['format_prompt']
            )
            
            # Save result as a Word document
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"result_{timestamp}.docx"
            result_path = save_result_as_docx(
                running_tasks[task_id]['prompt'], 
                result, 
                running_tasks[task_id]['logs'], 
                filename, 
                running_tasks[task_id]['format_prompt']
            )
            
            # Check if the task was gracefully ended
            was_ended = task_id in running_tasks and running_tasks[task_id]['end']
            
            # Emit appropriate task completion event
            if was_ended:
                socketio.emit('task_complete', {
                    'task_id': task_id,
                    'success': True,
                    'result': result,
                    'filename': filename,
                    'elapsed_time': time.time() - running_tasks[task_id]['start_time'],
                    'status': 'ended'  # Mark this as an ended task with partial results
                })
            else:
                socketio.emit('task_complete', {
                    'task_id': task_id,
                    'success': True,
                    'result': result,
                    'filename': filename,
                    'elapsed_time': time.time() - running_tasks[task_id]['start_time']
                })
            
        except Exception as e:
            error_message = str(e)
            log_handler(f"Error: {error_message}")
            
            # Emit task error event
            socketio.emit('task_error', {
                'task_id': task_id,
                'error': error_message,
                'elapsed_time': time.time() - running_tasks[task_id]['start_time']
            })
            
        except Exception as e:
            print(f"Error in start_task_with_socketio: {str(e)}")
            traceback.print_exc()
            socketio.emit('task_error', {
                'task_id': task_id,
                'error': str(e),
                'elapsed_time': 0
            })
    
    finally:
        # If the task still exists in running_tasks, handle its state
        if task_id in running_tasks:
            # If the task was marked for cancellation, emit a cancellation event
            if running_tasks[task_id].get('stop', False):
                socketio.emit('task_cancelled', {
                    'task_id': task_id,
                    'elapsed_time': time.time() - running_tasks[task_id]['start_time']
                })
            # Remove the task from running tasks
            running_tasks.pop(task_id, None)

if __name__ == '__main__':
    # Ensure results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Run the Flask app with Socket.IO, making it accessible on the local network
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)

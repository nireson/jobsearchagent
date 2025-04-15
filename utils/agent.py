import os
import asyncio
import time
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from browser_use import Agent, Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig
from dotenv import load_dotenv
import re

class BrowserAgent:
    """A class to manage browser-use agent interactions."""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize the LLM
        self.initialize_llm()
    
    def initialize_llm(self):
        """Initialize the language model based on environment settings."""
        # Get the model provider from environment or use default
        model_provider = os.environ.get("MODEL_PROVIDER", "openai")
        
        if model_provider == "anthropic":
            # Initialize Anthropic model
            model_name = os.environ.get("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key not found in environment variables")
            
            self.llm = ChatAnthropic(
                model=model_name,
                anthropic_api_key=api_key,
                temperature=0.7
            )
            self.model_name = model_name
        elif model_provider == "ollama":
            # Initialize Ollama model
            from langchain.llms import Ollama
            from langchain.chat_models import ChatOllama
            
            model_name = os.environ.get("OLLAMA_MODEL", "llama3")
            api_base = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")
            
            # Create the ChatOllama model
            self.llm = ChatOllama(
                model=model_name,
                base_url=api_base,
                temperature=0.7
            )
            self.model_name = model_name
        else:
            # Initialize OpenAI model (default)
            model_name = os.environ.get("OPENAI_MODEL", "gpt-4o")
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables")
            
            # Special handling for different model families
            model_params = {
                "model_name": model_name,
                "openai_api_key": api_key,
                "temperature": 0.7
            }
            
            # Update the model parameters for specific model types if needed
            if model_name.startswith(("o1-", "o3-")):
                # Add any specific settings needed for O-series models
                pass
            
            # Initialize the OpenAI model with appropriate parameters
            self.llm = ChatOpenAI(**model_params)
            self.model_name = model_name
    
    def reinitialize(self):
        """Reinitialize the agent with updated environment variables."""
        # Reload environment variables
        load_dotenv()
        
        # Reinitialize the LLM
        self.initialize_llm()
    
    def cleanup_resources(self):
        """Clean up any active browser resources to prevent memory leaks."""
        # We don't store browser instances directly in this class,
        # but we can implement a cleanup mechanism for any resources
        # that might be created during task execution.
        try:
            # In a future implementation, this could close any stored
            # browser instances or other resources
            pass
        except Exception as e:
            print(f"Error cleaning up browser resources: {str(e)}")
    
    async def format_result(self, raw_result, format_prompt, log_callback=None):
        """
        Format a result based on the format prompt using the LLM.
        
        Args:
            raw_result: The raw result from the browser task
            format_prompt: Instructions on how to format the result
            log_callback: Function to call with log messages
            
        Returns:
            The formatted result
        """
        if not format_prompt:
            return raw_result
            
        if log_callback:
            log_callback("Formatting result according to format prompt...")
            
        try:
            # Create a prompt template for formatting
            template = """
            You are an expert in formatting and structuring information. Your task is to format the provided content according to the specified format instructions.

            Here is the raw content to format:
            
            {raw_result}
            
            Please format this content according to these instructions:
            
            {format_prompt}
            
            Important formatting guidelines:
            1. Use clear headings and subheadings where appropriate
            2. Format lists with bullet points or numbers
            3. Include clickable links where URLs are present
            4. Use proper spacing and paragraph breaks
            5. Make the content human-readable and well-structured
            6. Remove any machine-readable formats (JSON, XML, etc.) and convert to natural text
            7. Preserve any important data while making it readable
            
            Return only the formatted result, ready for display in a document.
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            
            # Create a chain
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            # Run the chain
            formatted_result = await chain.arun(
                raw_result=raw_result,
                format_prompt=format_prompt
            )
            
            if log_callback:
                log_callback("Result formatting completed")
                
            return formatted_result
            
        except Exception as e:
            if log_callback:
                log_callback(f"Error during result formatting: {str(e)}")
            return raw_result  # Fall back to raw result on error
    
    async def run_task(self, task_description, log_callback=None, stop_check=None, end_check=None, timeout=3000, format_prompt=None):
        """
        Run a browser automation task using browser-use.
        
        Args:
            task_description: Natural language description of what to do
            log_callback: Function to call with log messages
            stop_check: Function that returns True if the task should be stopped
            end_check: Function that returns True if the task should end but preserve partial results
            timeout: Maximum time to wait (in seconds)
            format_prompt: Optional prompt to format the result
            
        Returns:
            The result of the browser task
        """
        # Store partial results if we need to end gracefully
        partial_result = None
        
        if log_callback:
            model_provider = os.environ.get("MODEL_PROVIDER", "openai")
            api_key = os.environ.get(f"{model_provider.upper()}_API_KEY")
            log_callback(f"Starting task with {model_provider} model: {self.model_name}")
            log_callback(f"API key status: {'Set' if api_key else 'Not set'}")
            log_callback("Initializing browser...")
        
        # Create the browser-use agent
        try:
            # Check if Chrome path is configured
            chrome_path = os.environ.get("CHROME_PATH", "").strip('"\'')
            
            # Debug log the chrome path
            if log_callback:
                log_callback(f"Chrome path from env: '{chrome_path}'")
                if chrome_path:
                    log_callback(f"File exists check: {os.path.exists(chrome_path)}")
                    log_callback(f"Is file check: {os.path.isfile(chrome_path)}")
                    
                    # Get directory listing to verify
                    try:
                        parent_dir = os.path.dirname(chrome_path)
                        if os.path.exists(parent_dir):
                            log_callback(f"Parent directory exists: {parent_dir}")
                            files = os.listdir(parent_dir)
                            log_callback(f"Files in directory: {', '.join(files[:5])}{'...' if len(files) > 5 else ''}")
                    except Exception as e:
                        log_callback(f"Error listing directory: {str(e)}")
            
            if chrome_path and os.path.isfile(chrome_path):
                if log_callback:
                    log_callback(f"Using custom Chrome instance at: {chrome_path}")
                
                # Create context configuration with explicit window size
                context_config = BrowserContextConfig(
                    browser_window_size={'width': 1280, 'height': 800},
                    wait_for_network_idle_page_load_time=3.0
                )
                
                # Create browser with explicit configuration
                browser = Browser(
                    config=BrowserConfig(
                        chrome_instance_path=chrome_path,
                        headless=False,  # Show the browser window
                        new_context_config=context_config  # Apply context config
                    )
                )
                
                # Create agent with custom browser
                agent = Agent(
                    task=task_description,
                    llm=self.llm,
                    browser=browser
                )

            else:
                # If chrome path is set but invalid, log a warning
                if chrome_path:
                    if log_callback:
                        log_callback(f"Chrome path not found: '{chrome_path}'. Using default browser.")
                
                # Create context configuration with explicit window size
                context_config = BrowserContextConfig(
                    browser_window_size={'width': 1280, 'height': 800},
                    wait_for_network_idle_page_load_time=3.0
                )
                
                # Create agent with default configuration but non-headless
                agent = Agent(
                    task=task_description,
                    llm=self.llm,
                    browser=Browser(
                        config=BrowserConfig(
                            headless=False,  # Show the browser window
                            new_context_config=context_config  # Apply context config
                        )
                    )
                )

        except Exception as e:
            if log_callback:
                log_callback(f"Error setting up browser: {str(e)}. Using default browser.")
            
            # Fallback to basic agent configuration but non-headless
            agent = Agent(
                task=task_description,
                llm=self.llm,
                browser=Browser(
                    config=BrowserConfig(
                        headless=False  # Show the browser window
                        # No context config in fallback to minimize error chance
                    )
                )
            )
        
        if log_callback:
            log_callback("Browser initialized. Starting task execution...")
        
        # Set up agent event handler to forward logs to our log_callback
        if log_callback and hasattr(agent, 'register_event_handler'):
            def agent_event_handler(event_type, data):
                # Format different types of events for better readability
                if event_type == 'thinking':
                    log_callback(f"🤔 Agent thinking: {data}")
                elif event_type == 'action':
                    if isinstance(data, dict) and 'action' in data:
                        action = data['action']
                        args = data.get('args', {})
                        args_str = ', '.join(f"{k}='{v}'" for k, v in args.items() if k != 'browser') if args else ''
                        log_callback(f"🔄 Action: {action}({args_str})")
                    else:
                        log_callback(f"🔄 Action: {data}")
                elif event_type == 'observation':
                    if isinstance(data, str) and len(data) > 500:
                        # Truncate long observations
                        log_callback(f"👁️ Observation: {data[:500]}... (truncated)")
                    else:
                        log_callback(f"👁️ Observation: {data}")
                elif event_type == 'error':
                    log_callback(f"❌ Error: {data}")
                else:
                    log_callback(f"Event ({event_type}): {data}")
                    
            # Register the event handler with the agent
            try:
                agent.register_event_handler(agent_event_handler)
                log_callback("Registered event handler for detailed agent logs")
            except Exception as e:
                log_callback(f"Note: Could not register event handler: {str(e)}")
        
        # Track start time for timeout handling
        start_time = time.time()
        
        # Define a function to process the raw result
        def process_result(raw_result):
            if isinstance(raw_result, dict):
                # Extract relevant information from the dictionary
                result = raw_result.get('output', '')
                if not result:
                    result = raw_result.get('result', '')
                if not result and 'error' in raw_result:
                    result = f"Error: {raw_result['error']}"
                
                # If we have page content, try to extract meaningful text
                if 'page_content' in raw_result:
                    page_content = raw_result['page_content']
                    # Extract text content between HTML tags
                    text_content = re.sub(r'<[^>]+>', ' ', page_content)
                    # Remove extra whitespace
                    text_content = ' '.join(text_content.split())
                    if text_content:
                        result = text_content
                
                # If we have extracted text, try to clean it up
                if 'extracted_text' in raw_result:
                    extracted_text = raw_result['extracted_text']
                    if isinstance(extracted_text, str):
                        result = extracted_text
                    elif isinstance(extracted_text, (list, dict)):
                        # Try to extract meaningful text from the structure
                        import json
                        try:
                            # If it's JSON-like data, try to extract text fields
                            if isinstance(extracted_text, str):
                                data = json.loads(extracted_text)
                            else:
                                data = extracted_text
                            
                            # Function to recursively extract text from JSON
                            def extract_text_from_json(obj):
                                if isinstance(obj, str):
                                    return obj
                                elif isinstance(obj, (list, tuple)):
                                    return '\n'.join(extract_text_from_json(item) for item in obj if extract_text_from_json(item))
                                elif isinstance(obj, dict):
                                    text_fields = []
                                    for key, value in obj.items():
                                        # Focus on fields likely to contain meaningful text
                                        if key.lower() in ['text', 'title', 'description', 'name', 'message', 'content']:
                                            extracted = extract_text_from_json(value)
                                            if extracted:
                                                text_fields.append(extracted)
                                    return '\n'.join(text_fields)
                                return ''
                            
                            extracted_result = extract_text_from_json(data)
                            if extracted_result:
                                result = extracted_result
                        except:
                            # If JSON parsing fails, try to get string representation
                            result = str(extracted_text)
            
            elif isinstance(raw_result, (list, tuple)):
                # Join list items with newlines
                result = '\n'.join(str(item) for item in raw_result if item)
            else:
                # Convert to string if not already
                result = str(raw_result)
            
            # Clean up the result
            result = result.strip()
            
            # Remove any remaining JSON-like artifacts
            result = re.sub(r'[\{\}\[\]"]', '', result)
            result = re.sub(r'[:,](?!\d)', ' ', result)
            result = re.sub(r'\$type[^,}]+,?', '', result)
            result = re.sub(r'urn:[^\s,}]+', '', result)
            result = re.sub(r'\s+', ' ', result)
            
            return result
        
        try:
            # Periodically check if task should end early
            async def check_end():
                while True:
                    # Wait a short time before checking
                    await asyncio.sleep(2)
                    
                    # Check if we should stop completely
                    if stop_check and stop_check():
                        if log_callback:
                            log_callback("Task cancellation requested")
                        return "Task was cancelled"
                    
                    # Check if we should end gracefully with partial results
                    if end_check and end_check():
                        if log_callback:
                            log_callback("Task ending requested - preserving current progress")
                        
                        # Try to get the current state of the browser
                        try:
                            if hasattr(agent, 'browser') and agent.browser:
                                # Get the current page content if possible
                                if log_callback:
                                    log_callback("Capturing current page state for partial results...")
                                
                                # Get the current URL
                                current_url = None
                                try:
                                    if hasattr(agent, 'browser') and hasattr(agent.browser, 'current_url'):
                                        current_url = agent.browser.current_url
                                        if current_url and log_callback:
                                            log_callback(f"Current URL: {current_url}")
                                except Exception as e:
                                    if log_callback:
                                        log_callback(f"Error getting current URL: {str(e)}")
                                
                                # Get the current page content if possible
                                try:
                                    # Get page content through the browser
                                    page_content = None
                                    if hasattr(agent.browser, 'page') and agent.browser.page:
                                        page_content = await agent.browser.page.content()
                                        if log_callback:
                                            log_callback("Successfully captured page content")
                                except Exception as e:
                                    if log_callback:
                                        log_callback(f"Error capturing page content: {str(e)}")
                                
                                # This is a basic example - in a real implementation,
                                # you would want to extract more complete state from the browser
                                nonlocal partial_result
                                partial_result = {
                                    'output': "Task ended before completion. Partial results:",
                                    'page_content': page_content or "Task was ended by user before completion. These are partial results.",
                                    'extracted_text': f"PARTIAL RESULTS (task ended at step {agent.current_step if hasattr(agent, 'current_step') else 'unknown'})"
                                }
                                
                                if current_url:
                                    partial_result['extracted_text'] += f"\nCurrent URL: {current_url}"
                                
                                return True
                        except Exception as e:
                            if log_callback:
                                log_callback(f"Error capturing partial results: {str(e)}")
                    
                    # Check if we've exceeded the timeout
                    if time.time() - start_time > timeout:
                        if log_callback:
                            log_callback(f"Task timeout exceeded ({timeout} seconds)")
                        return True
            
            # Start the end-check task
            end_check_task = asyncio.create_task(check_end())
            
            # Run the agent task with minimal parameters
            agent_task = asyncio.create_task(agent.run())
            
            # Wait for either the agent to finish or the end check to trigger
            done, pending = await asyncio.wait(
                [agent_task, end_check_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
            
            # Get the result
            raw_result = None
            for task in done:
                result = task.result()
                if result is True or isinstance(result, str):
                    # This is from the end_check task
                    if result is True and partial_result:
                        raw_result = partial_result
                    elif isinstance(result, str):
                        return result
                else:
                    # This is from the agent task
                    raw_result = result
            
            # If we have a partial result but no raw_result, use the partial result
            if not raw_result and partial_result:
                raw_result = partial_result
                if log_callback:
                    log_callback("Using partial results captured before task ended")
            
            # Log the raw result for debugging
            if log_callback and raw_result:
                if isinstance(raw_result, dict):
                    log_callback("Raw result received (showing keys): " + ", ".join(raw_result.keys()))
                else:
                    log_callback(f"Raw result type: {type(raw_result)}")
            
            # Process the result
            result = process_result(raw_result)
            
            # Format the result if a format prompt is provided
            if format_prompt and result:
                if log_callback:
                    log_callback("Task completed. Formatting result...")
                formatted_result = await self.format_result(result, format_prompt, log_callback)
                result = formatted_result
            else:
                if log_callback:
                    log_callback("Task completed successfully!")
                    
            return result
            
        except asyncio.CancelledError:
            if log_callback:
                log_callback(f"Task was cancelled after {int(time.time() - start_time)} seconds")
            
            # If we have partial results, use them
            if partial_result:
                result = process_result(partial_result)
                if log_callback:
                    log_callback("Using partial results captured before cancellation")
                return f"Task was cancelled, but partial results were saved:\n\n{result}"
            
            return "Task was cancelled"
        except Exception as e:
            if log_callback:
                log_callback(f"Error during task execution: {str(e)}")
                # Log the full traceback for better debugging
                import traceback
                log_callback(f"Traceback: {traceback.format_exc()}")
            return f"Error: {str(e)}"

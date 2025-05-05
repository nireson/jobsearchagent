import os
import asyncio
import time
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from browser_use import Agent, Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig
from dotenv import load_dotenv
import re
from tavily import TavilyClient # Import Tavily client

# Helper function to load LLM - assuming this was present before
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

class BrowserAgent:
    """A class to manage browser-use agent interactions."""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize the LLM
        self.initialize_llm()
        
        # Store the default model settings
        self.default_model = None
        if hasattr(self, 'model_name'):
            self.default_model = self.model_name
    
    def initialize_llm(self):
        """Initialize the language model based on environment settings."""
        model_provider = os.environ.get("MODEL_PROVIDER", "openai").lower()
        
        if model_provider == "anthropic":
            model_name = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key: raise ValueError("Anthropic API key not found...")
            self.llm = ChatAnthropic(model=model_name, anthropic_api_key=api_key, temperature=0.7)
            self.model_name = model_name
            
        elif model_provider == "ollama" and OLLAMA_AVAILABLE:
            model_name = os.environ.get("OLLAMA_MODEL", "llama3")
            api_base = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")
            self.llm = ChatOllama(model=model_name, base_url=api_base, temperature=0.7)
            self.model_name = model_name
            
        else: # Default to OpenAI
            if model_provider == "ollama" and not OLLAMA_AVAILABLE:
                print("Warning: Ollama provider selected but langchain_ollama not installed. Falling back to OpenAI.")
            model_name = os.environ.get("OPENAI_MODEL", "gpt-4o")
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key: raise ValueError("OpenAI API key not found...")
            model_params = {"model_name": model_name, "openai_api_key": api_key, "temperature": 0.7}
            self.llm = ChatOpenAI(**model_params)
            self.model_name = model_name
    
    def reinitialize(self):
        load_dotenv(override=True)
        self.initialize_llm()
        self.default_model = self.model_name # Update default after reinitialization
    
    def cleanup_resources(self):
        # No specific resources stored directly in this class currently
        pass
    
    async def format_result(self, raw_result, format_prompt, log_callback=None):
        if not format_prompt or not raw_result:
            return raw_result
        try:
            if log_callback: log_callback("Formatting result...")
            content_to_format = raw_result
            if isinstance(raw_result, dict):
                if 'output' in raw_result: content_to_format = raw_result['output']
                elif 'result' in raw_result: content_to_format = raw_result['result']
                elif 'extracted_text' in raw_result: content_to_format = raw_result['extracted_text']
                elif 'page_content' in raw_result: content_to_format = raw_result['page_content']
            
            system_template = "You are an expert at formatting..."
            human_template = "Here is the content...\n{content}\n\nPlease format...\n{format_instructions}\n\nOnly return formatted content..."
            chat_prompt = ChatPromptTemplate.from_messages([("system", system_template), ("human", human_template)])
            chain = chat_prompt | self.llm | StrOutputParser()
            formatted_result = await chain.ainvoke({"content": content_to_format, "format_instructions": format_prompt})
            if log_callback: log_callback("Successfully formatted the result.")
            return formatted_result
        except Exception as e:
            if log_callback: log_callback(f"Error formatting result: {str(e)}")
            return raw_result
    
    def set_model_for_task(self, model_name):
        model_provider = os.environ.get("MODEL_PROVIDER", "openai").lower()
        try:
            if model_provider == "anthropic":
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key: return False
                self.llm = ChatAnthropic(model=model_name, anthropic_api_key=api_key, temperature=0.7)
            elif model_provider == "ollama" and OLLAMA_AVAILABLE:
                api_base = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")
                self.llm = ChatOllama(model=model_name, base_url=api_base, temperature=0.7)
            else: # OpenAI default
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key: return False
                self.llm = ChatOpenAI(model_name=model_name, openai_api_key=api_key, temperature=0.7)
            self.model_name = model_name
            return True
        except Exception as e:
            print(f"Error setting custom model: {str(e)}")
            return False
    
    def reset_model(self):
        if self.default_model and self.model_name != self.default_model:
            print(f"Resetting model from {self.model_name} to default: {self.default_model}")
            return self.set_model_for_task(self.default_model)
        return False # Return False if no reset was needed or default wasn't set

    async def run_task(self, task_description, log_callback=None, stop_check=None, end_check=None, format_prompt=None, max_steps=30, task_specific_model=None, agent_type='browser', run_headless=False):
        """Runs the agent task, handling model switching, agent type, headless mode, and result formatting."""
        # Default log/stop/end checks
        if log_callback is None: log_callback = lambda msg, level='info': print(f"[{level.upper()}] {msg}")
        if stop_check is None: stop_check = lambda: False
        if end_check is None: end_check = lambda: False

        log_callback(f"Executing task with agent type: {agent_type}")

        # --- Model Selection Logic ---
        original_model = self.model_name 
        use_temporary_model = False
        if task_specific_model and task_specific_model != original_model:
            log_callback(f"Attempting to use task-specific model: {task_specific_model}")
            if self.set_model_for_task(task_specific_model):
                log_callback(f"Successfully switched to model: {task_specific_model}")
                use_temporary_model = True
            else:
                log_callback(f"Failed to switch... Using default: {original_model}", level='warning')
        else:
            log_callback(f"Using default model: {original_model}")
        # --- End Model Selection Logic ---

        final_task_result = None
        log_callback(f"DEBUG: Inside run_task. Received agent_type = '{agent_type}' BEFORE conditional check.", level='debug') 
        
        try:
            if agent_type == 'tavily':
                log_callback(f"DEBUG: Entering TAVILY agent execution block (agent_type='{agent_type}').", level='debug') 
                log_callback("Tavily agent selected. Using LLM to generate search query...")
                query_generation_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an expert..."),
                    ("human", "Analyze...\n\nUser Request: {user_request}")
                ])
                query_generation_chain = query_generation_prompt | self.llm | StrOutputParser()
                try:
                    generated_query = await query_generation_chain.ainvoke({"user_request": task_description})
                    log_callback(f"LLM generated search query: '{generated_query}'")
                except Exception as llm_err:
                    log_callback(f"Error using LLM... Falling back...", level='warning')
                    generated_query = task_description 
                log_callback("Initializing Tavily client...")
                api_key = os.environ.get("TAVILY_API_KEY")
                if not api_key: raise ValueError("Tavily API key is required.")
                tavily_client = TavilyClient(api_key=api_key)
                log_callback("Tavily client initialized.")
                log_callback(f"Performing Tavily search... '{generated_query[:100]}...'")
                search_result = tavily_client.search(query=generated_query, search_depth="advanced", max_results=7)
                log_callback("Tavily search completed.")
                if 'results' in search_result and search_result['results']:
                    formatted_tavily_result = "Search Results:\n\n"
                    for i, result in enumerate(search_result['results']):
                        formatted_tavily_result += f"{i+1}. {result.get('title', 'N/A')}\n   URL: {result.get('url', 'N/A')}\n   Content: {result.get('content', 'N/A')[:500]}...\n\n"
                    final_task_result = formatted_tavily_result
                else:
                     final_task_result = "Tavily search returned no results..."
                     log_callback("Tavily returned no results.", level='warning')
                final_task_result = await self.format_result(final_task_result, format_prompt, log_callback)
            
            else: # Browser Agent Execution
                log_callback(f"DEBUG: Entering BROWSER agent execution block (agent_type='{agent_type}')...", level='debug')
                log_callback("Proceeding with Browser Agent execution.")
                browser = None
                browser_initialized = False
                try:
                    chrome_path = os.environ.get("CHROME_PATH", "").strip('"\'')
                    log_callback(f"Attempting to initialize browser (headless={run_headless}) with CHROME_PATH: '{chrome_path}'", level='debug')
                    
                    if chrome_path and os.path.isfile(chrome_path):
                        log_callback("Valid CHROME_PATH found...")
                        try:
                            browser_config = BrowserConfig(headless=run_headless, browser_binary_path=chrome_path)
                            browser = Browser(config=browser_config)
                            log_callback(f"Browser configured with path and headless={run_headless}.")
                            browser_initialized = True
                        except Exception as e:
                            log_callback(f"Failed initializing BrowserConfig with path and headless={run_headless}: {e}.", level='error')
                    else:
                        if chrome_path:
                             log_callback(f"CHROME_PATH '{chrome_path}' not found or invalid. Using default browser config.", level='warning')
                        else: 
                             log_callback("CHROME_PATH not set. Using default browser config.", level='info')

                    if not browser_initialized:
                        log_callback(f"Using default browser settings (headless={run_headless})...")
                        try:
                            browser_config = BrowserConfig(headless=run_headless)
                            browser = Browser(config=browser_config)
                            log_callback(f"Browser configured with default settings (headless={run_headless}).")
                            browser_initialized = True 
                        except Exception as e:
                            log_callback(f"Failed initializing BrowserConfig with default settings (headless={run_headless}): {e}", level='error')

                except Exception as browser_init_err:
                    log_callback(f"Error during browser initialization attempt: {browser_init_err}", level='error')
                    # Fallback should probably respect headless setting too, but might fail if first try failed
                    log_callback("Attempting basic fallback browser initialization (headless=False)...", level='warning')
                    try:
                        browser_config = BrowserConfig(headless=False) # Fallback to visible for stability
                        browser = Browser(config=browser_config)
                        log_callback("Browser configured with default settings (fallback, headless=False).")
                        browser_initialized = True
                    except Exception as fallback_err:
                         log_callback(f"CRITICAL: Failed to initialize browser even with fallback: {fallback_err}", level='error')
                         raise ValueError(f"Failed to initialize browser. Details: {fallback_err}") 
                if not browser_initialized or browser is None:
                    log_callback("CRITICAL: Browser initialization failed...", level='error')
                    raise ValueError("Browser initialization failed unexpectedly.") 
                
                # --- End Browser Initialization --- 
                
                # --- Define run_agent --- 
                async def run_agent():
                    nonlocal browser 
                    agent_result = None
                    task = None # Initialize task variable
                    try:
                        log_callback("Initializing browser-use Agent...")
                        agent = Agent(
                            task=task_description,
                            llm=self.llm,
                            browser=browser
                        )
                        log_callback("browser-use Agent initialized. Starting run...")
                        
                        task = asyncio.create_task(agent.run(
                            max_steps=max_steps # Pass max_steps
                        ))
                        
                        # Monitor the task and handle stop/end signals
                        while not task.done():
                            if stop_check():
                                log_callback("Stop signal detected, cancelling agent task...")
                                task.cancel()
                                # Wait briefly for cancellation to propagate
                                await asyncio.sleep(0.1) 
                                break # Exit monitoring loop
                            if end_check():
                                log_callback("End signal detected, cancelling agent task...")
                                task.cancel()
                                # Wait briefly for cancellation to propagate
                                await asyncio.sleep(0.1)
                                break # Exit monitoring loop
                            await asyncio.sleep(0.5)

                        # Await the task result or handle exceptions
                        try:
                            agent_result = await task
                            log_callback("Agent run completed normally.")
                        except asyncio.CancelledError:
                            log_callback("Agent run was cancelled.")
                            # We handle the result processing outside this inner try
                            agent_result = "Task Cancelled" 
                            # No need to re-raise here, let the outer logic handle it
                        except Exception as agent_run_err:
                            log_callback(f"Error during agent.run(): {agent_run_err}", level='error')
                            agent_result = f"Error during agent execution: {agent_run_err}"
                        
                        # Process the result (even if cancelled or errored)
                        def process_result(agent_history):
                            """Extracts the final result from the AgentHistoryList or error message."""
                            # Handle explicit cancellation/error messages first
                            if isinstance(agent_history, str) and (agent_history == "Task Cancelled" or agent_history.startswith("Error during")):
                                return agent_history
                                
                            if not agent_history: return "Agent returned no result."
                            
                            # Check if it has the expected structure (duck typing)
                            if hasattr(agent_history, 'extracted_content') and callable(agent_history.extracted_content):
                                extracted = agent_history.extracted_content()
                                if extracted and isinstance(extracted, list) and len(extracted) > 0:
                                    final_answer = extracted[-1]
                                    if isinstance(final_answer, dict) and 'text' in final_answer:
                                        return final_answer['text']
                                    return str(final_answer)
                                elif hasattr(agent_history, 'all_results') and isinstance(agent_history.all_results, list) and len(agent_history.all_results) > 0:
                                     last_action_result = agent_history.all_results[-1]
                                     if hasattr(last_action_result, 'extracted_content') and last_action_result.extracted_content:
                                         return str(last_action_result.extracted_content)
                            
                            log_callback("Warning: Could not extract specific final result from AgentHistoryList. Using string representation.", level='warning')
                            return str(agent_history)
                        
                        processed_result = process_result(agent_result)
                        final_result = await self.format_result(processed_result, format_prompt, log_callback)
                        return final_result
                        
                    except Exception as e:
                        # Catch errors from agent initialization or result processing
                        log_callback(f"Error in run_agent scope: {str(e)}", level='error')
                        import traceback
                        traceback.print_exc() 
                        # Ensure a result is assigned even on error
                        return f"Error encountered: {str(e)}"
                    finally:
                        # This finally block ensures browser cleanup happens after task completion/cancellation/error
                        log_callback("Cleaning up browser instance within run_agent...")
                        if browser:
                            try:
                                await browser.close()
                                log_callback("Browser closed successfully.")
                            except Exception as close_err:
                                log_callback(f"Error closing browser: {close_err}", level='error')
                        browser = None # Ensure browser is marked as closed
                # --- End define run_agent --- 
                
                # Execute run_agent and capture its result
                final_task_result = await run_agent()
        
        except asyncio.CancelledError:
            # This catches cancellation triggered from OUTSIDE run_agent (e.g., from start_task_with_socketio)
            log_callback("Task run was cancelled (outer catch). Handling final result.")
            final_task_result = "Task Cancelled"
        except Exception as e:
            log_callback(f"Error executing task (outer catch): {str(e)}", level='critical')
            final_task_result = f"Critical Error: {str(e)}"
        finally:
            # Reset the model if a temporary one was used
            if use_temporary_model:
                log_callback(f"Resetting model back to default: {original_model}") # Added log for clarity
                self.reset_model()
        
        return final_task_result

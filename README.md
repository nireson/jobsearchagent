# Browser Agent UI

A Flask web application that provides a user interface for the browser-use agent. This application allows you to:

- Submit natural language prompts for browser automation tasks
- View real-time logs as the agent works
- Save results to Word documents
- Manage API keys through a settings page
- Browse and download past results

## Installation

1. Clone the repository or download the source code
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
5. Install Playwright browsers (required for browser-use):
   ```
   playwright install chromium
   ```

## Configuration

1. Create a `.env` file in the project root directory
2. Add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_MODEL=gpt-4o
   TIMEOUT=300
   DEBUG=False
   ```

Alternatively, you can configure these settings through the Settings page in the application.

## Running the Application

1. Activate your virtual environment (if not already activated)
2. Run the Flask application:
   ```
   python app.py
   ```
3. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## Features

### Home Page
- Submit browser automation tasks with natural language prompts
- View real-time logs as the agent executes the task
- Cancel running tasks
- See the final result and download it as a Word document

### Results Page
- Browse all past results
- Download result documents
- Delete unwanted results

### Settings Page
- Configure API keys
- Set OpenAI model selection
- Adjust task timeout settings

## Project Structure

```
browser_agent_app/
├── app.py                  # Main Flask application
├── config.py               # Configuration settings
├── requirements.txt        # Project dependencies
├── .env                    # Environment variables (API keys)
├── static/                 # Static files
│   ├── css/
│   │   └── styles.css      # Custom CSS
│   └── js/
│       └── main.js         # JavaScript for UI interactivity
├── templates/              # HTML templates
│   ├── base.html           # Base template
│   ├── index.html          # Homepage with prompt input
│   ├── settings.html       # API key management
│   └── results.html        # Results browser
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── agent.py            # Browser agent functionality
│   ├── document.py         # Document generation utilities
│   └── env_manager.py      # .env file management
└── results/                # Directory for saved results
```

## Dependencies

- Flask - Web framework
- Flask-SocketIO - Real-time communication
- browser-use - Browser automation library
- LangChain - LLM orchestration
- python-docx - Word document generation
- python-dotenv - Environment variable management

## Troubleshooting

- **Browser opens but task gets stuck**: This might be due to the website having anti-bot measures. Try adjusting the timeout in settings.
- **API key issues**: Make sure your OpenAI API key is valid and has sufficient credits.
- **Socket connection errors**: Check your browser console for errors. The application uses WebSockets for real-time communication.

## License

MIT License

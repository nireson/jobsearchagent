# Browser Agent UI

A Flask web application that provides a user interface for the browser-use agent. This application allows you to:

- Submit natural language prompts for browser automation tasks
- View real-time logs as the agent works
- Save results to Word documents
- Manage API keys through a settings page
- Browse and download past results

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd jobsagent2
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
- Create a `.env` file in the project root
- Set required API keys and configuration (see GETTING_STARTED.md)

## Usage

1. Start the application:
```bash
python app.py
```

2. Access the web interface at `http://localhost:5000`

3. Configure your settings in the Settings page

4. Enter your research prompt and optional format instructions

5. Start the task and monitor progress

6. Download or manage results from the Results page

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

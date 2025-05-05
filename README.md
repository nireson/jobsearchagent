# AOTT AI Research Agent

A powerful web-based AI research agent that automates browser tasks and generates formatted results. This application allows users to perform automated research tasks using AI-powered browser automation and receive well-formatted results in Word documents.

## Features

- **AI-Powered Browser Automation**: Automate web research tasks using advanced AI models
- **Multiple AI Model Support**: Compatible with OpenAI, Anthropic, and Ollama models
- **Real-time Task Monitoring**: Track task progress with live updates
- **Formatted Results**: Generate well-structured Word documents with formatted content
- **Task Management**: Start, stop, and end tasks with partial results preservation
- **File Management**: Download and manage result documents
- **Settings Configuration**: Configure API keys and model settings
- **Diagnostic Tools**: System diagnostics and environment variable management

## Tech Stack

- **Backend**: Python, Flask, Socket.IO
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **AI Integration**: OpenAI API, Anthropic API, Ollama
- **Document Processing**: python-docx
- **Browser Automation**: Custom browser-use agent

## Requirements

- Python 3.8+
- Chrome browser
- OpenAI API key (for OpenAI models)
- Anthropic API key (for Anthropic models)
- Ollama (for local model support)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nireson/jobsearchagent
cd jobsearchagent
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
jobsagent2/
├── app.py                 # Main Flask application
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── static/              # Static assets
│   ├── css/            # Stylesheets
│   └── js/             # JavaScript files
├── templates/          # HTML templates
├── utils/             # Utility modules
│   ├── agent.py       # Browser agent implementation
│   ├── document.py    # Document processing
│   └── env_manager.py # Environment management
└── results/           # Generated result documents
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Specify your license here]

## Support

For support, please [open an issue](<repository-url>/issues) or contact the maintainers.

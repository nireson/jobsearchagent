# Job Search Agent

A Flask web application that helps automate job searching and application processes. This application allows you to:

- Submit natural language prompts for job search tasks
- Use multiple AI models (OpenAI, Anthropic, Ollama) for intelligent processing
- Generate well-formatted documents from search results
- View real-time progress as the agent works
- Save and manage search results
- Configure multiple API providers and settings

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd jobsagent2
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On Unix/MacOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
- Copy `.env_example` to `.env`
- Set required API keys and configuration (see GETTING_STARTED.md)
- Required API keys include OpenAI, Anthropic (optional), and other service providers

## Usage

1. Start the application:
```bash
python app.py
```

2. Access the web interface at `http://localhost:5000`

3. Configure your settings:
   - Add your API keys in the Settings page
   - Choose your preferred AI models
   - Configure job search preferences

4. Start a job search:
   - Enter your job search criteria
   - Specify any special requirements or filters
   - Monitor the search progress in real-time

5. Review and manage results:
   - View formatted search results
   - Download results as Word documents
   - Access historical searches

## Project Structure

```
jobsagent2/
├── app.py                  # Main Flask application
├── config.py              # Configuration settings
├── requirements.txt       # Project dependencies
├── .env                  # Environment variables (API keys)
├── .env_example          # Example environment configuration
├── static/               # Static files (CSS, JS)
├── templates/            # HTML templates
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── document.py       # Document handling utilities
│   └── other utilities
├── migrations/           # Database migrations
├── instance/            # Instance-specific files
└── results/             # Search results storage
```

## Key Dependencies

- Flask & Flask-SocketIO - Web framework and real-time communication
- LangChain - LLM orchestration and chains
- OpenAI, Anthropic, Ollama - AI model providers
- SQLAlchemy - Database ORM
- python-docx - Word document generation
- pandas - Data processing
- playwright - Web automation

## Features

- Multi-model AI support (GPT-4, Claude, Local models)
- Real-time search progress monitoring
- Structured document generation
- Configurable search parameters
- Result history management
- Multiple API provider support
- Database-backed result storage

## Troubleshooting

- **API Key Issues**: Verify your API keys are correctly set in the `.env` file
- **Model Selection**: Ensure you have access to the selected AI models
- **Document Generation**: Check write permissions in the results directory
- **Database Issues**: Verify the database URL in your configuration

## Getting Started

See GETTING_STARTED.md for detailed setup instructions and examples.

## License

MIT License

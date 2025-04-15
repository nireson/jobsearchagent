# Getting Started with AOTT AI Research Agent

This guide will help you set up and configure the AOTT AI Research Agent for your needs.

## Prerequisites

Before you begin, ensure you have the following:

1. **Python 3.8 or higher**
   ```bash
   python --version
   ```

2. **Chrome Browser**
   - Latest version of Google Chrome installed
   - Chrome executable path configured (see Configuration section)

3. **API Keys** (depending on your chosen model provider)
   - OpenAI API key (for GPT models)
   - Anthropic API key (for Claude models)
   - Ollama (for local model support)

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd jobsagent2
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Required: Choose one model provider
MODEL_PROVIDER=openai  # or anthropic or ollama

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4  # or gpt-3.5-turbo

# Anthropic Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Ollama Configuration
OLLAMA_API_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Chrome Configuration
CHROME_PATH=C:\Program Files\Google\Chrome\Application\chrome.exe  # Windows
# or
CHROME_PATH=/usr/bin/google-chrome  # Linux
# or
CHROME_PATH=/Applications/Google Chrome.app/Contents/MacOS/Google Chrome  # macOS

# Application Settings
DEBUG=False
TIMEOUT=300  # Task timeout in seconds
```

## Running the Application

1. **Start the Server**
   ```bash
   python app.py
   ```

2. **Access the Web Interface**
   - Open your browser and navigate to `http://localhost:5000`
   - The application will be accessible on your local network

## Configuration Guide

### Model Provider Selection

1. **OpenAI (GPT Models)**
   - Requires an OpenAI API key
   - Supports GPT-4 and GPT-3.5 models
   - Recommended for most use cases

2. **Anthropic (Claude Models)**
   - Requires an Anthropic API key
   - Supports Claude 3 models
   - Good for complex reasoning tasks

3. **Ollama (Local Models)**
   - Requires Ollama installed locally
   - Supports various open-source models
   - Good for privacy-sensitive applications

### Chrome Configuration

1. **Windows**
   - Default path: `C:\Program Files\Google\Chrome\Application\chrome.exe`
   - If using a different installation path, update `CHROME_PATH` in `.env`

2. **macOS**
   - Default path: `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`
   - Verify the path exists and is executable

3. **Linux**
   - Default path: `/usr/bin/google-chrome`
   - May need to install Chrome if not present

### Task Configuration

1. **Timeout Settings**
   - Default: 300 seconds (5 minutes)
   - Adjust based on task complexity
   - Set in `.env` as `TIMEOUT=seconds`

2. **Debug Mode**
   - Enable for detailed logging
   - Set `DEBUG=True` in `.env`
   - Useful for troubleshooting

## Using the Application

### 1. Settings Page

- Configure your chosen model provider
- Set API keys
- Adjust Chrome path if needed
- Configure timeout settings

### 2. Main Interface

1. **Enter Research Prompt**
   - Describe your research task
   - Be specific about requirements
   - Include any format preferences

2. **Optional Format Instructions**
   - Specify how you want results formatted
   - Define structure and style preferences
   - Include any specific requirements

3. **Start Task**
   - Monitor progress in real-time
   - View detailed logs
   - Cancel or end task if needed

### 3. Results Management

1. **Download Results**
   - Access from Results page
   - Download as Word documents
   - View creation date and size

2. **Manage Files**
   - Delete unwanted results
   - Refresh results list
   - Sort by date or size

## Troubleshooting

### Common Issues

1. **Chrome Not Found**
   - Verify Chrome installation
   - Check `CHROME_PATH` in `.env`
   - Ensure path is correct for your OS

2. **API Key Issues**
   - Verify API key is valid
   - Check provider selection
   - Ensure proper format in `.env`

3. **Task Timeout**
   - Increase timeout in settings
   - Check network connectivity
   - Verify model availability

4. **Socket Connection Errors**
   - Check browser console
   - Verify network settings
   - Ensure WebSocket support

### Getting Help

1. **Check Logs**
   - Enable debug mode
   - Review application logs
   - Check browser console

2. **Diagnostics Page**
   - Access system diagnostics
   - View environment variables
   - Check Chrome configuration

3. **Support**
   - Open an issue on GitHub
   - Contact maintainers
   - Check documentation

## Best Practices

1. **Task Design**
   - Be specific in prompts
   - Include clear objectives
   - Define success criteria

2. **Format Instructions**
   - Use clear structure
   - Specify important details
   - Include examples if possible

3. **Resource Management**
   - Monitor API usage
   - Clean up old results
   - Regular system checks

4. **Security**
   - Keep API keys secure
   - Regular updates
   - Monitor access logs

# Getting Started with Job Search Agent

This guide will help you set up and start using the Job Search Agent.

## Step 1: Set up your environment

First, make sure you have Python 3.11.x installed (Python 3.12 is not yet supported). You can download Python 3.11 from the [official Python website](https://www.python.org/downloads/release/python-3116/).

Verify your Python version:
```bash
python --version  # Should show Python 3.11.x
```

Then follow these steps:

```bash
# Clone the repository
git clone <repository-url>
cd jobsagent2

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
# First upgrade pip and setuptools
python -m pip install --upgrade pip setuptools

# Then install project dependencies
pip install -r requirements.txt
```

## Step 2: Configure API keys

You'll need API keys for the following services:

1. OpenAI API key (required) - Get from [OpenAI](https://platform.openai.com/api-keys)
2. Anthropic API key (optional) - Get from [Anthropic](https://console.anthropic.com/)
3. Other optional API keys as needed

Create a `.env` file in the project root directory and add your API keys:
```
OPENAI_API_KEY=your_api_key_here
ANTHROPIC_API_KEY=your_api_key_here  # Optional
```

You can also add your API keys through the Settings page after starting the application.

## Step 3: Run the application

```bash
# Make sure your virtual environment is activated
python app.py
```

This will start the Flask development server. You should see output like:
```
* Serving Flask app 'app'
* Debug mode: on
* Running on http://127.0.0.1:5000
```

Open your web browser and navigate to http://127.0.0.1:5000

## Step 4: Using the application

### Home Page
1. Enter your task instructions in the text area
2. Click "Run Task" to start the browser agent
3. Monitor the logs in real-time
4. View the result when the task is complete
5. Download the result as a Word document

Example tasks:
- "Go to Wikipedia and find information about browser automation"
- "Search Google News for the latest articles about AI"
- "Visit reddit.com/r/python and find the most upvoted post this week"

### Results Page
- Browse all past task results
- Download Word documents
- Delete unwanted results

### Settings Page
- Manage your API keys
- Change the OpenAI model (GPT-4o recommended)
- Adjust task timeout

## Step 5: Troubleshooting

If you encounter any issues:

1. **Browser opens but task gets stuck**: This could be due to the website having anti-bot measures. Try a different website.

2. **API key error**: Check that your OpenAI API key is valid and has sufficient credits.

3. **Socket connection issues**: This can happen if the browser agent is taking too long. Try refreshing the page and running a simpler task.

4. **Browser fails to launch**: Make sure you have installed the browser driver correctly with `playwright install chromium`.

For more detailed information, check the logs in the terminal where you're running the Flask application.

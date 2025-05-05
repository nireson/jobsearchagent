# Getting Started with Browser Agent UI

This guide will help you set up and start using the Browser Agent UI.

## Step 1: Set up your environment

First, make sure you have Python 3.8+ installed. Then follow these steps:

```bash
# Clone the repository (if using Git)
git clone https://github.com/yourusername/browser-agent-ui.git
cd browser-agent-ui

# Or simply create a directory for the project files

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install browser driver for browser-use
playwright install chromium
```

## Step 2: Configure API keys

You need an OpenAI API key to use the browser agent:

1. Get your API key from [OpenAI](https://platform.openai.com/api-keys)
2. Create a `.env` file in the project root directory
3. Add your API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

Alternatively, you can add your API key through the Settings page after starting the application.

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

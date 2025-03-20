# ScrapeGraphAI with Groq: Web Scraping Using LLaMA-Compatible API

## Overview
ScrapeGraphAI is a Python library that enables flexible web scraping using Large Language Models (LLMs) and graph-based logic. This guide explains how to configure ScrapeGraphAI with Groq, a high-performance AI inference platform, to scrape Wikipedia for information about the assassination attempt on Caliph Umar bin al-Khattab.

## Prerequisites
Ensure you have the following:
- Python 3.8+
- Internet access
- Basic knowledge of Python and terminal usage
- A Groq API key

## Step 1: Obtain a Groq API Key
1. **Sign Up for Groq**: Visit [Groq Console](https://console.groq.com) and create an account.
2. **Generate an API Key**:
   - Log in and navigate to **API Keys**.
   - Click **Create API Key**, name it (e.g., `ScrapeGraphAI-Key`), and generate it.
   - Copy and securely save the key (e.g., `gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`).
3. **Verify Access**: Check rate limits in Groq Settings (typically 6000 TPM on the free tier).

## Step 2: Set Up Your Python Environment
### Create and Activate a Virtual Environment
```bash
cd D:\Study\Courses\pingPongScript
python -m venv venv
```
#### Windows:
```bash
D:\Study\Courses\pingPongScript\venv\Scripts\activate
```

## Step 3: Install Required Libraries
```bash
pip install scrapegraphai groq
playwright install
```

## Step 4: Create the Python Script
Create a `testing.py` file in your project directory and add the following code:

```python
from groq import Groq
from scrapegraphai.graphs import SmartScraperGraph
import json
import time

# Custom Groq LLM class for ScrapeGraphAI
class GroqLLM:
    def __init__(self, api_key, model="llama-3.3-70b-versatile", tpm_limit=6000):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.tpm_limit = tpm_limit
        self.tokens_used = 0
        self.last_reset = time.time()

    def __call__(self, messages, **kwargs):
        # Reset token count every minute
        current_time = time.time()
        if current_time - self.last_reset >= 60:
            self.tokens_used = 0
            self.last_reset = current_time

        # Normalize messages to list of dictionaries
        if not isinstance(messages, list):
            messages = [{"role": "user", "content": str(messages)}]
        else:
            messages = [
                {"role": msg.get("role", "user"), "content": msg.get("content", str(msg))}
                for msg in messages
            ]

        # Estimate tokens (4 chars ~ 1 token)
        token_estimate = sum(len(str(m["content"])) for m in messages) // 4
        if self.tokens_used + token_estimate > self.tpm_limit:
            sleep_time = 60 - (current_time - self.last_reset)
            print(f"Rate limit approaching. Sleeping for {sleep_time:.2f} seconds.")
            time.sleep(max(sleep_time, 0))

        # Make API call
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=1,
                max_completion_tokens=512,
                top_p=1,
                stream=False,
            )
            self.tokens_used += token_estimate + 512
            return completion.choices[0].message.content
        except Exception as e:
            print(f"API call failed: {e}")
            raise

# Instantiate Groq LLM with your API key
groq_llm = GroqLLM(api_key="YOUR_GROQ_API_KEY_HERE")

# ScrapeGraphAI configuration
graph_config = {
    "llm": {"model_instance": groq_llm, "model_tokens": 6000},
    "verbose": True,
    "headless": True,
    "chunk_size": 1000,
    "max_chunks": 10,
}

# Define the prompt
prompt = "Find the assassination attempt on Caliph Umar bin al-Khattab, including details like date, location, and perpetrator."

# Create and run the SmartScraperGraph
smart_scraper_graph = SmartScraperGraph(
    prompt=prompt,
    source="https://en.wikipedia.org/wiki/Umar",
    config=graph_config
)

result = smart_scraper_graph.run()

# Print the result
print("Final result:")
print(json.dumps(result, indent=4))

# Debug: Print HTML snippet
state = smart_scraper_graph.final_state
if "document" in state:
    print("Fetched HTML snippet:")
    print(state["document"][:1000])
```

### Replace the API Key
Replace `YOUR_GROQ_API_KEY_HERE` with the actual API key (e.g., `gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`).

## Step 5: Run the Script
Activate the virtual environment if not already active:
```bash
D:\Study\Courses\pingPongScript\venv\Scripts\activate
```

Execute the script:
```bash
python testing.py
```

## Step 6: Interpret the Output
### Successful Output Example
```json
{
    "assassination_attempt": {
        "date": "November 3, 644 CE",
        "location": "Mosque in Medina",
        "perpetrator": "Abu Lu'lu'a Firuz",
        "details": "Umar was stabbed with a double-edged dagger while leading morning prayers."
    }
}
```

### Debugging:
- Check the HTML snippet printed at the end to ensure it contains the "Assassination" section from Wikipedia.
- Adjust chunk size or max chunks if the section isn't being fetched.

## Troubleshooting Common Issues
- **Rate Limit Error (429)**: Wait for 2 minutes and retry.
- **Empty or Incorrect Output**: Ensure the prompt is specific enough and the fetched content includes relevant information.
- **Playwright Issues**: If fetching fails, rerun `playwright install`.

## Additional Notes
- **Model Choice**: The script uses `llama-3.3-70b-versatile`. Check Groq’s Model List for alternatives.
- **Token Management**: The script estimates tokens as 4 chars = 1 token. Adjust based on feedback.
- **Ethical Use**: Scrape only public data and respect Wikipedia’s terms of service.

## Conclusion
You've successfully set up ScrapeGraphAI with Groq to scrape structured data using a LLaMA-compatible model. Experiment with different prompts and sources to refine your results!


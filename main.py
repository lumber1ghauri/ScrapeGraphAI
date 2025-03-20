from groq import Groq
from scrapegraphai.graphs import SmartScraperGraph
import json
import time

# Initialize the Groq client with your API key
client = Groq(api_key="you api key here")

# Define a custom LLM wrapper with rate limiting
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
                {
                    "role": msg[0] if isinstance(msg, tuple) and len(msg) > 1 else (msg.get("role", "user") if isinstance(msg, dict) else "user"),
                    "content": msg[1] if isinstance(msg, tuple) and len(msg) > 1 else (msg.get("content", str(msg)) if isinstance(msg, dict) else str(msg))
                }
                for msg in messages
            ]

        # Estimate tokens after normalization (4 chars ~ 1 token)
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
                stop=None,
            )
            self.tokens_used += token_estimate + 512  # Rough estimate including output
            return completion.choices[0].message.content
        except Exception as e:
            print(f"API call failed: {e}")
            raise

# Instantiate the custom Groq LLM
groq_llm = GroqLLM(api_key="your api key here")

# Configuration for ScrapeGraphAI
graph_config = {
    "llm": {
        "model_instance": groq_llm,
        "model_tokens": 6000,
    },
    "verbose": True,
    "headless": True,
    "chunk_size": 1000,  # Reduced to stay within TPM
    "max_chunks": 10,
}

# Define the scraping prompt
prompt = "give some prompt here to scrape"

# Create and run the SmartScraperGraph instance
smart_scraper_graph = SmartScraperGraph(
    prompt=prompt,
    source="paste the link of the website to scrape",
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
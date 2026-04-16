"""Gemini 2.5 Flash model integration via Google GenAI API.

HOW THIS FILE WORKS:
1. Reads your API key from the .env file
2. Creates a client connection to Google's GenAI service
3. Exposes get_answer(question, context) — the same interface as every
   other model file, so the evaluation runner can treat all models the same

KEY CONCEPTS:
- "Client pattern": We create a genai.Client object that handles all
  communication with Google's servers. This is a common pattern in API
  libraries — one client object, many requests through it.
- "Lazy initialization": We don't create the client until the first time
  get_answer() is called. Avoids setup if it's never used.
- "Rate limiting": Google's free tier allows 15 requests per minute for
  flash models. We add a delay between calls to stay under this limit.
- "Retry with exponential backoff": If an API call fails, we wait and
  try again, waiting longer each time (2s, 4s, 8s).
"""

import os
import time

from dotenv import load_dotenv
from google import genai

from models.base import build_prompt

# load_dotenv() reads the .env file and makes those values available
# via os.getenv(). This keeps secrets out of your code.
load_dotenv()

# Read the API key from environment variables.
_api_key = os.getenv("GEMINI_API_KEY")

# The client object handles all communication with Google's API.
# None initially — created on first use (lazy initialization).
_client = None

# Which Gemini model to use. "flash" models are optimized for speed
# and have generous free tier quotas.
_MODEL_NAME = "gemini-2.5-flash"

# Delay between API calls in seconds.
# Free tier = 15 requests/minute, so ~4 seconds between calls.
_MIN_DELAY_SECONDS = 4.0


def _get_client():
    """Create the GenAI client (only once, on first use).

    Returns the client object used to make API calls. Creates it on
    first call, returns the cached one on subsequent calls.
    """
    global _client
    if _client is None:
        if not _api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not found in environment. "
                "Copy .env.example to .env and add your key. "
                "Get a free key at: https://aistudio.google.com/app/apikey"
            )
        _client = genai.Client(api_key=_api_key)
    return _client


def get_answer(question: str, context: str) -> str:
    """Send a question to Gemini and return the answer.

    This is the function that the evaluation runner calls. It has the
    same signature (same inputs and output type) as get_answer() in
    every other model file. This consistency is what lets the runner
    treat all models identically.

    Args:
        question: The question text (in Hindi).
        context: The passage providing context for the question.

    Returns:
        The model's answer as a string.

    Raises:
        RuntimeError: If GEMINI_API_KEY is not set.
        Exception: If the API call fails after all retries.
    """
    # Step 1: Build the prompt using our shared template
    prompt = build_prompt(question, context)

    # Step 2: Get the client (creates it on first call)
    client = _get_client()

    # Step 3: Call the API with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # client.models.generate_content() sends the prompt to
            # Google's servers and returns the model's response.
            # 'model' specifies which Gemini model to use.
            # 'contents' is the prompt text to send.
            response = client.models.generate_content(
                model=_MODEL_NAME,
                contents=prompt,
            )

            # Sleep to respect rate limits
            time.sleep(_MIN_DELAY_SECONDS)

            # Extract the text from the response and clean it up
            return response.text.strip()

        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff: 2s, then 4s, then give up
                wait = 2 ** (attempt + 1)
                print(f"Gemini API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

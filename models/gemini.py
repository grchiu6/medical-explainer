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
# Free tier for gemini-2.5-flash = 5 requests/minute = 1 call per 12s.
# 13s gives a small safety margin to absorb network jitter.
_MIN_DELAY_SECONDS = 13.0


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


def get_answer(prompt: str) -> str:
    """Send a pre-built prompt to Gemini and return the answer.

    The caller (evaluate.py) is responsible for building the prompt
    using the right template for the task (extractive vs MCQ etc.).
    This file just handles the API plumbing — it doesn't know or care
    what kind of task the prompt represents.

    Args:
        prompt: The fully-formatted prompt string to send.

    Returns:
        The model's answer as a string.

    Raises:
        RuntimeError: If GEMINI_API_KEY is not set.
        Exception: If the API call fails after all retries.
    """
    client = _get_client()

    # Call the API with retry logic
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
                # Backoff: 30s, then 60s. Long enough to clear a typical
                # rate-limit window (free tier is 5 RPM = 60s window).
                wait = 30 * (attempt + 1)
                print(f"Gemini API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

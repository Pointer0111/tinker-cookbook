"""
Shared utilities for DashScope embedding generation with retry logic
(OpenAI-compatible API)
"""

import asyncio
from logging import getLogger
from os import environ

from dotenv import load_dotenv
from openai import AsyncOpenAI

_dotenv_loaded = load_dotenv()

logger = getLogger(__name__)

if _dotenv_loaded:
    print("[embedding] .env file loaded successfully")
else:
    print("[embedding] no .env file found, using environment variables")

MAX_RETRIES = 10
RETRY_DELAY = 1.0


def get_dashscope_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> AsyncOpenAI:
    api_key = api_key or environ.get("EMBEDDING_BINDING_API_KEY")
    if api_key is None:
        raise ValueError("$EMBEDDING_BINDING_API_KEY is not set")

    base_url = base_url or environ.get(
        "EMBEDDING_BINDING_HOST",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    return AsyncOpenAI(api_key=api_key, base_url=base_url)


async def get_dashscope_embedding(
    client: AsyncOpenAI,
    texts: list[str],
    model: str = "text-embedding-v4",
    embedding_dim: int = 1024,
    max_retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_DELAY,
) -> list[list[float]]:
    """
    Get embeddings from DashScope (OpenAI-compatible API) with exponential backoff retry.

    Args:
        client: AsyncOpenAI client pointed at DashScope.
        texts: List of texts to embed.
        model: Embedding model name (default: "text-embedding-v4").
        embedding_dim: Output embedding dimension (default: 1024).
        max_retries: Maximum number of retries (default: 10).
        retry_delay: Base delay between retries in seconds (default: 1.0).

    Returns:
        List of embeddings, same length as input texts.

    Raises:
        Exception: If embedding generation fails after all retries.
    """
    if not texts:
        raise ValueError("No texts provided for embedding generation")

    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError(f"Text at index {i} is not a string: {type(text)} = {text}")
        if not text.strip():
            raise ValueError(f"Text at index {i} is empty or whitespace only")

    for attempt in range(max_retries):
        try:
            async with asyncio.timeout(30):
                response = await client.embeddings.create(
                    model=model,
                    input=texts,
                    dimensions=embedding_dim,
                )

            if not response.data:
                raise ValueError("No embeddings returned from DashScope API")

            if len(response.data) != len(texts):
                raise ValueError(
                    f"Mismatch: expected {len(texts)} embeddings, got {len(response.data)}"
                )

            return [item.embedding for item in response.data]

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (1.5**attempt)
                logger.error(
                    f"Attempt {attempt + 1}/{max_retries} failed for embedding "
                    f"({len(texts)} texts): {e!r}. Retrying in {wait_time:.1f}s..."
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    f"All {max_retries} attempts failed for embedding ({len(texts)} texts): {e!r}"
                )
                raise

    raise RuntimeError("Unexpected error in retry logic")

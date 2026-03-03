"""
Shared LLM utilities for all agents.
Provides lazy initialization of ChatOpenAI instance.
"""
import logging
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Lazy initialization of LLM
_llm = None


def get_llm():
    """
    Get or create shared LLM instance.
    Uses lazy initialization to avoid creating LLM before API key is available.
    
    Returns:
        ChatOpenAI instance (gpt-4o-mini, temperature=0)
    """
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        logger.info("LLM instance created (lazy initialization)")
    return _llm


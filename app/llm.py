from langchain_openai import AzureChatOpenAI
from .config import settings

def get_llm() -> AzureChatOpenAI:
    # Temperature low for reliability; adjust later per use case.
    return AzureChatOpenAI(
        azure_deployment=settings.azure_openai_deployment,
        api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        temperature=0.2,
        timeout=60,
        max_retries=2,
    )
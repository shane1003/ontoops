from langfuse import Langfuse
from pydantic import SecretStr

from config import settings

_provider = settings.llm_provider

# SDK-level retry count. The value 2 equals the openai/anthropic SDK default, so
# behavior is unchanged; it is explicit because the worst-case latency must not hang on
# a library default: if not passed, the SDK default applies, and a single uv lock could
# silently change the worst-case request time with zero lines of our code changed (the
# timeout multiplier arithmetic in config.py would become false from that day).
# Why not lower it to 0: cq_generator has no with_retry, so this layer is the only
# retry.
_MAX_RETRIES = 2


def _make_llm(max_tokens: int, temperature: float, timeout: int):
    if _provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=SecretStr(settings.openai_api_key),
            model=settings.openai_model,
            temperature=temperature,
            max_completion_tokens=max_tokens or None,
            timeout=timeout,
            max_retries=_MAX_RETRIES,
        )
    elif _provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            api_key=SecretStr(settings.anthropic_api_key),
            model_name=settings.anthropic_model,
            # temperature=temperature,
            max_tokens_to_sample=max_tokens or None,
            timeout=timeout,
            max_retries=_MAX_RETRIES,
            stop=None,
        )
    else:  # local
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            base_url=settings.vllm_base_url,
            api_key=SecretStr("dummy"),
            model=settings.vllm_model,
            temperature=temperature,
            max_completion_tokens=max_tokens or None,
            timeout=timeout,
            max_retries=_MAX_RETRIES,
        )


generator_llm = _make_llm(
    max_tokens=settings.cq_generator_max_tokens,
    temperature=settings.cq_generator_temperature,
    timeout=settings.cq_generator_request_timeout,
)
verifier_llm = _make_llm(
    max_tokens=settings.cq_verify_max_tokens,
    temperature=settings.cq_verify_temperature,
    timeout=settings.cq_verify_request_timeout,
)
refiner_llm = _make_llm(
    max_tokens=settings.cq_refine_max_tokens,
    temperature=settings.cq_refine_temperature,
    timeout=settings.cq_refine_request_timeout,
)
terminologizer_llm = _make_llm(
    max_tokens=settings.cq_terminologizer_max_tokens,
    temperature=settings.cq_terminologizer_temperature,
    timeout=settings.cq_terminologizer_request_timeout,
)
structure_llm = _make_llm(
    max_tokens=settings.odl_structure_max_tokens,
    temperature=settings.odl_structure_temperature,
    timeout=settings.odl_structure_request_timeout,
)

# Initialize the langfuse global client from settings. client.py is imported by both CLI
# and web, so CallbackHandler() reuses this setup (env fallback if not initialized).
langfuse = Langfuse(
    secret_key=settings.langfuse_secret_key,
    public_key=settings.langfuse_public_key,
    host=settings.langfuse_host,
)

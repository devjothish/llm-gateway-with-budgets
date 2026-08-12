"""Tracing setup. Imported and called before anything that talks to a provider.

Redaction is configured here, before the first span is emitted, rather than
after someone notices a key in a trace. `grounded-rag` already flags
`instrument_httpx(capture_all=True)` as noisier than expected; on this service
that setting would capture provider `Authorization` headers and the client's
own API key on every request, so it is off and the interesting fields are
attached to spans deliberately instead.
"""

import logfire
from fastapi import FastAPI

from app.settings import settings

# Anything matching these never reaches a span attribute. Belt and braces: no
# code path below should be putting these in a span in the first place.
_SCRUB_PATTERNS = [
    "api[-_]?key",
    "authorization",
    "x-api-key",
    "secret",
    "password",
    "bearer",
]


def configure_observability() -> None:
    logfire.configure(
        token=settings.logfire_token,
        service_name=settings.service_name,
        environment=settings.environment,
        # No token still prints spans to the console, so local dev is traced
        # rather than silently untraced.
        send_to_logfire="if-token-present",
        scrubbing=logfire.ScrubbingOptions(extra_patterns=_SCRUB_PATTERNS),
    )
    # capture_all stays off: it records request headers, which on this service
    # means provider credentials.
    logfire.instrument_httpx()
    logfire.instrument_asyncpg()
    logfire.instrument_redis()


def instrument_app(app: FastAPI) -> None:
    logfire.instrument_fastapi(app)

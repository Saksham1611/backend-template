import logging

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> None:
    """
    Replaces standard logging handlers with Rich.
    Call this ONCE at app startup, before anything else.
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers (like the default Uvicorn one)
    # so we don't get double logs
    logger.handlers = []

    # Create the Rich Handler
    rich_handler = RichHandler(
        rich_tracebacks=True,  # Pretty colorful tracebacks
        markup=True,  # Allow [bold red]text[/] syntax
        show_time=True,
        show_path=False,
    )

    # Format
    formatter = logging.Formatter("%(message)s")
    rich_handler.setFormatter(formatter)

    logger.addHandler(rich_handler)

    # Silence noisy libraries if needed
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

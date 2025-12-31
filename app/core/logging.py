import logging

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Simple Rich logging configuration with filename and line numbers.
    """
    logging.basicConfig(
        level=level,
        format="%(filename)s:%(lineno)d - %(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                markup=True,
                show_time=False,  # Remove timestamp
                show_path=False,  # Manual path in format
            )
        ],
        force=True,  # Override any existing config
    )
    return logging.getLogger("app")


# Initialize and export logger
logger = setup_logging()

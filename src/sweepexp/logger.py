"""Custom logger for SweepExp."""
from __future__ import annotations

import logging

# Create a custom logger
log: logging.Logger = logging.getLogger("sweepexp")

# Create handlers
c_handler: logging.StreamHandler = logging.StreamHandler()

# Create formatters and add it to handlers
c_format: logging.Formatter = logging.Formatter(
    "%(levelname)s - %(message)s",
)
c_handler.setFormatter(c_format)

# Add handlers to the logger
log.addHandler(c_handler)

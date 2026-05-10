import logging

TRACE = 5
logging.addLevelName(TRACE, "TRACE")


def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kwargs)


logging.Logger.trace = trace


def setup_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        numeric = logging.INFO
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

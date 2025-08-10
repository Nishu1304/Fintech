import logging

logger = logging.getLogger("fintech")
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(console_handler)
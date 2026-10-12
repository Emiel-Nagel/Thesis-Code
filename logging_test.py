from thesis_code.logging import Logger

logger = Logger(run_name="Test run", seed=0)
logger.log("Does this log?")
logger.log("Maybe this logs too?")

import traceback

try:
    raise TypeError("Let's test the errors as well")
except Exception as e:
    logger.log(f"An error occurred: {e}, {traceback.format_exception(e)}")
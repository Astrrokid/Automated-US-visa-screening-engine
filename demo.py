from us_visa.logger import logging
from us_visa.exception import USvisaException
import sys
# logging.info("This is a log message from demo.py")

try:
    a = 2/0
except Exception as e:
    raise USvisaException(e, sys)
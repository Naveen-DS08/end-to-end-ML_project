import logging 
import os 
from datetime import datetime

LOG_NAME_STRUCTURE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"
LOG_FILE_PATH = os.path.join(os.getcwd(), "logs", LOG_NAME_STRUCTURE)
os.makedirs(LOG_FILE_PATH, exist_ok=True)
LOG_FILE_NAME = LOG_NAME_STRUCTURE+".log" 
LOG_FILE = os.path.join(LOG_FILE_PATH, LOG_FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE,
    format= "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level= logging.INFO,
)

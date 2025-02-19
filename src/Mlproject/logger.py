


import logging
import os
from datetime import datetime

# Generate log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define log directory path
log_path = os.path.join(os.getcwd(), "logs")
os.makedirs(log_path, exist_ok=True)  # Create logs directory if not exists

# Define full log file path
LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)  # Fixed variable name

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Example usage
# logging.info("Logging is set up successfully!")


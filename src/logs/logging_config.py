import logging

# 1. Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Capture everything

# 2. Define how the logs should look
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 3. Create a handler for the Terminal
terminal_handler = logging.StreamHandler()
terminal_handler.setLevel(logging.INFO) # Only show INFO and above in the terminal
terminal_handler.setFormatter(formatter)

# 4. Create a handler for a File
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG) # Save EVERYTHING to the file
file_handler.setFormatter(formatter)

# 5. Add both handlers to your logger
logger.addHandler(terminal_handler)
logger.addHandler(file_handler)


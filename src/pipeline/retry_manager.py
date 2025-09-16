import time
import random

class RetryManager:
    """A manager to handle retries with exponential backoff."""

    def __init__(self, max_retries=3, initial_delay=1, max_delay=10):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay

    def execute(self, func, *args, **kwargs):
        """
        Execute a function with retries on failure.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                if retries >= self.max_retries:
                    raise e
                delay = min(self.max_delay, self.initial_delay * (2 ** retries))
                delay += random.uniform(0, 0.1 * delay)  # Add jitter
                print(f"Error: {e}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
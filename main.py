import sys
import os

# Add the 'src' directory to the system path to enable module imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Import the main pipeline function for multi-scale SSCA processing
from pipeline import multiscale_ssca

# Main entry point of the script
if __name__ == "__main__":
    # Execute the multi-scale SSCA processing pipeline
    multiscale_ssca()

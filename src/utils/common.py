import os
import sys

# Get project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Add to Python path
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
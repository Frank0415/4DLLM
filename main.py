#!/usr/bin/env python3
"""
Main entry point for the 4D-STEM Research Platform.
Starts the MCP server for database access and analysis tools.
"""

import sys
import os

# Add the project root to the path so we can import postgres_mcp modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from postgres_mcp import main as mcp_main

if __name__ == "__main__":
    # Call the postgres_mcp package's entry point
    sys.exit(mcp_main())
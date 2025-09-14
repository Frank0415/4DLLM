#!/bin/bash

# Helper script to set up API configuration for LLM analysis

echo "Setting up API configuration for 4DLLM LLM analysis..."
echo ""

CONFIG_FILE="config/api_keys.json"
EXAMPLE_FILE="config/api_keys_example_real.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ $CONFIG_FILE not found!"
    echo ""
    echo "Please create $CONFIG_FILE with your real API keys."
    echo "You can use $EXAMPLE_FILE as a template:"
    echo ""
    cat "$EXAMPLE_FILE"
    echo ""
    echo "To get a Gemini API key:"
    echo "1. Go to https://ai.google.dev/"
    echo "2. Click 'Get API key in Google AI Studio'"
    echo "3. Create a new API key"
    echo "4. Replace the placeholder keys in $CONFIG_FILE"
    exit 1
fi

echo "✅ $CONFIG_FILE found"

# Check if it contains placeholder values
if grep -q "your-api-key" "$CONFIG_FILE"; then
    echo "❌ $CONFIG_FILE contains placeholder values!"
    echo ""
    echo "Please replace the placeholder API keys with real ones:"
    echo "- Get API keys from https://ai.google.dev/"
    echo "- Update the 'api_keys' array in $CONFIG_FILE"
    exit 1
fi

echo "✅ API configuration appears to be set up correctly"
echo ""

# Test if the log file is writable
LOG_FILE="/tmp/llm_logging"
if touch "$LOG_FILE" 2>/dev/null; then
    echo "✅ Log file $LOG_FILE is writable"
else
    echo "❌ Cannot write to $LOG_FILE"
    echo "Creating log file with proper permissions..."
    sudo touch "$LOG_FILE"
    sudo chmod 666 "$LOG_FILE"
    echo "✅ Log file created"
fi

echo ""
echo "🎉 Setup complete! You can now use the LLM analysis tools."
echo ""
echo "Test the setup with:"
echo "  test_llm_analysis(image_path='/path/to/your/image.png')"
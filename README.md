# Agentic AI Learning

This project demonstrates agentic AI using CrewAI to find conference venues. It includes examples of multi-agent collaboration with tools for web search.

## Prerequisites

- Python 3.11 or higher
- Virtual environment (recommended)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rajeshkumar/agentic-ai.git
   cd agentic-ai
   ```
   Or navigate to the project directory if already cloned.

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install crewai crewai-tools langchain-ollama
   ```

## Running with Ollama (Local LLM)

For free, local execution without API keys:

### 1. Install Ollama

- Download from [ollama.ai](https://ollama.ai/)
- Install the macOS app or CLI

### 2. Pull and run a model

```bash
# Pull a model (e.g., Llama 2)
ollama pull llama2

# Start the Ollama server (in another terminal)
ollama serve
```

### 3. Configure the script for Ollama

The script is pre-configured to use Ollama. Ensure Ollama is running with the model loaded.

### 4. Run the script

```bash
python crewAI/venue-finder.py
```

Or activate the virtual environment first:

```bash
source .venv/bin/activate
python crewAI/venue-finder.py
```

## Running with OpenAI (Cloud)

If you prefer OpenAI:

1. Get an OpenAI API key from [platform.openai.com](https://platform.openai.com/)
2. Add to `.env`:
   ```
   SERPER_API_KEY=...  # For web search
   ```
3. Run the script as-is (it loads from `.env`)

## Notes

- Ollama provides free, local inference
- OpenAI requires billing after trial credits
- The script uses SerperDevTool for web search (requires API key)

## Troubleshooting

- Ensure Ollama server is running: `ollama serve`
- Check model availability: `ollama list`
- For OpenAI quota issues, add billing or use Ollama

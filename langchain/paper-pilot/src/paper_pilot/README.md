# Paper Pilot 🚀

A Streamlit application for summarizing research papers and extracting key insights using LangChain.

## Setup & Running the Application

Follow these steps to set up the development environment, format code, run static checks, and launch the application.

### 1. Install Dependencies
Install all required project dependencies (including development tools) using `uv`:

```bash
uv sync --group dev
```



### 2. Format Codebase

```bash
uv run ruff format .
```

### 3. Lint & Auto-Fix Code Issues

```bash
uv run ruff check --fix .
```

### 4. Run Static Type Checking

```bash
uv run mypy .
```

### 5. Launch the Streamlit App

```bash
uv run streamlit run src/paper_pilot/paper_summary_app.py
```

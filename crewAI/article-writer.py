import warnings
import os
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
os.environ['PIP_ROOT_USER_ACTION'] = 'ignore'

# Load API keys from an external .env file at the project root.
env_file = Path(__file__).resolve().parent.parent / '.env'
if not env_file.exists():
    raise RuntimeError(f"Environment file not found: {env_file}")

with env_file.open('r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)

from crewai import LLM

# Configure LLM - use Ollama local server
llm = LLM(
    model="ollama/llama2:latest",
    base_url="http://localhost:11434",
    api_key="testapikey"
)

# Create article directory if it doesn't exist
article_dir = Path(__file__).resolve().parent / "article"
article_dir.mkdir(exist_ok=True)

# Simple research and writing without CrewAI agents
def research_topic(topic):
    """Generate research on a given topic using Ollama"""
    prompt = f"""You are a senior researcher. Conduct a thorough analysis on the topic: {topic}
    
Provide:
1. Key facts and findings
2. Important details and context
3. Recent developments
4. Expert opinions

Format the response as a detailed research report."""
    
    response = llm.call(
        messages=[{"role": "user", "content": prompt}]
    )
    return response

def write_article(topic, research_data):
    """Write an article based on research data"""
    prompt = f"""You are a senior writer. Based on the following research, write an engaging 6-paragraph article about: {topic}
    
Research data:
{research_data}

The article should be clear, engaging, and easy to understand. Make it compelling and informative."""
    
    response = llm.call(
        messages=[{"role": "user", "content": prompt}]
    )
    return response

def save_article_as_html(topic, article_content, research_content):
    """Save article as an HTML file"""
    # Generate filename from topic
    filename = f"{topic.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    filepath = article_dir / filename
    
    # Create HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{topic}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .metadata {{
            color: #7f8c8d;
            font-size: 14px;
            margin-bottom: 20px;
        }}
        .article {{
            margin-top: 20px;
            text-align: justify;
        }}
        p {{
            margin-bottom: 15px;
        }}
        .research-section {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
        }}
        .research-section h2 {{
            color: #2c3e50;
            font-size: 18px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{topic}</h1>
        <div class="metadata">
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="article">
            {article_content.replace(chr(10), '</p><p>')}
        </div>
        
        <div class="research-section">
            <h2>Research Data</h2>
            <p>{research_content.replace(chr(10), '</p><p>')}</p>
        </div>
    </div>
</body>
</html>"""
    
    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return filepath

# Main execution
research_inputs = {
    'topic': ''
}

if not research_inputs['topic']:
    research_inputs['topic'] = input("Please enter a topic for research and article writing: ").strip()

if research_inputs['topic']:
    print("\n🔍 Researching topic...")
    research = research_topic(research_inputs['topic'])
    print(f"\n📊 Research Results:\n{research}\n")
    
    print("\n✍️  Writing article...")
    article = write_article(research_inputs['topic'], research)
    print(f"\n📝 Article:\n{article}")
    
    print("\n💾 Saving article as HTML...")
    filepath = save_article_as_html(research_inputs['topic'], article, research)
    print(f"\n✅ Article saved successfully!")
    print(f"📁 File location: {filepath}")
else:
    print("No topic provided. Exiting.")
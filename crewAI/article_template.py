"""
Article HTML Template Module
Contains reusable HTML templates for article generation
"""

# HTML Template for articles
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
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
        <h1>{title}</h1>
        <div class="metadata">
            <p>Generated on: {timestamp}</p>
        </div>
        
        <div class="article">
            {article_content}
        </div>
        
        <div class="research-section">
            <h2>Research Data</h2>
            {research_content}
        </div>
    </div>
</body>
</html>"""


def format_text_for_html(text):
    """Convert plain text to HTML paragraphs
    
    Args:
        text (str): Plain text content
        
    Returns:
        str: HTML formatted content with <p> tags
    """
    paragraphs = text.strip().split('\n')
    return ''.join(f'<p>{p.strip()}</p>' for p in paragraphs if p.strip())


def render_article(title, timestamp, article_content, research_content):
    """Render article using the HTML template
    
    Args:
        title (str): Article title
        timestamp (str): Generation timestamp
        article_content (str): Plain text article content
        research_content (str): Plain text research content
        
    Returns:
        str: Rendered HTML string
    """
    article_html = format_text_for_html(article_content)
    research_html = format_text_for_html(research_content)
    
    return HTML_TEMPLATE.format(
        title=title,
        timestamp=timestamp,
        article_content=article_html,
        research_content=research_html
    )

import datetime
import os
import warnings
from pathlib import Path

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

from crewai import Agent, Task, Crew, LLM

# Configure LLM - use Ollama local server
# Ollama speaks an OpenAI-compatible API, so we use the ollama provider string.
llm = LLM(
    model="ollama/llama2:latest",
    base_url="http://localhost:11434",
    api_key="testapikey"
)

venue_finder_agent = Agent(
    name="Venue Finder Agent",
    role="Conference Venue Finder",
    goal="Find the best venue for the upcoming conference",
    backstory=(
        "You are an experienced event planner with a knack for finding the perfect venues. "
        "Your expertise ensures that all conference requirements are met efficiently. "
        "Your goal is to provide the client with the best possible venue options."
    ),
    llm=llm,
    verbose=True
)


venue_quality_assurance_agent = Agent(
    name="Venue Quality Assurance Agent",
    role="Venue Quality Assurance Specialist",
    goal="Ensure the selected venue meets all the necessary quality standards for the conference",
    backstory=(
        "You are a meticulous quality assurance specialist with extensive experience in evaluating conference venues. "
        "Your expertise allows you to assess venues based on various criteria such as location, amenities, capacity, and overall suitability for the conference. "
        "Your goal is to ensure that the selected venue meets all the necessary quality standards and provides an exceptional experience for conference attendees. "
        "Your job is to thoroughly evaluate potential venues and provide detailed feedback on their suitability for the conference, ensuring that the final selection is of the highest quality."
    ),
    llm=llm,
    verbose=True
)

# Create a Task
venue_finding_task = Task(
    name="Find Conference Venue",
    description=(
        "Conduct a thorough search to find the best venue for the upcoming conference, ensuring it meets all necessary requirements and quality standards. "
        "Consider factors such as location, amenities, capacity, pricing, and overall suitability for the conference. "
        "Provide a detailed report on the top venue options and their respective evaluations. "
        "Use online resources, reviews, and any other relevant information to make informed decisions about the best venues for the conference."
    ),
    expected_output=(
        "A comprehensive list of top 5 potential venues, each with a detailed report outlining their locations, amenities, capacities, and pricing. "
        "The report should also include a thorough evaluation of each venue based on the specified criteria, along with recommendations for the best venue choice."
    ),
    agent=venue_finder_agent
)

#review task and add the quality assurance agent to the task

quality_assurance_task = Task(
    name="Venue Quality Assurance Task",
    description=(
        "Review the venue options provided by the Venue Finder Agent and ensure they meet all necessary quality standards for the conference. "
        "Evaluate each venue based on factors such as location, amenities, capacity, pricing, and overall suitability for the conference. "
        "Provide a detailed report on the quality of each venue and make recommendations for the best venue choice based on the evaluations."
    ),
    expected_output=(
        "A detailed review of top 5 potential venues provided by the Venue Finder Agent, including an assessment of their quality based on the specified criteria. "
        "The report should include a comprehensive evaluation of each venue, highlighting their strengths and weaknesses, and provide a clear recommendation for the best venue choice based on the overall quality assessment."
    ),
    agent=venue_quality_assurance_agent
)


# Create a Crew and add the Task to the Crew
venue_finding_planning_crew = Crew(
    name="Venue Finding Planning Crew",
    agents=[venue_finder_agent, venue_quality_assurance_agent],
    tasks=[venue_finding_task, quality_assurance_task],
    verbose=True,
)

# Run the Crew
input_data = {
    "conference_name": "Tech Innovators Conference 2026",
    "conference_date": (datetime.datetime.now().date() + datetime.timedelta(days=7)).isoformat(), # Set the conference date to 7 days from now
    "required_amenities": ["Wi-Fi", "Projector", "Catering Services", "Parking"],
    "expected_attendees": 200,
    "location_preference": "Mumbai, India",
    "budget": 5000
}

# Extract location for dynamic task descriptions
location = input_data.get("location_preference", "")

# Update task descriptions with the location
if location:
    venue_finding_task.description = (
        f"Conduct a thorough search to find the best venue for the upcoming conference in {location}, ensuring it meets all necessary requirements and quality standards. "
        f"The venue MUST be located in {location}. Consider factors such as location, amenities, capacity, pricing, and overall suitability for the conference. "
        "Provide a detailed report on the top venue options and their respective evaluations. "
        f"Use online resources, reviews, and any other relevant information to make informed decisions about the best venues for the conference in {location}."
    )
    venue_finding_task.expected_output = (
        f"A comprehensive list of top 5 potential venues in {location}, each with a detailed report outlining their locations, amenities, capacities, and pricing. "
        "The report should also include a thorough evaluation of each venue based on the specified criteria, along with recommendations for the best venue choice. "
        f"ALL venues must be located in {location}."
    )
    
    quality_assurance_task.description = (
        f"Review the venue options in {location} provided by the Venue Finder Agent and ensure they meet all necessary quality standards for the conference. "
        f"ALL venues must be located in {location}. Evaluate each venue based on factors such as location, amenities, capacity, pricing, and overall suitability for the conference. "
        "Provide a detailed report on the quality of each venue and make recommendations for the best venue choice based on the evaluations."
    )
    quality_assurance_task.expected_output = (
        f"A detailed review of top 5 potential venues in {location} provided by the Venue Finder Agent, including an assessment of their quality based on the specified criteria. "
        "The report should include a comprehensive evaluation of each venue, highlighting their strengths and weaknesses, and provide a clear recommendation for the best venue choice. "
        f"Confirm that ALL venues are located in {location}."
    )

venue_finding_planning_crew.kickoff(input_data)

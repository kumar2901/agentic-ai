from pydantic import BaseModel, Field


class PaperSummary(BaseModel):
    title: str = Field(..., description="The title of the research paper to summarize.")
    key_ideas: str = Field(..., description="The key ideas or concepts from the research paper.")
    equations: list[str] = Field(
        ...,
        description="Any relevant mathematical equations from the research paper in latex format.",
    )
    analogy: str = Field(
        ...,
        description="A simple analogy to explain the research paper's concepts in a relatable way.",
    )
    difficulty_level: int = Field(
        le=5,
        ge=1,
        description="The difficulty level 1 (easy) to 5 (very difficult) of the research paper and 3 (medium) by default.",
    )


class Keywords(BaseModel):
    keywords: list[str] = Field(description="5 short keys describing the paper.")


class Source(BaseModel):
    """Where a fact came from"""

    paper: str = Field(description="Paper title exactly as written in the context labe")
    page: int = Field(ge=1, description="Page number from the context label")


class RagAnswer(BaseModel):
    """An Answer grounded only in the retrieved paper chhunks."""

    answer: str = Field(description='The answer, or "I don\'t know" if the context lacks it')
    sources: list[Source] = Field(
        description="Paper and page for every fact used; empty if unknown"
    )

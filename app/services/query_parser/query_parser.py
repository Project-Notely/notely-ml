from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.core.config import settings


class ExtractionQuery(BaseModel):
    """The user's query specifying what to extract from the document."""

    query: str = Field(
        description="A clear, concise query for an object detection model. For example: 'the main title', 'all the paragraphs and the chart on the left'."
    )


class QueryParser:
    """Parses a user's natural language query into a structured format."""

    def __init__(self):
        """Initializes the QueryParser with a structured LLM."""
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0,
        )
        self.structured_llm = llm.with_structured_output(ExtractionQuery)

    async def execute(self, user_query: str) -> ExtractionQuery:
        """Parses the user's query using a structured LLM call.

        Args:
            user_query: The natural language query from the user.

        Returns:
            A structured query object.
        """
        prompt = f"Extract the key entities the user wants to find in the document from the following query: '{user_query}'"
        return await self.structured_llm.ainvoke(prompt)

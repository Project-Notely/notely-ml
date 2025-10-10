from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.core.config import settings


class ExtractionQuery(BaseModel):
    query: str = Field(
        description=(
            "A clear, concise query for an object detection model. "
            "For example: 'the main title', 'all the paragraphs and "
            "the chart on the left'."
        )
    )


class QueryParser:
    def __init__(self):
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0,
        )
        self.structured_llm = llm.with_structured_output(ExtractionQuery)

    async def execute(self, user_query: str) -> ExtractionQuery:
        prompt = (
            "Extract the key entities the user wants to find in the document "
            f"from the following query: '{user_query}'"
        )
        return await self.structured_llm.ainvoke(prompt)

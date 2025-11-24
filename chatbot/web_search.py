import os

from dotenv import load_dotenv
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import tool, Tool
from pydantic import BaseModel
from serpapi import GoogleSearch


# we use the article object for parsing serpapi results later
class Article(BaseModel):
    title: str
    source: str
    link: str
    snippet: str

    @classmethod
    def from_serpapi_result(cls, result: dict) -> "Article":
        return cls(
            title=result["title"],
            source=result["source"],
            link=result["link"],
            snippet=result["snippet"],
        )


@tool
def serpapi(query: str) -> list[Article]:
    """Use this tool to search the web."""
    load_dotenv()
    params = {
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "engine": "google",
        "q": query,
        "google_domain": "google.com",
        "gl": "it",
        "hl": "en"
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    return [Article.from_serpapi_result(organic_result) for organic_result in results["organic_results"]]
    # return [Article.from_serpapi_result({"title": "Weather in Lecce", "source": "google.it", "link": "www.google.it", "snippet": "Weather is good"})]

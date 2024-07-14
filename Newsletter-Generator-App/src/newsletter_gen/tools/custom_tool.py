from crewai_tools import BaseTool
from exa_py import Exa
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time


load_dotenv()

class SearchAndContents(BaseTool):
    name: str = "Search and Contents Tool"
    description: str = (
        "Searches the web based on a search query for the latest results. Results are only from the last week. Uses the Exa API. This also returns the contents of the search results."
    )

    def _run(self, search_query: str) -> str:
        exa = Exa(api_key=os.getenv("EXA_API_KEY"))
        one_week_ago = datetime.now() - timedelta(days=1)
        date_cutoff = one_week_ago.strftime("%Y-%m-%d")
        max_articles_per_minute = 30
        total_articles = 120
        all_results = []

        for _ in range(total_articles // max_articles_per_minute):
            search_results = exa.search_and_contents(
                query=search_query,
                use_autoprompt=True,
                start_published_date=date_cutoff,
                text={"include_html_tags": False, "max_characters": 8000},
                limit=max_articles_per_minute  # Control the number of articles retrieved per minute
            )
            all_results.extend(search_results)
            time.sleep(60)  # Cool down for one minute before the next batch

        return all_results
    
class FindSimilar(BaseTool):
    name:str = "Find Similar Tool"
    description: str = (
        "Searches for similar articles to a given article using the Exa APi. Takes in a url of the artile."
        )
    def _run(self, article_url: str) -> str:
        one_week_ago = datetime.now() - timedelta(days=1)
        date_cutoff = one_week_ago.strftime("%Y-%m-%d")
        exa = Exa(api_key=os.getenv("EXA_API_KEY"))
        search_results = exa.find_similar(
            url = article_url,
            start_published_date = date_cutoff
        )
        return search_results

class GetContents(BaseTool):
    name:str = "Get Contents Tool"
    description: str = (
            "Gets the contents of a specific article using the Exa API. Takes in the ID of the article in a list, like this: ['https://www.cnbc.com/2024/04/18/my-news-story']."
        )
    def _run(self, article_ids: list) -> str:
        exa = Exa(api_key=os.getenv("EXA_API_KEY"))
        contents = exa.get_contents(article_ids)
        return contents

# if __name__=="__main__":
#     # search_and_contents = SearchAndContents()
#     # search_results = search_and_contents.run("Latest News On Job Market in Singapore")
#     # print(search_results)
#     # find_similar = FindSimilar()
#     # similar_results = find_similar.run(article_url = "https://www.reuters.com/business/finance/sgx-has-no-immediate-plans-allow-crypto-listings-ceo-says-2024-07-09/")
#     # print(similar_results)
#     get_contents = GetContents()
#     contents = get_contents.run(article_ids = ["https://www.reuters.com/business/finance/sgx-has-no-immediate-plans-allow-crypto-listings-ceo-says-2024-07-09/"])
#     print(contents)
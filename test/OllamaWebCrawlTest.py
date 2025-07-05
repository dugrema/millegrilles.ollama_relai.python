import asyncio
import json
import urllib.parse
import requests

from bs4 import BeautifulSoup
from ollama import AsyncClient

# CONST_MODEL = "deepseek-r1:8b-0528-qwen3-q8_0"
# CONST_MODEL = "gemma3:27b-it-qat-LOPT"
# CONST_MODEL = "gemma3:12b-it-qat-LOPT"
CONST_MODEL = "gemma3n:e4b-it-q8_0-LOPT"
# CONST_MODEL = "gemma3n:e2b-it-q4_K_M-LOPT"
# CONST_MODEL = "llama3.2:3b-instruct-q8_0"

CONST_SUMMARY_NUM_PREDICT_KEYWORDS = 512
CONST_SUMMARY_NUM_PREDICT_RESPONSE = 1536
CONST_THINK = None

CONST_LIMIT_ARTICLE = 10_000

SYSTEM_KEYWORDS = """
You are an AI assistant that generates a summary and keywords to find relevant information on a local instance of Wikipedia.
You will output plain JSON as a response, this is not an interactive prompt.

# Instructions

* Generate a short summary, between 75 and 500 words on the topic of the query, depending on complexity of the topic.
* Generate 3 keywords in English to apply to a local search engine.
* Each keyword should be a single word unless it is a name.
** Do not translate the name of people, use it as provided.
** If the keyword is a name in another language for a place, technology, geographic feature, etc. then translate it to English.
* Each keyword *must* be directly related to the user query.
* You must output plain JSON. Do not use markdown (md) formatting in the response. 
* Return the following response in JSON format: {"s": "YOUR SUMMARY", "kw": ["word1", "word2", "word3"]}.
* Only if the user query is not a question on a factual topic, return a null list of keywords.
"""

SYSTEM_FIND_PAGE = """
You are an AI assistant that identifies a page to use for fact-checking a user prompt.
You are provided with a user prompt and a list of search results.

# Instructions

* Check the user prompt.
* From the list of search results, identify a URL that most closely matches the user query. 
* If a page name matches names or concepts directly from the user query, prefer that link.
* Do not use markdown (md) formatting in the response. You must output plain JSON.
* Return that url using plain JSON in the form of: {"url": "/kiwix/content/wikipedia_en_all_maxi_2023-11/A/XXXXXXXXX"}
* Make sure to *copy the entire* "href" element properly, do not truncate the url.
"""

SYSTEM_USE_ARTICLE = """
You are an AI assistant that fact checks affirmations and explanations.
Since you have a cutoff date and your information may also be incomplete, a authoritative article was later selected to assist
you in providing the most factual and up to date information. This article is in the element authority.

This is a fact-checking step for the content of valueToCheck.  

# Instructions

* The valueToCheck is a 500 word or less answer to the user query. Check for contradictions only; the valueToCheck is meant to be incomplete compared to the full article. 
* Only if the article contradicts the valueToCheck response, put a disclaimer stating that the checked content is inaccurate and then list the inaccuracies using the article.
* Make sure the valueToCheck answered the user's query directly and factually.
* The article's information is authoritative on its topic, it's main purpose is to fact-check the valueToCheck response on that topic.
* Only if the article does not correspond to the user's question or intent, indicate that you have not been able to complete the fact-check.
* Respond in plain text.
"""

async def crawl_search(keywords: list[str]):
    params = [
        "books.name=wikipedia_en_all_maxi_2023-11",
        f"pattern={urllib.parse.quote_plus(", ".join(keywords))}",
        "userlang=en"
    ]
    params = "&".join(params)
    url = f"https://libs.millegrilles.com/kiwix/search?{params}"

    response = await asyncio.to_thread(requests.get, url)
    response.raise_for_status()

    data = response.text
    soup = BeautifulSoup(data, "html.parser")
    results = soup.select_one(".results")
    results_html = str(results)

    # print(f"HTML result page\n{results_html}")

    return results_html

async def crawl_get_page(client: AsyncClient, user_prompt: str, search_results: str):
    prompt = f"""
<query>
{user_prompt}
</query>
<searchResults>
{search_results[:CONST_LIMIT_ARTICLE]}
</searchResults>
"""

    output = await client.generate(
        model=CONST_MODEL,
        system=SYSTEM_FIND_PAGE,
        prompt=prompt,
        think=CONST_THINK,
        options={"num_predict": CONST_SUMMARY_NUM_PREDICT_KEYWORDS, "temperature": 0.01},
    )

    print(f"Chosen page: {output.response}")
    response_dict = json.loads(output.response)
    page_url = response_dict['url']
    if not page_url.startswith('/'):
        page_url = f"/{page_url}"
    page_url = page_url.replace('/kiwix/content/wiki/', '/kiwix/content/wikipedia_en_all_maxi_2023-11/')
    url = f"https://libs.millegrilles.com{page_url}"

    response = await asyncio.to_thread(requests.get, url)
    response.raise_for_status()

    data = response.text
    soup = BeautifulSoup(data, "html.parser")
    content = soup.select_one("#mw-content-text")
    content_string = content.text

    return url, content_string

async def review_article(client: AsyncClient, user_prompt: str, assistant_response: str, article: str):

    article_truncated = article[:CONST_LIMIT_ARTICLE]

    prompt = f"""
<userProfile>
user_language: en_CA
</userProfile>

<query>
{user_prompt}
</query>

<authority>
{article_truncated}
</authority>

<valueToCheck>
{assistant_response}
</valueToCheck>
"""

    # print(article_truncated)
    print(f"Prompt len: {len(prompt)}, System prompt len: {len(SYSTEM_USE_ARTICLE)}")

    stream = await client.generate(
        model=CONST_MODEL,
        system=SYSTEM_USE_ARTICLE,
        think=CONST_THINK,
        prompt=prompt,
        stream=True,
        options={"num_predict": CONST_SUMMARY_NUM_PREDICT_RESPONSE, "temperature": 0.01},
    )

    async for value in stream:
        yield value

async def query_with_keywords():

    client = AsyncClient()

    prompt = "Where did the St. Lawrence river in Canada get its name from?"
    # prompt = "D'où vient le nom Saint-Laurent pour le fleuve canadien?"
    # prompt = "What was the cause of the French Revolution?"
    # prompt = "What is retrieval augmented generation (RAG)?"
    # prompt = "When did aliens invade earth?"
    # prompt = "Are UFO encounters considered real or is it a hoax?"
    # prompt = "Hi"
    # prompt = "What is your cutoff date?"
    # prompt = "What is taxonomy in biology?"

    output = await client.generate(
        model=CONST_MODEL,
        system=SYSTEM_KEYWORDS,
        prompt=prompt,
        think=CONST_THINK,
        options={"num_predict": CONST_SUMMARY_NUM_PREDICT_KEYWORDS, "temperature": 0.01},
    )

    print(f"Summary/Keywords: {output.response}")
    try:
        response_dict = json.loads(output.response)
    except json.decoder.JSONDecodeError:
        # Try to remove markdown
        response_str = output.response.replace("```json", "").replace('```', "")
        response_dict = json.loads(response_str)

    assistant_response = response_dict["s"]
    print(f"Initial response\n{assistant_response}\n")

    keywords = response_dict['kw']
    if not keywords or len(keywords) == 0:
        print("*** Response done ***")
        return  # Done

    search_result = await crawl_search(keywords)
    url, article = await crawl_get_page(client, prompt, search_result)

    print(f"Verifying response with url: {url}")

    flag_init = False
    async for chunk in review_article(client, prompt, assistant_response, article):
        if not flag_init:
            print("*** Response ***\n")
            flag_init = True
        print(chunk.response, end='')
    print("\n*** Response done ***")

async def main():
    await query_with_keywords()


if __name__ == '__main__':
    asyncio.run(main())

import asyncio
import json
import logging

import requests
import urllib.parse
from urllib.parse import urlparse

from typing import AsyncGenerator, Any, Union, Optional

from bs4 import BeautifulSoup
from ollama import GenerateResponse, AsyncClient

from millegrilles_ollama_relai.OllamaContext import OllamaContext
from millegrilles_ollama_relai.OllamaInstanceManager import OllamaInstance
from millegrilles_ollama_relai.Structs import SummaryKeywords, LinkIdPicker, KnowledgeBaseSearchResponse, \
    MardownTextResponse, MatchResult
from millegrilles_ollama_relai.Util import check_token_len
from millegrilles_ollama_relai.prompts.knowledge_base_prompt import KNOWLEDGE_BASE_INITIAL_SUMMARY_PROMPT, \
    KNOWLEDGE_BASE_FIND_PAGE_PROMPT, KNOWLEDGE_BASE_SYSTEM_USE_ARTICLE_PROMPT, KNOWLEDGE_BASE_SUMMARY_ARTICLE_PROMPT, \
    KNOWLEDGE_BASE_CHECK_ARTICLE_PROMPT

CONST_DEFAULT_CONTEXT_LENGTH = 8192

CONST_KIWIX_WIKIPEDIA_EN_SEARCH_LABEL = 'kiwixWikipediaEnSearch'

class KnowledgBaseHandler:

    def __init__(self, context: OllamaContext, client: AsyncClient):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context = context
        self.__client = client
        # self.__server_hostname = 'https://libs.millegrilles.com'

        self.__model = context.model_configuration.get('knowledge_model_name') or context.chat_configuration['default_model']

        self.__context_length = CONST_DEFAULT_CONTEXT_LENGTH
        self.__num_predict_summary = 768
        self.__num_predict_response = 1536

    @property
    def _limit_article(self):
        return int(2.8 * (self.__context_length - 1024))

    async def parse_query(self, query: str) -> SummaryKeywords:
        output = await self.__client.generate(
            model=self.__model,
            system=KNOWLEDGE_BASE_INITIAL_SUMMARY_PROMPT,
            prompt=query,
            think=None,
            format=SummaryKeywords.model_json_schema(),
            options={"num_predict": self.__num_predict_summary, "temperature": 0.01},
        )
        response = SummaryKeywords.model_validate_json(output.response)
        return response

    async def search_query(self, topic: str, query: str, search_results: list[dict]):
        # Extract the title, description and linkId to put in the context
        mapped_results = [{
            'link_id': r['linkId'],
            'title': r['title'],
        } for r in search_results]

        prompt = f"""
    <searchResults>
    {json.dumps(mapped_results)}
    </searchResults>
    """

        params = {'query': query}
        formatted_system_prompt = KNOWLEDGE_BASE_FIND_PAGE_PROMPT.format(**params)
        output = await self.__client.generate(
            model=self.__model,
            system=formatted_system_prompt,
            prompt=prompt,
            think=None,
            format=LinkIdPicker.model_json_schema(),
            options={"num_predict": self.__num_predict_summary, "temperature": 0.01},
        )

        # Fetch the selected link by linkId
        response_dict = LinkIdPicker.model_validate_json(output.response)
        link_ids = response_dict.link_ids
        chosen_links = [l for l in search_results if l['linkId'] in link_ids]
        self.__logger.debug("Chosen links: %s", json.dumps(chosen_links, indent=2))

        selected_article = await self.check_articles(query, chosen_links)
        # chosen_link = chosen_links[0]
        # page_url = chosen_link['url']
        #
        # url = f"{self.__server_hostname}{page_url}"
        # # response = await asyncio.to_thread(requests.get, url)
        # # response.raise_for_status()
        # #
        # # data = response.text
        # # soup = BeautifulSoup(data, "html.parser")
        # # content = soup.select_one("#mw-content-text")
        # # content_string = content.text
        # content_string = await asyncio.to_thread(fetch_page_content, url)

        if selected_article:
            return selected_article  # chosen_link, url, content_string
        else:
            return None

    async def check_articles(self, query: str, articles: list[dict]) -> Optional[dict]:
        for article in articles:
            reference_url = article['url']
            parsed_url = urlparse(reference_url)
            local_hostname = urlparse(self.__context.url_configuration['urls'][CONST_KIWIX_WIKIPEDIA_EN_SEARCH_LABEL])
            if parsed_url.hostname and parsed_url.hostname != local_hostname.hostname:
                raise ValueError(f'Unauthorized URL provided: {reference_url}')
            url = f"{local_hostname.scheme}://{local_hostname.hostname}{parsed_url.path}"
            if parsed_url.fragment:
                url += f'#{parsed_url.fragment}'
            if parsed_url.query:
                url += f'?{parsed_url.query}'

            reference_content = await asyncio.to_thread(fetch_page_content, url)
            article_truncated = reference_content[:self._limit_article]
            prompt = f"""
<article>
{article_truncated}
</article>
            """

            params = {'query': query}
            system_prompt = KNOWLEDGE_BASE_CHECK_ARTICLE_PROMPT.format(**params)
            output = await self.__client.generate(
                model=self.__model,
                system=system_prompt,
                prompt=prompt,
                think=None,
                format=MatchResult.model_json_schema(),
                options={"num_predict": self.__num_predict_summary, "temperature": 0.01},
            )

            result_value = MatchResult.model_validate_json(output.response)
            if result_value.match:
                article['content'] = article_truncated
                article['url'] = url
                return article
            else:
                self.__logger.debug("Article %s does not answer the user's query", article['title'])

        return None


    async def search_topic(self, topic: str):
        search_url = self.__context.url_configuration['urls'][CONST_KIWIX_WIKIPEDIA_EN_SEARCH_LABEL]
        params = {'query': urllib.parse.quote_plus(topic)}
        search_url = search_url.format(**params)

        # params = [
        #     "books.name=wikipedia_en_all_maxi_2023-11",
        #     # f"pattern={urllib.parse.quote_plus(", ".join(keywords))}",
        #     f"pattern={urllib.parse.quote_plus(topic)}",
        #     "userlang=en",
        #     "start=1",
        #     "pageLength=30",
        # ]
        params = "&".join(params)
        # search_url = f"{self.__server_hostname}/kiwix/search?{params}"

        response = await asyncio.to_thread(requests.get, search_url)
        response.raise_for_status()

        data = response.text
        soup = BeautifulSoup(data, "html.parser")
        results = soup.select_one(".results")

        items = results.select("li")

        links = list()

        item_id = 1
        for item in items:
            anchor = item.select_one("a")
            url = anchor.attrs.get("href")
            title = anchor.text.strip()
            description = item.select_one("cite").text[:100]
            links.append({"linkId": item_id, "title": title, "description": description, "url": url})
            item_id += 1

        return search_url, links

    async def review_article(self, user_prompt: str, language: str, article: str):
        article_truncated = article[:self._limit_article]

        prompt = f"""
    <query>
    {user_prompt}
    </query>

    <article>
    {article_truncated}
    </article>

    <instructions>
    Answer in {language}.
    Put the header # Review.
    Follow the system prompt.
    </instructions>
    """

        params = {'language': language}
        formatted_prompt = KNOWLEDGE_BASE_SUMMARY_ARTICLE_PROMPT.format(**params)

        stream = await self.__client.generate(
            model=self.__model,
            system=formatted_prompt,
            think=None,
            prompt=prompt,
            stream=True,
            options={
                "num_predict": self.__num_predict_response,
                "temperature": 0.1,
            },
        )

        async for value in stream:
            yield value

    async def check_summary(self, user_prompt: str, language: str, assistant_response: str, article: str):
        article_truncated = article[:self._limit_article]

        prompt = f"""
    <query>
    {user_prompt}
    </query>

    <reference>
    {article_truncated}
    </reference>

    <summary>
    {assistant_response}
    </summary>

    <instructions>
    Important: answer in **{language}**.
    Put the heading: # Verification
    Proceed to verify the summary element using the reference element as per system prompt.
    </instructions>
    """

        # print(article_truncated)
        # token_len = check_token_len(prompt + KNOWLEDGE_BASE_SYSTEM_USE_ARTICLE_PROMPT)
        # print(f"Prompt len: {len(prompt)}, System prompt len: {len(KNOWLEDGE_BASE_SYSTEM_USE_ARTICLE_PROMPT)}, Total token len: {token_len}")
        # if token_len > CONST_CONTEXT_LEN - 700:  # Reserve 700 token for model template (not exact)
        #    raise Exception("Prompt too long")

        params = {'language': language}
        formatted_prompt = KNOWLEDGE_BASE_SYSTEM_USE_ARTICLE_PROMPT.format(**params)

        stream = await self.__client.generate(
            model=self.__model,
            system=formatted_prompt,
            think=None,
            prompt=prompt,
            stream=True,
            options={
                "num_predict": self.__num_predict_response,
                "temperature": 0.1,
            },
        )

        async for value in stream:
            yield value

    async def __process(self, instance: OllamaInstance, query: str) -> AsyncGenerator[Union[SummaryKeywords, KnowledgeBaseSearchResponse, GenerateResponse, MardownTextResponse], Any]:
        # Find model, update context_length
        model_info = instance.get_model(self.__model)
        context_length = model_info.context_length or self.__context_length
        self.__context_length = context_length

        # Extract information from the user query
        summary = await self.parse_query(query)
        yield summary

        # Perform search
        if summary.t is None:
            return  # Done

        reference_url = summary.url

        if not reference_url:
            search_url, links = await self.search_topic(summary.q)
            # Find matching page
            # chosen_link, reference_url, reference_content = await self.search_query(summary.q, query, links)
            selected_article = await self.search_query(summary.q, query, links)
            if selected_article is None:
                yield MardownTextResponse(text='**No match** found for this topic. You may want to verify this information from a reliable source.')
                return  # Done
            reference_url = selected_article['url']
            reference_content = selected_article['content']
            yield KnowledgeBaseSearchResponse(search_url=search_url, reference_title=selected_article['title'], reference_url=reference_url)
        else:
            parsed_url = urlparse(reference_url)
            local_hostname = urlparse(self.__context.url_configuration['urls'][CONST_KIWIX_WIKIPEDIA_EN_SEARCH_LABEL])
            if parsed_url.hostname and parsed_url.hostname != local_hostname.hostname:
                raise ValueError(f'Unauthorized URL provided: {reference_url}')
            url = f"{local_hostname.scheme}://{local_hostname.hostname}{parsed_url.path}"
            if parsed_url.fragment:
                url += f'#{parsed_url.fragment}'
            if parsed_url.query:
                url += f'?{parsed_url.query}'
            reference_content = await asyncio.to_thread(fetch_page_content, url)
            yield KnowledgeBaseSearchResponse(search_url=None, reference_title="Provided link", reference_url=reference_url)

        # Fact check summary with article
        async for chunk in self.review_article(query, summary.l, reference_content):
            yield chunk

        # Fact check summary with article
        yield MardownTextResponse(text="\n\n")
        async for chunk in self.check_summary(query, summary.l, summary.s, reference_content):
            yield chunk

    async def run_query(self, instance: OllamaInstance, query: str) -> AsyncGenerator[MardownTextResponse, Any]:
        async for chunk in self.__process(instance, query):
            if isinstance(chunk, SummaryKeywords):
                content = f'# Summary\n\nNotice: This **may be inaccurate** and will be double-checked further down.\n\n{chunk.s}\n\nNow looking for a reference on: **{chunk.t}**, searching using: {chunk.q}\n'
                yield MardownTextResponse(text=content, complete_block=True)
            elif isinstance(chunk, KnowledgeBaseSearchResponse):
                content = f'Reference: [{chunk.reference_title}]({chunk.reference_url})'
                yield MardownTextResponse(text=content, complete_block=True)
            elif isinstance(chunk, GenerateResponse):
                # Final pass with fact checking stream
                yield MardownTextResponse(text=chunk.response)
            elif isinstance(chunk, MardownTextResponse):  # Passthrough for formatting
                yield chunk
            else:
                raise ValueError("Unsupported chunk type", chunk)


def fetch_page_content(url: str):
    response = requests.get(url)
    response.raise_for_status()

    data = response.text
    soup = BeautifulSoup(data, "html.parser")
    content = soup.select_one("#mw-content-text")
    content_string = content.text

    return content_string

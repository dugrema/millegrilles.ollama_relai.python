import asyncio
import json
import logging

import openai
import pydantic
import requests
import urllib.parse

from urllib.parse import urlparse

from typing import AsyncGenerator, Any, Union, Optional

from bs4 import BeautifulSoup
from ollama import GenerateResponse

from millegrilles_ollama_relai.Constantes import MODEL_TYPE_KNOWLEDGE
from millegrilles_ollama_relai.InstancesDao import InstanceDao, MessageWrapper, OllamaModelParams
from millegrilles_ollama_relai.OllamaContext import OllamaContext
from millegrilles_ollama_relai.OllamaInstanceManager import OllamaInstance
from millegrilles_ollama_relai.Structs import SummaryKeywords, LinkIdPicker, KnowledgeBaseSearchResponse, \
    MardownTextResponse, MatchResult
from millegrilles_ollama_relai.Util import cleanup_json_output
from millegrilles_ollama_relai.prompts.knowledge_base_prompt import KNOWLEDGE_BASE_INITIAL_SUMMARY_PROMPT, \
    KNOWLEDGE_BASE_FIND_PAGE_PROMPT, KNOWLEDGE_BASE_SUMMARY_ARTICLE_PROMPT, \
    KNOWLEDGE_BASE_CHECK_ARTICLE_PROMPT, KNOWLEDGE_BASE_SUBSEQUENT_SUMMARY_SYSTEM, KNOWLEDGE_BASE_SUBSEQUENT_SUMMARY_PROMPT

CONST_DEFAULT_CONTEXT_LENGTH = 8192
CONST_TEMPERATURE = 0.2
CONST_TEMPERATURE_SEARCH = 1.0

CONST_KIWIX_WIKIPEDIA_EN_SEARCH_LABEL = 'kiwixWikipediaEnSearch'

class KnowledgBaseHandler:

    def __init__(self, context: OllamaContext, client: InstanceDao):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context = context
        self.__client = client
        # self.__server_hostname = 'https://libs.millegrilles.com'

        # self.__model: Optional[str] = None
        self.__model_info: Optional[OllamaModelParams] = None

        self.__context_length = CONST_DEFAULT_CONTEXT_LENGTH
        self.__num_predict_summary = 768
        self.__num_predict_response = 1536

    @property
    def _limit_article(self):
        return int(2.5 * (self.__context_length - 1024))

    @property
    def model_name(self) -> str:
        if self.__model_info:
            return self.__model_info.name
        raise Exception('No model selected')

    def select_model(self, instance: OllamaInstance) -> OllamaModelParams:
        model = None
        try:
            model = self.__context.model_configuration[MODEL_TYPE_KNOWLEDGE]
            model_info = instance.get_model(model)
            self.__model_info = model_info
        except KeyError:
            self.__logger.warning("Knowledge model %s not available, falling back to default", model)
            try:
                model = self.__context.chat_configuration['model_name']
            except KeyError as ke:
                self.__logger.error("Default model not configured available")
                raise ke
            try:
                model_info = instance.get_model(model)
            except KeyError as ke:
                self.__logger.error("Default model %s not available", model)
                raise ke
            self.__model_info = model_info

        return model_info

    async def parse_query(self, query: str, previous_summary: list[SummaryKeywords]) -> SummaryKeywords:
        unsuccessful_keywords = '\n'.join([s.t for s in previous_summary]) if len(previous_summary) > 0 else None
        if unsuccessful_keywords:
            params = {'query': query, 'previous': unsuccessful_keywords}
            command_prompt = KNOWLEDGE_BASE_SUBSEQUENT_SUMMARY_PROMPT.format(**params)
            output = await self.__client.generate(
                model=self.model_name,
                system=KNOWLEDGE_BASE_SUBSEQUENT_SUMMARY_SYSTEM,
                prompt=command_prompt,
                think=None,
                response_format=SummaryKeywords,
                max_len=self.__num_predict_summary,
                temperature=CONST_TEMPERATURE_SEARCH,
            )
        else:
            output = await self.__client.generate(
                model=self.model_name,
                system=KNOWLEDGE_BASE_INITIAL_SUMMARY_PROMPT,
                prompt=query,
                think=None,
                response_format=SummaryKeywords,
                max_len=self.__num_predict_summary,
                temperature=CONST_TEMPERATURE,
            )

        try:
            content = cleanup_json_output(output.message['content'])
        except IndexError as e:
            self.__logger.error("No query information received: {}", output.message['content'])
            raise e

        response = SummaryKeywords.model_validate_json(content)
        return response

    async def search_query(self, query: str, search_results: list[dict]) -> AsyncGenerator[dict, Any]:
        # Extract the title, description and linkId to put in the context
        mapped_results = [{
            'link_id': r['linkId'],
            'title': r['title'],
        } for r in search_results]

        try:
            mapped_results = mapped_results[0:4]  # Keep 4 at most
        except IndexError:
            pass

        prompt = f"""
    <searchResults>
    {json.dumps(mapped_results)}
    </searchResults>
    """

        params = {'query': query}
        formatted_system_prompt = KNOWLEDGE_BASE_FIND_PAGE_PROMPT.format(**params)
        output = await self.__client.generate(
            model=self.model_name,
            system=formatted_system_prompt,
            prompt=prompt,
            think=None,
            response_format=LinkIdPicker,
            max_len=self.__num_predict_summary,
            temperature=CONST_TEMPERATURE,
        )

        # Fetch the selected link by linkId
        try:
            content = cleanup_json_output(output.message['content'])
            if content == '':
                return  # Nothing produced
        except IndexError:
            return  # Nothing produced

        try:
            response_dict = LinkIdPicker.model_validate_json(content)
            link_ids = response_dict.link_ids
            chosen_links = [l for l in search_results if l['linkId'] in link_ids]
            self.__logger.debug("Chosen links: %s", json.dumps(chosen_links, indent=2))
        except pydantic.ValidationError:
            self.__logger.warning("LinkIdPicker.model_validate_json: Invalid response format: {}", content)
            return

        # selected_articles = await self.check_articles(query, chosen_links)
        async for article in self.check_articles(query, chosen_links):
            yield article

    async def check_articles(self, query: str, articles: list[dict]) -> AsyncGenerator[dict, Any]:
        # summarys: list[dict] = list()

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
            try:
                output = await self.__client.generate(
                    model=self.model_name,
                    system=system_prompt,
                    prompt=prompt,
                    think=None,
                    response_format=MatchResult,
                    max_len=self.__num_predict_summary,
                    temperature=CONST_TEMPERATURE,
                )
            except openai.BadRequestError:
                self.__logger.exception("Error processing article, skipping")
                continue

            content = cleanup_json_output(output.message['content'])
            result_value = MatchResult.model_validate_json(content)
            if result_value.match:
                article['content'] = article_truncated
                article['summary'] = result_value.summary
                article['url'] = url
                # return article
                # summarys.append(article)
                self.__logger.debug("Article chosen: %s", article)
                yield article
            else:
                self.__logger.debug("Article %s does not answer the user's query", article['title'])

        # if len(summarys) == 0:
        #     return None
        #
        # return summarys


    async def search_topic(self, topic: str):
        search_url = self.__context.url_configuration['urls'][CONST_KIWIX_WIKIPEDIA_EN_SEARCH_LABEL]
        params = {'query': urllib.parse.quote_plus(topic)}
        search_url = search_url.format(**params)

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
    Follow the system prompt.
    </instructions>
    """

        params = {'language': language}
        formatted_prompt = KNOWLEDGE_BASE_SUMMARY_ARTICLE_PROMPT.format(**params)

        stream = await self.__client.generate(
            model=self.model_name,
            system=formatted_prompt,
            think=None,
            prompt=prompt,
            stream=True,
            max_len=self.__num_predict_response,
            temperature=0.1,
        )

        async for value in stream:
            yield value

    async def __process(self, instance: OllamaInstance, query: str) -> AsyncGenerator[Union[SummaryKeywords, KnowledgeBaseSearchResponse, GenerateResponse, MardownTextResponse], Any]:
        # Find model, update context_length
        try:
            self.__logger.info("Available models: %s", [': '.join([m.id, m.name]) for m in instance.models])
            model_info = self.select_model(instance)
        except KeyError as ke:
            self.__logger.error("No model available for running the query: %s", ke)
            return

        # model_info = instance.get_model(self.__model)
        context_length = model_info.context_length or self.__context_length
        self.__context_length = context_length

        previous_summary: list[SummaryKeywords] = list()

        # Extract information from the user query
        try:
            summary = await self.parse_query(query, previous_summary)
        except IndexError:
            return  # No more search patterns to try

        # Perform search
        if summary.t is None:
            return  # Done

        reference_url = summary.url

        if not reference_url:
            reference_content_list = list()
            for i in range(0, 4):
                # Support iterative refining of keywords
                if len(previous_summary) > 0:
                    summary = await self.parse_query(query, previous_summary)

                previous_summary.append(summary)
                yield summary  # New search keywords

                search_url, links = await self.search_topic(summary.t)

                # selected_articles = list()

                # Find matching page
                async for article in self.search_query(query, links):
                    # selected_articles.append(article)
                    reference_content_list.append('\n'.join([article['title'], article['summary']]))
                    yield KnowledgeBaseSearchResponse(search_url=search_url, reference_title=article['title'], reference_url=article['url'], summary=article['summary'])

                if len(reference_content_list) == 0:
                    continue  # Retry
                elif len(reference_content_list) == 1:
                    # Only one article. The summary is the answer.
                    yield MardownTextResponse(text='\n\n# Answer\nNo further references found. The answer is the summary above.')
                    return  # Done

                reference_content = '\n\n'.join(reference_content_list)
                break  # Multiple articles, produce a summary of
            else:
                yield MardownTextResponse(text='**No match** found for this topic.')
                return  # Done
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
        yield MardownTextResponse(text='\n\n# Answer\n')
        async for chunk in self.review_article(query, summary.l, reference_content):
            yield chunk

    async def run_query(self, instance: OllamaInstance, query: str) -> AsyncGenerator[MardownTextResponse, Any]:
        async for chunk in self.__process(instance, query):
            if isinstance(chunk, SummaryKeywords):
                content = f'Looking for references on: **{chunk.t}**\n'
                yield MardownTextResponse(text=content, complete_block=True)
            elif isinstance(chunk, KnowledgeBaseSearchResponse):
                content = f'Reference: [{chunk.reference_title}]({chunk.reference_url})\n\n{chunk.summary}'
                yield MardownTextResponse(text=content, complete_block=True)
            elif isinstance(chunk, MessageWrapper):
                yield MardownTextResponse(text=chunk.message['content'])
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

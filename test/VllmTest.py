import asyncio
import base64
import json
from typing import Optional

import tiktoken

import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompt_values import ChatPromptValue
from openai import AsyncOpenAI

from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletion, ChatCompletionMessage, \
    ChatCompletionUserMessageParam, ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.shared_params import ResponseFormatJSONSchema
from openai.types.shared_params.response_format_json_schema import JSONSchema
from pydantic import BaseModel

from millegrilles_ollama_relai.Util import check_token_len

CONST_MODEL='google/gemma-3-4b-it-qat-q4_0-unquantized'


def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url='http://bureau1.maple.maceroc.com:8001/v1', api_key='DUMMY')


async def completions_1(client: AsyncOpenAI):
    response = await client.completions.create(
        model=CONST_MODEL,
        prompt="Hi.",
        stream=False,
        max_tokens=2048,
    )
    # content = response.model_dump_json()
    content = response.choices[0]
    print(f"Response: {content.text}")


async def chat_1(client: AsyncOpenAI):
    response = await client.chat.completions.create(
        messages=[
            ChatCompletionSystemMessageParam(content="Your are a helpful assistant.", role="system"),
            ChatCompletionUserMessageParam(content="Hi.", role="user"),
        ],
        model=CONST_MODEL,
        max_tokens=2048,
    )
    # content = response.model_dump_json()
    content = response.choices[0]
    print(f"Response: {content.message}")


async def chat_2(client: AsyncOpenAI):
    response = await client.chat.completions.create(
        messages=[
            ChatCompletionSystemMessageParam(content="Your are a helpful assistant.", role="system"),
            ChatCompletionUserMessageParam(content="What is the cause of the war in Syria?", role="user"),
        ],
        model=CONST_MODEL,
        max_tokens=2048,
        stream=True
    )
    async for chunk in response:
        value = chunk.choices[0]
        print(value.delta.content, end='')
    print()


async def chat_image(client: AsyncOpenAI):

    with open("/home/mathieu/Pictures/generes/txt2img-images/2025-05-19/00009-341513889.png", "rb") as image_file:
        b64_image = base64.b64encode(image_file.read()).decode("utf-8")

    response = await client.chat.completions.create(
        messages=[
            ChatCompletionSystemMessageParam(content="Your are a helpful assistant.", role="system"),
            ChatCompletionUserMessageParam(
                role="user",
                content=[
                    ChatCompletionContentPartTextParam(type="text", text="Describe this image."),
                    ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url=ImageURL(url=f"data:image/png;base64,{b64_image}", detail="high")
                    )
                ]
            ),
        ],
        model=CONST_MODEL,
        max_tokens=2048,
    )
    # content = response.model_dump_json()
    content = response.choices[0]
    print(f"Response: {content.message}")


class Conflict(BaseModel):
    year: int
    description: str


class YearsAnswer(BaseModel):
    years: list[Conflict]


async def chat_formatted_1(client: AsyncOpenAI):

    json_schema_mapped = YearsAnswer.model_json_schema()
    json_schema = JSONSchema(name='response', schema=json_schema_mapped)

    response = await client.chat.completions.create(
        messages=[
            ChatCompletionSystemMessageParam(content="Your are a helpful assistant.", role="system"),
            ChatCompletionUserMessageParam(content="What years had a new conflicts erupt in the middle east?", role="user"),
        ],
        model=CONST_MODEL,
        max_tokens=2048,
        response_format=ResponseFormatJSONSchema(
            json_schema=json_schema,
            type="json_schema"
        )
    )
    # content = response.model_dump_json()
    content = response.choices[0]
    print(f"Response: {content.message}")
    value = YearsAnswer.model_validate_json(content.message.content)
    print('Conflicts\n-----')
    for conflict in value.years:
        print(f"{conflict.year}: {conflict.description}")


class SummaryResponse(BaseModel):
    summary: str
    content_iso_language: str
    # empty_document: Optional[bool]


async def chat_large_prompt(client: AsyncOpenAI):
    pdf_files_path = [
        '/home/mathieu/Downloads/Articles/Mai 2025/NIST.FIPS.186-5.pdf',
        '/home/mathieu/Downloads/Articles/Mai 2025/Journal of Nuclear Medicine - Issue 2024-08.pdf',
        '/home/mathieu/Downloads/Articles/Mai 2025/Apple Research 2025-06-09 - the-illusion-of-thinking.pdf',
        '/home/mathieu/Downloads/Articles/Mai 2025/Aurores boréales au Québec _ quels secteurs pourraient être chanceux_ - MétéoMédia.pdf',
        '/home/mathieu/Downloads/Articles/Mai 2025/The Pulse #137_ Builder.ai did not “fake AI with 700 engineers”.pdf',
    ]

    with open("/home/mathieu/Pictures/generes/txt2img-images/2025-05-19/00009-341513889.png", "rb") as image_file:
        b64_image = base64.b64encode(image_file.read()).decode("utf-8")

    extraction_kwargs = {'strict': False}
    for pdf_file_path in pdf_files_path:
        loader = PyPDFLoader(pdf_file_path, mode="single", extraction_kwargs=extraction_kwargs)
        document_list = await asyncio.to_thread(loader.load)
        content = document_list[0].page_content

        encoding = tiktoken.encoding_for_model("text-embedding-3-small")

        SYSTEM_PROMPT = "Summarize the file content from the content element. Provide the content language in ISO format, e.g. en, es, fr, jp."
        system_prompt_len = len(encoding.encode(SYSTEM_PROMPT))

        CTX_LEN = 10240
        RESPONSE_LEN = 1024
        TEMPLATE_LEN = 955

        token_pdf = CTX_LEN - RESPONSE_LEN - TEMPLATE_LEN - system_prompt_len

        encoded_content = encoding.encode(content)
        if len(encoded_content) > token_pdf:
            original_len = len(content)
            encoded_content = encoded_content[:token_pdf]
            content = encoding.decode(encoded_content)
            print(f"{pdf_file_path}: Content truncated from {original_len} to {len(content)}")
        else:
            print(f"{pdf_file_path}: Document len: {len(content)}")

        json_schema_mapped = SummaryResponse.model_json_schema()
        json_schema = JSONSchema(name='summary', schema=json_schema_mapped)

        try:
            response = await client.chat.completions.create(
                messages=[
                    ChatCompletionSystemMessageParam(content=SYSTEM_PROMPT, role="system"),
                    # ChatCompletionUserMessageParam(content=f"<content>{content}</content>", role="user"),
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=[
                            # ChatCompletionContentPartImageParam(
                            #     type="image_url",
                            #     image_url=ImageURL(url=f"data:image/png;base64,{b64_image}", detail="high")
                            # ),
                            ChatCompletionContentPartTextParam(type="text", text="<content>{content}</content>"),
                        ]
                    ),
                ],
                model=CONST_MODEL,
                max_tokens=RESPONSE_LEN,
                response_format=ResponseFormatJSONSchema(
                    json_schema=json_schema,
                    type="json_schema"
                )
            )
            content = response.choices[0]
            print(f"{pdf_file_path} Response: {content.message}")
            value = SummaryResponse.model_validate_json(content.message.content)
            language = value.content_iso_language
            # empty_doc = value.empty_document
            # print(f"Language: {language}, empty: {empty_doc}")
            print(f"Language: {language}")
            summary = value.summary
            print(f"Summary\n{summary}\n----------")
        except openai.BadRequestError as bre:
            print(f"Bad request error: {bre}")
            error_content = bre.response.json()
            print(error_content)

async def main():
    client = get_client()
    # await completions_1(client)
    # await chat_1(client)
    # await chat_2(client)
    # await chat_image(client)
    # await chat_formatted_1(client)
    await chat_large_prompt(client)


if __name__ == '__main__':
    asyncio.run(main())

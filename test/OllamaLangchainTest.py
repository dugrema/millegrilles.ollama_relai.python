import asyncio
import ollama
import pathlib

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_core.retrievers import RetrieverLike
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TextSplitter, CharacterTextSplitter, RecursiveCharacterTextSplitter

documents = [
  "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
  "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
  "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
  "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
  "Llamas are vegetarians and have very efficient digestive systems",
  "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
]


async def populate_vector_db(vector_store: Chroma):
    # await vector_store.aadd_texts(documents)

    doc_id_ctn = 0
    for doc in documents:
        doc_id = f"6f79b693-c2f8-4315-96a1-0435b579ad{doc_id_ctn}"
        await vector_store.aadd_documents([Document(doc, id=doc_id)])
        doc_id_ctn += 1

    pdf_path = pathlib.Path('/home/mathieu/Downloads/Articles')
    for pdf_file in pdf_path.iterdir():
        loader = PyPDFLoader(pdf_file, mode="single")
        document_list = await asyncio.to_thread(loader.load)
        document = document_list[0]
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        chunks = splitter.split_documents([document])
        chunk_no = 1
        for chunk in chunks:
            doc_id = f"6f79b693-c2f8-4315-96a1-0435b579ad{doc_id_ctn}/{chunk_no}"
            chunk_no += 1
            chunk.id = doc_id
        await vector_store.aadd_documents(list(chunks))
        doc_id_ctn += 1


async def connect_vector_db() -> RetrieverLike:
    embeddings = OllamaEmbeddings(
        model="all-minilm",
        # model="llama3.2:3b-instruct-q8_0",
        # num_gpu=0,  # Disable GPU to avoid swapping models on ollama
    )

    path_db = pathlib.Path("/tmp/chroma_langchain_db")
    must_populate = path_db.exists() is False

    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=str(path_db),  # Where to save data locally, remove if not necessary
    )

    if must_populate:  # Fill vector database
        await populate_vector_db(vector_store)

    # Use the vectorstore as a retriever
    retriever = vector_store.as_retriever()

    return retriever


async def ollama_generate(prompt: str):
    # generate a response combining the prompt and data we retrieved in step 2
    output = ollama.generate(
        model="llama3.2:3b-instruct-q8_0",
        # model="granite3.3:8b",
        # model="gemma3:4b-it-qat",
        # prompt=f"Using this data: {data}. Respond to this prompt: {input}"
        prompt=prompt,
        options={"temperature": 0.0}
    )

    return output


async def run_query(retriever: RetrieverLike):
    # query = "What animals are llamas related to?"
    # query = "How tall can a llama get?"
    # query = "What do llamas eat?"
    # query = "What is the size and weight of a llama?"
    # query = "How long can a llama live when it is well fed?"
    # query = "What is happening at the Canada Revenue Agency (CRA)?"
    # query = "Quel est le résultat de l'élection pour le Bloc Québécois?"
    query = "Why does Alberta claim they should get over 50% of the CPP?"

    prompt = await format_prompt(retriever, query)
    print("Prompt:\n%s" % prompt)
    response = await ollama_generate(prompt)

    print("Response:\n%s" % response['response'])



async def format_prompt(retriever: RetrieverLike, query: str) -> str:
    context_response = await retriever.ainvoke(query, k=6)

    context_tags = [f'<source id="{elem.id}">{elem.page_content}</source>' for elem in context_response]

    prompt = f"""
### Task:
Respond to the user query using the provided context, incorporating inline citations in the format [id] **only when the <source> tag includes an explicit id attribute** (e.g., <source id="1">).

### Guidelines:
- If you don't know the answer, clearly state that.
- If uncertain, ask the user for clarification.
- Respond in the same language as the user's query.
- If the context is unreadable or of poor quality, inform the user and provide the best possible answer.
- If the answer isn't present in the context but you possess the knowledge, explain this to the user and provide the answer using your own understanding.
- **Only include inline citations using [id] (e.g., [abcd-01], [efgh-22]) when the <source> tag includes an id attribute.**
- Do not cite if the <source> tag does not contain an id attribute.
- Do not use XML tags in your response.
- Ensure citations are concise and directly related to the information provided.

### Example of Citation:
If the user asks about a specific topic and the information is found in a source with a provided id attribute, the response should include the citation like in the following example:
* "According to the study, the proposed method increases efficiency by 20% [abcd-01]."

### Output:
Provide a clear and direct response to the user's query, including inline citations in the format [id] only when the <source> tag with id attribute is present in the context.

<context>
{"\n".join(context_tags)}
</context>

<user_query>
{query}
</user_query>    
    """

    return prompt


async def main():
    retriever = await connect_vector_db()
    await run_query(retriever)


if __name__ == '__main__':
    asyncio.run(main())

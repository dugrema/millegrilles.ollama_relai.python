# import asyncio
# import ollama
# import chromadb
# from chromadb.errors import InternalError
# from chromadb.types import Collection
#
# documents = [
#   "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
#   "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
#   "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
#   "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
#   "Llamas are vegetarians and have very efficient digestive systems",
#   "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
# ]
#
# async def create_docs(collection: Collection):
#     # store each document in a vector embedding database
#     for i, d in enumerate(documents):
#         response = ollama.embed(model="all-minilm", input=d)
#         embeddings = response["embeddings"]
#         collection.add(
#             ids=[str(i)],
#             embeddings=embeddings,
#             documents=[d]
#         )
#
#
# async def run_query(collection: Collection):
#     # an example input
#     # input = "What animals are llamas related to?"
#     # input = "How tall are the tallest llamas?"
#     # input = "What do llamas eat?"
#     # input = "What is the size and weight of a llama?"
#     input = "How long can a llama live when it is well fed?"
#
#     # generate an embedding for the input and retrieve the most relevant doc
#     response = ollama.embed(
#         model="all-minilm",
#         input=input
#     )
#     results = collection.query(
#         query_embeddings=[response["embeddings"][0]],
#         n_results=2
#     )
#     data = results['documents'][0]
#
#     prompt = f"""
# ### Task:
# Respond to the user query using the provided context, incorporating inline citations in the format [id] **only when the <source> tag includes an explicit id attribute** (e.g., <source id="1">).
#
# ### Guidelines:
# - If you don't know the answer, clearly state that.
# - If uncertain, ask the user for clarification.
# - Respond in the same language as the user's query.
# - If the context is unreadable or of poor quality, inform the user and provide the best possible answer.
# - If the answer isn't present in the context but you possess the knowledge, explain this to the user and provide the answer using your own understanding.
# - **Only include inline citations using [id] (e.g., [1], [2]) when the <source> tag includes an id attribute.**
# - Do not cite if the <source> tag does not contain an id attribute.
# - Do not use XML tags in your response.
# - Ensure citations are concise and directly related to the information provided.
#
# ### Example of Citation:
# If the user asks about a specific topic and the information is found in a source with a provided id attribute, the response should include the citation like in the following example:
# * "According to the study, the proposed method increases efficiency by 20% [1]."
#
# ### Output:
# Provide a clear and direct response to the user's query, including inline citations in the format [id] only when the <source> tag with id attribute is present in the context.
#
# <context>
# {data}
# </context>
#
# <user_query>
# {input}
# </user_query>
# """
#
#     # generate a response combining the prompt and data we retrieved in step 2
#     output = ollama.generate(
#         # model="llama3.2:3b-instruct-q8_0",
#         # model="granite3.3:8b",
#         model="gemma3:4b-it-qat",
#         # prompt=f"Using this data: {data}. Respond to this prompt: {input}"
#         prompt=prompt
#     )
#
#     print(output['response'])
#
#
# async def main():
#     client = chromadb.PersistentClient(path="/tmp/chromallm")
#     try:
#         collection = client.create_collection(name="docs")
#     except InternalError:  # DB Exists
#         collection = client.get_collection(name="docs")
#     else:
#         await create_docs(collection)
#
#     await run_query(collection)
#
#
# if __name__ == '__main__':
#     asyncio.run(main())

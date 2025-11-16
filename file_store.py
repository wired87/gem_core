
from google.genai import types

from gem_core import GAIC

def create_store():
    store = GAIC.file_search_stores.create()
    return store

def upload_file(store_name, file_path):
    upload_op = GAIC.file_search_stores.upload_to_file_search_store(
        file_search_store_name=store_name,
        file=file_path
    )
    return upload_op

def ask_rag(store_name):
    # Use the file search store as a tool in your generation call
    response = GAIC.models.generate_content(
        model='gemini-2.5-flash',
        contents='',
        config=types.GenerateContentConfig(
            tools=[types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[store.name]
                )
            )]
        )
    )
    print(response.text)

    # Support your response with links to the grounding sources.
    grounding = response.candidates[0].grounding_metadata
    sources = {c.retrieved_context.title for c in grounding.grounding_chunks}
    print('Sources:', *sources)
    return response.text



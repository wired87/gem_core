from google.genai import types

from gem_core import GAIC

store = GAIC.file_search_stores.create()

upload_op = GAIC.file_search_stores.upload_to_file_search_store(
    file_search_store_name=store.name,
    file='path/to/your/document.pdf'
)

while not upload_op.done:
  time.sleep(5)
  upload_ops = client.operations.get(upload_op)

# Use the file search store as a tool in your generation call
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='What does the research say about ...',
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
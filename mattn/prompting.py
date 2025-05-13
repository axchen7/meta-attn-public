from .document import Document
from .query import Query


def ask_for_query_from_document(document: Document) -> str:
    return f"""
Come up with a one-sentence question that asks about a fact in the following document.
The question should should be about a specific detail that is unrelated to the document's name or main topic.
This means that the document name and main topic SHOULD NOT appear in the question.

Document name: {document.name}

Document begins below:

{document.content}
    """.strip()


def create_query_document_input(query: Query, document: Document) -> str:
    input = f"""
Answer the following question based on the provided document.
Output the answer in a single sentence.

Question: {query.content}

Document begins below:

{document.content}

Answer:
    """.strip()

    input += "\n\n"  # otherwise, model may begin the output its own \n\n
    return input

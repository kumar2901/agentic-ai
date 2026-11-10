from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from pydantic import BaseModel

from paper_pilot.schemas import Keywords, PaperSummary, RagAnswer

prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            (
                "You are helpful research paper experts. "
                "Summarise papers in with the given style and length. "
                "Include mathematical details and simple analogies when applicable."
            ),
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        (
            "human",
            'Summarise the research paper titled "{paper_input}"  Style "{style_input}" Length "{length_input}".',
        ),
    ]
)


keyword_template = ChatPromptTemplate(
    [
        ("system", "You extract search keywords from research papers titles. "),
        ("human", 'Give 5 keywords for the research paper titled "{paper_input}".'),
    ]
)


def _structured(
    model: BaseChatModel,
    schema: type[BaseModel],
    fallback: BaseChatModel | None,
) -> Runnable[Any, Any]:
    """Return a runnable that uses the model to produce structured output based on the schema."""

    primary = model.with_structured_output(schema).with_retry(stop_after_attempt=3)

    if fallback is None:
        return primary

    backup = fallback.with_structured_output(schema).with_retry(stop_after_attempt=3)

    return primary.with_fallbacks([backup])


def build_summary_chain(
    model: BaseChatModel, fallback: BaseChatModel | None = None
) -> Runnable[dict[str, Any], Any]:
    """Build a chain to summarise research papers."""
    return prompt_template | _structured(model, PaperSummary, fallback)


def build_keyword_chain(
    model: BaseChatModel, fallback: BaseChatModel | None = None
) -> Runnable[dict[str, Any], Any]:
    """Build a chain to extract keywords from research papers."""
    return keyword_template | _structured(model, Keywords, fallback)


# parallel chain to summarise and extract keywords
def build_summary_and_keyword_chain(
    model: BaseChatModel,
    fallback: BaseChatModel | None = None,
) -> Runnable[dict[str, Any], dict[str, Any]]:
    """Build a chain to summarise research papers and extract keywords in parallel."""
    summary_chain = build_summary_chain(model, fallback)
    keyword_chain = build_keyword_chain(model, fallback)
    return RunnableParallel(summary=summary_chain, keywords=keyword_chain)


rag_prompt = ChatPromptTemplate(
    [
        (
            "system",
            (
                "You answer questions about research papers using ONLY the context provided. "
                "Each context block starts with a label like [Paper Title, p.7]. "
                "List every paper and page you used in `sources`. "
                'If the context does not contains the answer, reply "I don\'t know" with empty '
                "sources.  Never use outside knowledge."
            ),
        ),
        ("human", "Context:\n{context}\n\n Question: {question}"),
    ]
)


def format_docs(docs: list[Document]) -> str:
    """Join chunks into one text, each labeled with paper and page so the model can cite it"""
    return "\n\n".join(
        f"[{d.metadata['paper']},p.{d.metadata['page']}]\n{d.page_content}" for d in docs
    )


def build_rag_chain(
    model: BaseChatModel, retriever: BaseRetriever, fallback: BaseChatModel | None = None
) -> Runnable[str, dict[str, Any]]:
    """question-> {"question,"docs"(retrieved chunks),"answer"(RAGAnswer)}."""
    retrieve: Runnable[str, dict[str, Any]] = RunnableParallel(
        docs=retriever, question=RunnablePassthrough()
    )
    answer = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | rag_prompt
        | _structured(model, RagAnswer, fallback)
    )
    return retrieve | RunnablePassthrough.assign(answer=answer)

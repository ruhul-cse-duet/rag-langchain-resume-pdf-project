# rag/chain.py
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
)
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseLanguageModel


def build_rag_chain(
    llm: BaseLanguageModel,
    vectorstore: VectorStore,
    top_k: int = 4,
):
    """
    Build a simple RAG chain: (question) -> retrieve -> prompt -> LLM -> answer string.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    system_prompt = (
        "You are an assistant answering questions based only on the provided context.\n"
        "If the answer is not in the context, say that you don't know clearly.\n"
        "Be concise but clear. Answer in the same language as the question."
    )

    template = (
        "{system_prompt}\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )

    prompt = ChatPromptTemplate.from_template(template)

    # Map: input question -> {question, context} then -> prompt -> llm -> str
    rag_chain = (
        RunnableParallel(
            question=RunnablePassthrough(),
            context=retriever,
        )
        | prompt.partial(system_prompt=system_prompt)
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.core.config import settings
from app.services.vector_store import VectorStoreService


class RAGEngine:
    def __init__(self):
        # Vector store + retriever
        self.vector_service = VectorStoreService()
        self.retriever = self.vector_service.get_retriever(k=4)

        # Lightweight local LLM (TinyLlama)
        self.llm = ChatOllama(
            model=settings.LLM_MODEL_NAME,
            base_url=settings.LLM_BASE_URL,
            temperature=0.1,  # Low temperature = factual answers
        )

        # Strict policy-grounded prompt
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a Senior Loan Policy Officer.
            You MUST answer strictly using the provided policy context.
            If the answer is not present in the context, say:
            "I cannot find this information in the provided policy documents."

            <context>
            {context}
            </context>

            Question: {question}

            Answer:
            """
        )

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_answer(self, question: str) -> str:
        rag_chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain.invoke(question)

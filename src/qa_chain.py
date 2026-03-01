"""
QA Chain Module
Integrates retriever with LLM for question answering.
"""

from typing import List, Dict, Any
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
import logging

logger = logging.getLogger(__name__)


class FinancialQAChain:
    """QA chain specialized for financial document analysis."""

    FINANCIAL_PROMPT_TEMPLATE = """You are an expert financial analyst specializing in 10-K filings. 
Answer the question based ONLY on the provided context from the financial documents.

Context from 10-K filings:
{context}

Question: {question}

Instructions:
1. Base your answer strictly on the provided context
2. If the answer is not in the context, say "I don't have this information in the documents"
3. Include specific numbers, percentages, and facts when relevant
4. Cite which company's 10-K (AAPL/TSLA) and year the information comes from
5. Maintain professional financial analysis tone

Answer:"""

    def __init__(self, retriever, model_name: str = "gpt-3.5-turbo", temperature: float = 0.3):
        """
        Initialize the QA chain.

        Args:
            retriever: DocumentRetriever instance
            model_name: OpenAI model to use
            temperature: LLM temperature (lower = more deterministic)
        """
        self.retriever = retriever
        self.model_name = model_name
        self.temperature = temperature

        # Initialize LLM
        logger.info(f"Initializing LLM: {model_name}")
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=500,
        )

        # Create prompt template
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=self.FINANCIAL_PROMPT_TEMPLATE,
        )

        # Create RAG chain
        self._setup_chain()

    def _setup_chain(self):
        """Set up the retrieval QA chain."""
        try:
            # Use RetrievalQA for straightforward QA
            self.chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",  # Use "stuff" for shorter context, "map_reduce" for longer
                retriever=self.retriever.retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.prompt},
            )
            logger.info("QA chain initialized successfully")
        except Exception as e:
            logger.error(f"Error setting up QA chain: {str(e)}")
            raise

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question based on retrieved documents.

        Args:
            question: User question

        Returns:
            Dictionary with answer and source documents
        """
        try:
            logger.info(f"Processing question: {question}")

            result = self.chain({"query": question})

            # Format response
            response = {
                "question": question,
                "answer": result.get("result", "No answer generated"),
                "source_documents": result.get("source_documents", []),
            }

            logger.info("Question processed successfully")
            return response

        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "source_documents": [],
            }

    def answer_with_citations(self, question: str) -> str:
        """
        Answer a question and include source citations.

        Args:
            question: User question

        Returns:
            Formatted answer with citations
        """
        response = self.answer_question(question)

        answer = response["answer"]
        sources = response["source_documents"]

        # Format with citations
        formatted_answer = f"Q: {question}\n\nA: {answer}\n"

        if sources:
            formatted_answer += "\nSources:\n"
            for i, doc in enumerate(sources, 1):
                ticker = doc.metadata.get("ticker", "Unknown")
                year = doc.metadata.get("year", "Unknown")
                page = doc.metadata.get("page", "N/A")
                formatted_answer += f"{i}. {ticker} 10-K ({year}), Page {page}\n"

        return formatted_answer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("QA Chain module initialized")


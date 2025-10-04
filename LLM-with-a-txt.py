import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. SETUP: Create a more detailed dummy document ---
# This expanded text provides more information to query.
dummy_text = """
Fictional Tech Inc. - Q2 2025 Comprehensive Report

**Executive Summary:**
The second quarter of 2025 was a landmark period for Fictional Tech Inc., marked by record-breaking revenue and strategic advancements in our core technology sectors. Net revenue reached $15 million, a 20% increase from the previous quarter. This success was primarily driven by the launch of our new flagship product, the "InnovateX" smartphone. The device, which features our proprietary "Photon" camera sensor, accounted for 60% of total sales.

**Financial Performance:**
- Gross Profit: $8 million
- Operating Expenses: $4 million, with a 30% increase in R&D spending.
- Net Profit: $4 million.
- Stock Symbol: FTI, currently trading at $125 per share.

**Product & Division Highlights:**
- **InnovateX Smartphone:** Sold over 500,000 units in its first month. Customer feedback has been overwhelmingly positive, particularly regarding battery life and the Photon camera's low-light performance.
- **Software Suite:** Our "Synergy" office suite saw a 10% growth in subscriptions. New AI-driven analytics features are planned for integration in Q3. The head of the Synergy division, Dr. Evelyn Reed, will be retiring in August.
- **R&D Division:** Made significant breakthroughs in quantum computing, though commercial application is still several years away.

**Future Outlook & Strategy for Q3 2025:**
Our focus for the third quarter will be on expanding the InnovateX market into Europe and scaling up production. We also plan to announce a successor to Dr. Reed for the Synergy division by the end of July. We anticipate a slight increase in operating costs due to marketing campaigns in new regions.
"""
FILE_PATH = "comprehensive_report.txt"
with open(FILE_PATH, "w") as f:
    f.write(dummy_text)
print("--- Detailed dummy document created: comprehensive_report.txt ---")


def setup_knowledge_base(file_path: str, model_name: str = "llama3"):
    """
    Loads a document, splits it into chunks, creates embeddings,
    and stores them in a Chroma vector database.
    """
    print("--- Indexing document... ---")
    # Load the document
    loader = TextLoader(file_path)
    docs = loader.load()

    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # Use Ollama for embeddings
    embeddings = OllamaEmbeddings(model=model_name)

    # Create and return the vector store
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    print("--- Document indexed successfully. ---")
    return vectorstore


def create_rag_chain(vectorstore, model_name: str = "llama3"):
    """
    Creates the complete RAG chain for question-answering.
    """
    retriever = vectorstore.as_retriever()

    template = """
    You are an expert financial analyst. Answer the question based only on the following context.
    If you don't know the answer, just say that you don't know. Be concise and professional.

    Context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    llm = OllamaLLM(model=model_name)
    output_parser = StrOutputParser()

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )
    return rag_chain


def main():
    """
    Main function to set up the chatbot and start the interactive loop.
    """
    vectorstore = setup_knowledge_base(FILE_PATH)
    rag_chain = create_rag_chain(vectorstore)

    print("\n--- RAG Chatbot is ready! Ask questions about the Q2 report. Type 'exit' to quit. ---")

    try:
        while True:
            question = input("\nYour Question: ")
            if question.lower() == 'exit':
                print("Exiting chatbot.")
                break

            answer = rag_chain.invoke(question)
            print(f"\nAnswer: {answer}")

    except KeyboardInterrupt:
        print("\nExiting chatbot.")
    finally:
        # Clean up the created files
        vectorstore.delete_collection()
        os.remove(FILE_PATH)
        print("--- Cleanup complete. ---")


if __name__ == "__main__":
    main()
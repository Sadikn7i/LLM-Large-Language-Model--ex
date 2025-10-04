import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
# Import the WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def setup_knowledge_base_from_url(url: str, model_name: str = "llama3"):
    """
    Loads content from a URL, splits it, creates embeddings,
    and stores them in a Chroma vector database.
    """
    print(f"--- Loading and indexing content from {url} ---")
    # Use WebBaseLoader to load the content from the URL
    loader = WebBaseLoader(url)
    docs = loader.load()

    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # Use Ollama for embeddings
    embeddings = OllamaEmbeddings(model=model_name)

    # Create and return the vector store
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    print("--- Content indexed successfully. ---")
    return vectorstore

def create_rag_chain(vectorstore, model_name: str = "llama3"):
    """
    Creates the complete RAG chain for question-answering.
    """
    retriever = vectorstore.as_retriever()

    template = """
    You are an expert researcher. Answer the question based only on the following context.
    If you don't know the answer from the context, just say that you don't know. Be concise.

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
    # The URL we want to ask questions about
    WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    
    vectorstore = setup_knowledge_base_from_url(WIKIPEDIA_URL)
    rag_chain = create_rag_chain(vectorstore)

    print("\n--- RAG Chatbot is ready! Ask questions about the Wikipedia article. Type 'exit' to quit. ---")

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
        # Clean up the vector store
        vectorstore.delete_collection()
        print("--- Cleanup complete. ---")


if __name__ == "__main__":
    main()
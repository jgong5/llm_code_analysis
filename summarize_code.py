# Reference: https://python.langchain.com/docs/use_cases/code_understanding/
# Revised by Claude3 Opus

import argparse
import os
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import dotenv
dotenv.load_dotenv()

def main(args):
    if args.remote_repo_url:
        repo = Repo.clone_from(args.remote_repo_url, to_path=args.local_repo_path)
    else:
        repo = Repo(args.local_repo_path)

    if not args.load_from_existing_db:
        if os.path.exists(args.db_persist_path):
            print(f"Error: Local persistent database already exists at {args.db_persist_path}. Please use --load-from-existing-db or specify a different path.")
            return

        loader = GenericLoader.from_filesystem(
            args.local_repo_path,
            glob="**/*",
            suffixes=[".py"],
            exclude=["**/non-utf8-encoding.py"],
            parser=LanguageParser(language=Language.PYTHON, parser_threshold=args.parser_threshold),
        )
        documents = loader.load()

        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )
        texts = python_splitter.split_documents(documents)

        db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()), persist_directory=args.db_persist_path)
    else:
        db = Chroma(persist_directory=args.db_persist_path, embedding_function=OpenAIEmbeddings(disallowed_special=()))

    retriever = db.as_retriever(
        search_type="mmr",  # Also test "similarity"
        search_kwargs={"k": 8},
    )

    llm = ChatOpenAI(model_name=args.openai_model_name)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
            ),
        ]
    )

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, prompt)

    qa = create_retrieval_chain(retriever_chain, document_chain)
    result = qa.invoke({"input": args.user_question})
    print(result["answer"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a git repository using langchain and chatgpt.")
    parser.add_argument("local_repo_path", type=str, help="Path to the local repository.")
    parser.add_argument("--remote-repo-url", type=str, help="URL of the remote repository to clone (optional).")
    parser.add_argument("--openai-model-name", type=str, default="gpt-3.5-turbo", help="Name of the OpenAI ChatGPT model to use.")
    parser.add_argument("--user-question", type=str, default="Please summarize the project for me.", help="User input question to analyze the repository.")
    parser.add_argument("--load-from-existing-db", action="store_true", help="Load from an existing database instead of creating from raw text.")
    parser.add_argument("--db-persist-path", type=str, default="./db", help="Path to persist the Chroma database.")
    parser.add_argument("--parser-threshold", type=int, default=500, help="Parser threshold for the LanguageParser.")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Chunk size for the RecursiveCharacterTextSplitter.")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap for the RecursiveCharacterTextSplitter.")

    args = parser.parse_args()
    main(args)
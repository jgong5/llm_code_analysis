# Reference: https://python.langchain.com/docs/use_cases/code_understanding/
# Revised by Claude3 Opus according to the following prompts:
# I have a python program that uses langchain to create a vector database for a git repository and then uses chatgpt to query and analyze its content. 
# I'd like to make the following updates:
# 
# 1. Allow a mode that only providing an already existing local repo path.
# 2. Allow a mode that loads from an existing db without the need of creating from raw text (with Chroma.from_documents) on the fly.
# 3. When the chroma db is created on the fly, please allow the user to specify a path to persist it.
# 4. Please use command line option style: "--abc-def" instead of "--abc_def".
# 5. When not loading from an existing db while the local persistent db exists, issue error, do nothing and exit.
# 6. Allow configuration for parser_threshold, chunk_size, chunk_overlap. Leave the default to what currently provided.
# 7. Make the "local repo path" a positional arg.
# Here is the python program:

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
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

import dotenv
dotenv.load_dotenv()

def main(args):
    if args.service == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        embedding = GoogleGenerativeAIEmbeddings(model=args.gemini_embedding_model)
        llm = ChatGoogleGenerativeAI(model=args.gemini_model, convert_system_message_to_human=True)
    else:
        assert args.service == "openai"
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        embedding = OpenAIEmbeddings(model=args.openai_embedding_model)
        llm = ChatOpenAI(model_name=args.openai_model)

    if args.remote_repo_url:
        assert args.local_repo_path
        Repo.clone_from(args.remote_repo_url, to_path=args.local_repo_path)

    if args.local_repo_path:
        if os.path.exists(args.db):
            print(f"Error: Local persistent database already exists at {args.db}. Please use --load-from-existing-db or specify a different path.")
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

        db = Chroma.from_documents(texts, embedding, persist_directory=args.db)
    else:
        db = Chroma(persist_directory=args.db, embedding_function=embedding)

    retriever = db.as_retriever(
        search_type="mmr",  # Also test "similarity"
        search_kwargs={"k": 8},
    )

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
    result = qa.invoke({"input": args.prompt})
    print(result["answer"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a git repository using langchain and chatgpt.")
    parser.add_argument("--local-repo-path", type=str, help="Path to the local repository.")
    parser.add_argument("--remote-repo-url", type=str, help="URL of the remote repository to clone (optional).")
    parser.add_argument("--prompt", type=str, default="Please summarize the project for me.", help="User input question to analyze the repository.")
    parser.add_argument("--db", type=str, default="./db", help="Path to persist the Chroma database.")
    parser.add_argument("--parser-threshold", type=int, default=0, help="Parser threshold for the LanguageParser.")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Chunk size for the RecursiveCharacterTextSplitter.")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap for the RecursiveCharacterTextSplitter.")

    gemini = parser.add_argument_group("Gemini options")
    gemini.add_argument("--gemini-model", type=str, default="gemini-pro", help="Name of the chat model to use.")
    gemini.add_argument("--gemini-embedding-model", type=str, default="models/embedding-001", help="Name of the embedding model to use.")

    openai = parser.add_argument_group("OpenAI options")
    openai.add_argument("--openai-model", type=str, default="gpt-3.5-turbo", help="Name of the chat model to use.")
    openai.add_argument("--openai-embedding-model", type=str, default="text-embedding-3-small", help="Name of the embedding model to use.")

    parser.add_argument('--service', choices=['gemini', 'openai'], default='gemini', help="Service to use for embedding and chat.")

    args = parser.parse_args()
    main(args)
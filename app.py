import os
from typing import List

from langchain.document_loaders import (
    UnstructuredHTMLLoader,
    PyPDFLoader,
    CSVLoader,
    TextLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain.chat_models import ChatOpenAI

from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

import chainlit as cl






# os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
os.environ["OPENAI_API_KEY"] = ""


@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text/pdf/html/csv file to begin!",
            accept=["text/plain", "text/html", "application/pdf", "text/csv"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
    await msg.send()

    if file.type == "text/plain":
        loader = TextLoader(file.path)
    if file.type == "text/html":
        loader = UnstructuredHTMLLoader(file.path)
    if file.type == "application/pdf":
        loader = PyPDFLoader(file.path)
    # loader = PyMuPDFLoader(file.path)
    if file.type == "text/csv":
        loader = CSVLoader(file.path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    documents = loader.load_and_split(text_splitter)
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_documents)(
        documents,
        embeddings,
    )

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()


# Can you identify a key quote from the speech that encapsulates its main message?
# What is the source of the above quote?

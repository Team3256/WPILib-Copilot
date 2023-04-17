# WPILib-Copilot

This repo is an implementation of a locally hosted chatbot specifically focused on question answering over the [WPILib documentation](https://docs.wpilib.org/en/stable/).
Built with [LangChain](https://github.com/hwchase17/langchain/) and [FastAPI](https://fastapi.tiangolo.com/).

The app leverages LangChain's streaming support and async API to update the page in real time for multiple users.

## ✅ Running locally
1. Install dependencies: `pip install -r requirements.txt`
1. Run `ingest.sh` to ingest WPILib docs data into the vectorstore (only needs to be done once).
1. Run the app: `make start`
   1. To enable tracing, make sure `langchain-server` is running locally and pass `tracing=True` to `get_chain` in `main.py`. You can find more documentation [here](https://langchain.readthedocs.io/en/latest/tracing.html).
1. Open [localhost:9000](http://localhost:9000) in your browser.

## 📚 Technical description

There are two components: ingestion and question-answering.

Ingestion has the following steps:

1. Pull html from documentation site
2. Load html with LangChain's [ReadTheDocs Loader](https://langchain.readthedocs.io/en/latest/modules/document_loaders/examples/readthedocs_documentation.html)
3. Split documents with LangChain's [TextSplitter](https://langchain.readthedocs.io/en/latest/reference/modules/text_splitter.html)
4. Create a vectorstore of embeddings, using LangChain's [vectorstore wrapper](https://langchain.readthedocs.io/en/latest/reference/modules/vectorstore.html) (with OpenAI's embeddings and FAISS vectorstore).

Question-Answering has the following steps, all handled by [ConversationalRetrievalChain](https://python.langchain.com/en/latest/reference/modules/chains.html#langchain.chains.ConversationalRetrievalChain):

1. Given the chat history and new user input, determine what a standalone question would be (using ChatGPT).
2. Given that standalone question, look up relevant documents from the vectorstore.
3. Pass the standalone question and relevant documents to ChatGPT to generate a final answer.

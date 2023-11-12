import os

import pandas as pd
import streamlit as st
from streamlit_chat import message
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, AgentType

from utilities.agent_tools import PdfSearchTool, CsvToolSearch, run_agent
from utilities.prompts import CUSTOM_CHATBOT_PREFIX, CUSTOM_CHATBOT_SUFFIX,WELCOME_MESSAGE


def get_pdf_text(pdf_doc):
    text = ""
    metadata = {}
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    metadata['source'] = pdf_doc
    return text, metadata


def get_document_chunks(text, metadata):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250, chunk_overlap=50, length_function=len, separators=["\n\n", "\n"]
    )
    chunks = text_splitter.split_text(text)
    docs=[]
    for chunk in chunks:
        # Generate documents
        docs.append(Document(page_content=chunk, metadata=metadata))
    return docs


def prepare_pdf_chunks(pdf_docs):
    doc_chunks = []
    for document in pdf_docs:
        # get pdf text
        raw_text, metadata = get_pdf_text(document)
        # get the text chunks
        doc_chunks.append(get_document_chunks(raw_text, metadata))

    doc_chunks = [item for sublist in doc_chunks for item in sublist]
    return doc_chunks


def get_file_type(uploaded_file):
    ftypes = []
    file_type = uploaded_file.type
    if file_type == 'application/pdf':
        ftypes.append("pdf")
        # Do something with the PDF file
    elif file_type == 'text/csv':
        ftypes.append("csv")
        # Do something with the CSV file
    else:
        print("Uploaded file is of an unsupported type.")
    return ftypes


def delete_stored_sessions():
    if "stored_session" in st.session_state:
        del st.session_state["stored_session"]


# Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []

    # reset also the agent_chain
    reset_agent_chain(delete_files=True)

    st.session_state.memory.clear()


def reset_agent_chain(delete_files=False):
    if "agent_chain" in st.session_state:
        try:
            del st.session_state["agent_chain"]
        except NameError:
            pass

    if delete_files:
        if "pdf_files" in st.session_state:
            try:
                del st.session_state["pdf_files"]
            except NameError:
                pass

        if "csv_files" in st.session_state:
            try:
                del st.session_state["csv_files"]
            except NameError:
                pass


load_dotenv()

os.environ["OPENAI_API_BASE"] = os.environ["AZURE_OPENAI_ENDPOINT"]
os.environ["OPENAI_API_VERSION"] = os.environ["AZURE_OPENAI_API_VERSION"]
os.environ["OPENAI_API_TYPE"] = "azure"
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
try:
    os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]
    API_O = True
except KeyError:
    API_O = False

# Set Streamlit page configuration
st.set_page_config(page_title='Chat with your documents', layout='wide')

# Set up sidebar with model options
with st.sidebar.expander("Model settings", expanded=True):
    MODEL = st.selectbox(label='Model',
                         options=["MOSE-GPT4-8k", "MOSE-GPT4-32k"])

uploaded_file = st.sidebar.file_uploader(
    "Upload your pdf or csv documents here. Click on 'Add Data'", accept_multiple_files=True
)

if "file_names" not in st.session_state:
    st.session_state["file_names"] = []

if "pdf_files" not in st.session_state:
    st.session_state["pdf_files"] = []

if "csv_files" not in st.session_state:
    st.session_state["csv_files"] = {}
    st.session_state["csv_files"]["name"] = []
    st.session_state["csv_files"]["df"] = []


# Ask the user to enter their OpenAI API key
if not API_O:
    API_O = st.sidebar.text_input("API-KEY", type="password")
    os.environ["OPENAI_API_KEY"] = API_O

if uploaded_file:
    for file in uploaded_file:
        file_type = get_file_type(file)
        if "pdf" in file_type:
            if file not in st.session_state["pdf_files"]:
                st.session_state["pdf_files"].append(file)
                st.session_state["update_tools"] = 1
        elif "csv" in file_type:
            # adding directly csv files to the csv_files session state causes error while reading it into dataframe
            # This is a workaround:
            if file.name not in st.session_state["csv_files"]["name"]:
                df = pd.read_csv(file)
                st.session_state["csv_files"]["df"].append(df)
                st.session_state["csv_files"]["name"].append(file.name)
                st.session_state["update_tools"] = 1
        else:
            print("Unsupported file type.")

    if API_O:
        llm = AzureChatOpenAI(deployment_name=MODEL, temperature=0.0)
        tools = []

        if st.session_state["pdf_files"]:
            # get chunks from pdf docs
            doc_chunks = prepare_pdf_chunks(st.session_state["pdf_files"])
            doc_tool = PdfSearchTool(llm=llm, doc_chunks=doc_chunks, embedding_model=EMBEDDING_MODEL)
            tools.append(doc_tool)

        if st.session_state["csv_files"]["name"]:
            # TODO: what happens with multiple csv files?
            csv_file = st.session_state["csv_files"]["name"][0]
            csv_tool = CsvToolSearch(llm=llm, df=st.session_state["csv_files"]["df"][0])
            tools.append(csv_tool)

        # set up chat memory
        if 'memory' not in st.session_state:
            memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=10)
            st.session_state.memory = memory

    else:
        st.sidebar.warning('API key required to try this app.The API key is not stored in any form.')

    # Add a button to start a new chat
    st.sidebar.button("New Chat", on_click=new_chat, type='primary')


    def conversational_chat(query):
        if ("agent_chain" not in st.session_state) or st.session_state["update_tools"]:
            # create final agent with tools
            final_agent = initialize_agent(agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                                           tools=tools,
                                           llm=llm,
                                           memory=st.session_state.memory,
                                           return_source_documents=True,
                                           agent_kwargs={"system_message": CUSTOM_CHATBOT_PREFIX,
                                                         "human_message": CUSTOM_CHATBOT_SUFFIX},
                                           handle_parsing_errors=True
                                           )

            st.session_state["agent_chain"] = final_agent

        result = run_agent(query, st.session_state["agent_chain"])
        st.session_state['history'].append((query, result))

        return result


    # initialize the chat history and session states
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = [WELCOME_MESSAGE]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
    if "stored_session" not in st.session_state:
        st.session_state["stored_session"] = []

    # container for the chat history
    response_container = st.container()
    # container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Ask your question here", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversational_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            st.session_state["update_tools"] = 0

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

    # Display stored conversation sessions in the sidebar
    if st.session_state["stored_session"]:
        for i, sublist in enumerate(st.session_state["stored_session"]):
            with st.sidebar.expander(label=f"Conversation-Session:{i}"):
                st.write(sublist)

    # Allow the user to clear all stored conversation sessions
    st.sidebar.button("Clear Sessions", on_click=delete_stored_sessions, type='primary')


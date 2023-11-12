import os
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import BaseTool
from langchain.chat_models import AzureChatOpenAI
from typing import Dict, Union, List, Any
from langchain.agents import initialize_agent, AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.vectorstores import FAISS
from pandas import DataFrame
from utilities.prompts import PDFSEARCH_PROMPT_PREFIX, CUSTOM_CHATBOT_PREFIX, CUSTOM_CHATBOT_SUFFIX, \
    CSV_PROMPT_PREFIX, CSV_PROMPT_SUFFIX
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import Tool


class PdfSearchTool(BaseTool):
    """Tool to search pdf documents"""

    name = "@pdfsearch"
    description = "useful when the questions includes the term: @pdfsearch.\n"
    # description = "useful when the questions are related to pdf files. You can use this tool to analyze pdf files.\n"

    llm: AzureChatOpenAI
    k: int = 10
    embedding_model: str = "embedding"
    doc_chunks: List[Any] = []

    def _get_retriever_tool(self, save_local=False, load_local=False) -> Tool:
        embeddings = OpenAIEmbeddings(deployment=self.embedding_model)
        if load_local:
            vectors = FAISS.load_local(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors", "docsearch_vectors"))
        else:
            # vectors = FAISS.from_texts(texts=self.text_chunks, embedding=embeddings)
            vectors = FAISS.from_documents(documents=self.doc_chunks, embedding=embeddings)
        retriever = vectors.as_retriever(k=self.k)
        tool = create_retriever_tool(
            retriever,
            "search_given_document",
            "Searches and returns documents regarding the given pdf files."
        )
        if save_local:
            vectors.save_local(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors", "docsearch_vectors"))

        return tool

    def _run(self, tool_input: Union[str, Dict], ) -> str:
        try:
            retriever_tool = self._get_retriever_tool()

            tools = [retriever_tool]

            parsed_input = self._parse_input(tool_input)

            # agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
            agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION

            agent_executor = initialize_agent(tools=tools,
                                              llm=self.llm,
                                              agent=agent,
                                              agent_kwargs={'prefix': PDFSEARCH_PROMPT_PREFIX},
                                              callback_manager=self.callbacks,
                                              verbose=self.verbose,
                                              handle_parsing_errors=True)

            for i in range(2):
                try:
                    # response = run_agent(parsed_input, agent_executor)
                    response_all = agent_executor(parsed_input)
                    response = response_all["output"]
                    break
                except Exception as e:
                    response = str(e)
                    continue

            return response

        except Exception as e:
            print(e)


class CsvToolSearch(BaseTool):
    """Tool to search csv documents"""

    name = "@csvsearch"
    # description = "useful when the questions are related to csv files. You can use this tool to analyze csv files.\n"
    description = "useful when the questions includes the term: @csvsearch.\n"

    llm: AzureChatOpenAI
    df: DataFrame

    # data_path: Any

    def _run(self, tool_input: Union[str, Dict], ) -> str:
        try:

            # agent = create_csv_agent(self.llm, self.data_path, verbose=True)
            agent = create_pandas_dataframe_agent(llm=self.llm, df=self.df, verbose=True,
                                                  agent_type=AgentType.OPENAI_FUNCTIONS)

            for i in range(2):
                try:
                    response = agent.run(CSV_PROMPT_PREFIX + tool_input + CSV_PROMPT_SUFFIX)
                    break
                except Exception as e:
                    response = str(e)
                    continue

            return response

        except Exception as e:
            print(e)


def run_agent(question: str, final_agent: Any) -> str:
    """Function to run the brain agent and deal with potential parsing errors"""
    for _ in range(2):
        try:
            response = final_agent(question)["output"]
            break
        except Exception as e:
            # If the agent has a parsing error, we use OpenAI model again to reformat the error and give a good answer
            print("parsing error")
            chatgpt_chain = LLMChain(
                llm=final_agent.agent.llm_chain.llm,
                prompt=PromptTemplate(
                    input_variables=["error"],
                    template="Remove any json formating from the below text, also remove any portion "
                    'that says someting similar this "Could not parse LLM output: ". '
                    "Reformat your response in beautiful Markdown. Just give me the "
                    f"reformated text, nothing else.\n Text: {e}",
                ),
                verbose=False,
            )
            response = chatgpt_chain.run(str(e))
            continue
    return response


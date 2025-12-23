
# main.py
import os, glob, json, asyncio, time
from dotenv import load_dotenv
from pathlib import Path

# Azure SDKs
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    ToolSet,
    FunctionTool,
    FileSearchTool,
    ConnectedAgentTool,
    FilePurpose,
    ListSortOrder,  # optional, if you want to inspect run steps
)
from azure.ai.agents.models import AgentsNamedToolChoice, AgentsNamedToolChoiceType


# Semantic Kernel (Azure OpenAI)
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings

load_dotenv()
PROJECT_ENDPOINT = os.environ["PROJECT_ENDPOINT"]
MODEL = os.environ["MODEL_DEPLOYMENT_NAME"]
cred = DefaultAzureCredential()

project = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=cred)
with project:
    agents: AgentsClient = project.agents

    # --- Upload precedents and create vector store (RAG) ---
    # file_ids = []
    # for path in glob.glob("data/precedents/*.txt"):
    #     uploaded = agents.files.upload_and_poll(file_path=path, purpose=FilePurpose.AGENTS)
    #     file_ids.append(uploaded.id)
    # if not file_ids:
    #     raise RuntimeError("No precedents found under data/precedents/*.txt")

    # vector_store = agents.vector_stores.create_and_poll(file_ids=file_ids, name="precedents_vs")
    file_search = FileSearchTool(vector_store_ids=["vs_VZ5O8TfZCwimHUpC8i4DKdwB"])

    # --- Python functions for ingestion and clause extraction ---
    # def ingest_contract(payload: dict) -> dict:
    #     path = payload.get("contractPath", "data/contracts/contract1.txt")
    #     with open(path, "r", encoding="utf-8") as f:
    #         txt = f.read()
    #     return {"contractId": "C1", "text": txt}

    # def find_clauses(payload: dict) -> dict:
    #     txt = payload.get("text", "")
    #     types = {
    #         "Termination": ["terminate", "termination", "notice"],
    #         "Indemnity": ["indemnify", "indemnification", "losses"],
    #         "Confidentiality": ["confidential", "non-disclosure", "secret"],
    #         "Governing Law": ["governing law", "new york", "jurisdiction", "law"],
    #         "Payment Terms": ["pay", "invoice", "days"],
    #     }
    #     clauses = []
    #     for line in [l.strip() for l in txt.split("\n") if l.strip()]:
    #         label = "Other"
    #         low = line.lower()
    #         for k, kws in types.items():
    #             if any(w in low for w in kws):
    #                 label = k
    #                 break
    #         clauses.append({"type": label, "text": line})
    #     return {"clauses": clauses}

    # # ✅ One FunctionTool containing both functions; add once to one ToolSet
    # fn_toolset = ToolSet()
    # function_tools = FunctionTool([find_clauses,ingest_contract])
    # fn_toolset.add(function_tools)

    # # Enable auto function calls ONCE (covers both functions)
    # agents.enable_auto_function_calls(fn_toolset)

    # # --- Child agents ---
    # ingestion_agent = agents.create_agent(
    #     model=MODEL,
    #     name="IngestionAgent",
    #     instructions="Use 'ingest_contract' to read the contract and return {contractId, text}.",
    #     toolset=fn_toolset
    # )

    # clause_agent = agents.create_agent(
    #     model=MODEL,
    #     name="ClauseExtractionAgent",
    #     instructions="Use 'find_clauses' on provided text. Return [{type,text}].",
    #     toolset=fn_toolset
    # )

    # compliance_agent = agents.create_agent(
    #     model=MODEL,
    #     name="ComplianceValidatorAgent",
    #     instructions=(
    #         "Validate each clause against curated legal precedents. "
    #         "Use FileSearch to retrieve relevant snippets. "
    #         "Return for every clause: {verdict, rationale, citations:[{source_id, snippet}]}. "
    #         "Always include citations."
    #     ),
    #     tools=file_search.definitions,
    #     tool_resources=file_search.resources
    # )

    # # --- Connected Agent tools (type=connected_agent) ---
    # ing_tool = ConnectedAgentTool(
    #     id=ingestion_agent.id, name="IngestionAgent",
    #     description="Reads the contract and returns {contractId, text}."
    # )
    # clause_tool = ConnectedAgentTool(
    #     id=clause_agent.id, name="ClauseExtractionAgent",
    #     description="Extracts clauses [{type, text}] from the contract text."
    # )
    # compliance_tool = ConnectedAgentTool(
    #     id=compliance_agent.id, name="ComplianceValidatorAgent",
    #     description="Validates clauses and returns verdict + citations."
    # )

    # # --- Orchestrator ---
    # orchestrator = agents.create_agent(
    #     model=MODEL,
    #     name="LegalOrchestrator",
    #     instructions=(
    #         "Workflow:\n"
    #         "1) Call IngestionAgent to ingest the contract and return {contractId,text}.\n"
    #         "2) Call ClauseExtractionAgent to get {clauses}.\n"
    #         "3) For each clause, call ComplianceValidatorAgent to evaluate.\n"
    #         "4) Produce an EXECUTIVE SUMMARY (grounded) and a table of results with citations.\n"
    #         "Respond in one final assistant message."
    #     ),
    #     tools=[ing_tool.definitions[0], clause_tool.definitions[0], compliance_tool.definitions[0]]
    # )
    agent = project.agents.get_agent("asst_iOMdLSwr09nGY4hZHVeCMiHX")
    # --- Semantic Kernel plan (Azure OpenAI) ---
    kernel = Kernel()
    planner_service = AzureChatCompletion(
        deployment_name=os.environ["MODEL_DEPLOYMENT_NAME"],
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-10-01-preview",
        service_id="planner",
    )
    kernel.add_service(planner_service)

    plan_prompt = (
        "Goal: review a contract for compliance using three agents "
        "(Ingestion, ClauseExtraction, Compliance).\nReturn 3 numbered steps."
    )

    async def make_plan() -> str:
        chat_service = kernel.get_service(type=ChatCompletionClientBase)
        history = ChatHistory(system_message="You are a helpful planner.")
        history.add_user_message(plan_prompt)
        settings = PromptExecutionSettings()
        result = await chat_service.get_chat_message_content(chat_history=history, settings=settings)
        return result.content

    plan_text = asyncio.run(make_plan())

    # --- Run orchestration (auto process tools) ---
    thread = agents.threads.create()
    contractPath = "data/contracts/contract1.txt"
    if Path(contractPath).exists():
        print(f"contractPath:{contractPath}")
        agents.messages.create(
            thread_id=thread.id,
            role="user",
            content=json.dumps({"contractPath": contractPath})
        )
    else:
        print("ruh ruh")
        exit

    # ✅ Use create_and_process to auto-execute tool 
    
    tool_choice = AgentsNamedToolChoice(
        type=AgentsNamedToolChoiceType.CONNECTED_AGENT
    )

    run = agents.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
    print("Run status:", run.status)
    if run.status == "failed":
        print("Run failed:", run.last_error)

    # Optional: inspect tool calls for debugging
    for step in agents.run_steps.list(thread_id=thread.id, run_id=run.id, order=ListSortOrder.ASCENDING):
        print(step.step_details)

    # --- Read assistant's final text message (not .content) ---
    messages = agents.messages.list(thread_id=thread.id)
    executive_summary = None
    for m in messages:
        if m.role == "assistant" and m.text_messages:
            executive_summary = m.text_messages[-1].text.value
            break

    print("\n=== EXECUTIVE SUMMARY (Grounded) ===\n")
    print(executive_summary or "(No assistant summary message found. Check run_steps for errors.)")

    print("\n=== SIMPLE PLAN (SK) ===\n")
    print(plan_text)

#%%
%load_ext autoreload
%autoreload 2
#%%
inputs = {
  "messages": [
    {
      "content": "Draft a response to this email:\n\nFrom: Sam Sharaf <samuel.sharaf@salesforce.com>\nTo: Harrison Chase <harrison@langchain.dev>\nSubject: Re: Agent Protocol\n\nThank you Harrison for sharing the agent protocol docs and introduction to\r\nNuno Campos. Also thank you for spending time with us to brainstorm ideas\r\non the agent interoperability topic. We are impressed by the work you and\r\nteam have done to date.\r\nWe would love to hear more about Cisco's multi-agent use cases/scenarios\r\nwhich you and team have been working on.\r\n\r\nBest,\r\nSam\r\n\r\nOn Mon, Jan 27, 2025 at 8:50 AM Harrison Chase <harrison@langchain.dev>\r\nwrote:\r\n\r\n> Hi all -\r\n>\r\n> Thanks for the time last week. I wanted to follow up:\r\n>\r\n> - Agent Protocol Docs: https://github.com/langchain-ai/agent-protocol\r\n> - I wanted to introduce you to @Nuno Campos <nuno@langchain.dev> - he\r\n> leads the protocol work. As you start playing around with it, if you have\r\n> any feedback or questions - he's the guy!\r\n>\r\n> Would love to hear initial thoughts and impressions. Happy to set up time\r\n> later this week if you've had time to dive into it\r\n>\r\n> - Harrison\r\n>\r\n\r\n\r\n-- \r\nSam Sharaf | Sr. Director, Product Management - Frontier AI Innovation\r\nPhone: 925-964-7049 | salesforce <http://salesforce.com/> |\r\nsamuel.sharaf@salesforce.com <mhammel@salesforce.com>\r\n*Certified Distinguished Enterprise Architect*\r\n",
      "additional_kwargs": {},
      "response_metadata": {},
      "type": "human",
      "id": "af9902c0-5d54-474a-a609-42f215bee380",
      "example": False
    },
    {
      "content": "Great to hear you're interested in Cisco's multi-agent use cases! I've looped in Jayesh to provide more insights. \n\n- Harrison",
      "additional_kwargs": {},
      "response_metadata": {},
      "type": "ai",
      "id": "3844e5c9-ac44-4b33-b92e-6d3f6b5c1139",
      "example": False,
      "tool_calls": [],
      "invalid_tool_calls": []
    },
    {
      "role": "user",
      "content": "A better response would have been: I introduced Jayesh to them in a separate email!\n\n- Harrison"
    }
  ],
  "existing": [
    [
      "eacdf504-e1e8-4998-b280-1bef00e42f94",
      "Memory",
      {
        "content": "As of January 2025, ongoing collaboration with Salesforce team (including Sam Sharaf, Sr. Director of Product Management - Frontier AI Innovation) regarding agent interoperability and agent protocol. Key details:\n- Shared agent protocol documentation\n- Introduced them to Jayesh for agent protocol work\n- Discussing Cisco's multi-agent use cases\n- Sam's contact: samuel.sharaf@salesforce.com, Phone: 925-964-7049"
      }
    ],
    [
      "415c9b3d-b7be-4923-9e73-dd472ac91d84",
      "Memory",
      {
        "content": "Response preferences update based on feedback: When coordinating meetings, while maintaining enthusiasm, keep responses exceptionally concise. A good pattern is: 1) Express excitement, 2) Acknowledge received information/availability, 3) Provide relevant context (e.g., who will attend), 4) Sign off. Avoid repeating availability times in response since they're in the email thread."
      }
    ],
    [
      "fd33e04a-4d0e-4fb3-b53e-32cd598320bb",
      "Memory",
      {
        "content": "Response to meeting requests from important clients should include: 1) Clear confirmation of availability, 2) Expression of enthusiasm/excitement, 3) Proactive suggestion for next steps (e.g., calendar invites), 4) Professional but warm tone."
      }
    ],
    [
      "15c5ee85-c5b0-478e-b3da-fca2844c371f",
      "Memory",
      {
        "content": "Important technical discussion with MongoDB senior leadership initiated after SF meeting with Sahir Azam (CPO):\n- Topic: Technical deployment challenges, Kubernetes infrastructure\n- Contact: Andrew Davidson (SVP Products, MongoDB, 12+ years experience, +1-650-269-7773)\n- Key Questions: \n  * Technical deployment challenges in customer environments\n  * Kubernetes vs non-Kubernetes deployments\n  * MongoDB Enterprise deployment patterns\n  * Control plane changes and upgrade handling\n  * Future deployment model recommendations\n- Mukil (LangChain Head of Infrastructure) to lead discussion\n- Coordinating through EA Viv Chacon (viv.chacon@mongodb.com, 516-543-2548)"
      }
    ],
    [
      "791dbf4d-80d8-422a-9994-9edd33fcc369",
      "Memory",
      {
        "content": "When responding to meeting coordination emails: 1) Keep responses extremely short and direct, 2) Only share availability times if explicitly requested, 3) Avoid repeating details already in thread. This pattern was reinforced through feedback - a good response simply clarifies attendees without volunteering additional information."
      }
    ],
    [
      "09fa79cb-fe6a-4a9f-964c-97b3abe2dde6",
      "Memory",
      {
        "content": "Contact information for commercial real estate agent Jay Shaffer: Managing Partner at Colton Commercial & Partners, contact info - Direct: (415) 590-7871, Cell: (415) 407-3997, Email: jshaffer@coltoncommercialsf.com. Has prior experience working with Suzie Ng for 555 19th street location and is partners with Shabber (Riffat's husband) who handled their Menlo office."
      }
    ],
    [
      "6763d5b8-2131-4c21-8855-8979a0637d68",
      "Memory",
      {
        "content": "When corresponding with high-value potential customers like JPMorgan Chase (JPMC): 1) Maintain an enthusiastic and warm tone in communications, explicitly expressing excitement about opportunities to meet, 2) Be proactive about meeting logistics, including suggesting calendar placeholders and confirmations, 3) Ensure communications demonstrate both eagerness to collaborate and professional competence."
      }
    ],
    [
      "bee69702-d4c6-466b-a35c-87247e332b84",
      "Memory",
      {
        "content": "Key contacts at Mercor as of Jan 2025: Adarsh Hiremath (adarsh@mercor.com) and Aksh Garg (aksh@mercor.com). They are authorized to work with LangSmith's tools and have special pricing arrangements."
      }
    ],
    [
      "408616ee-98ac-4772-aa8a-2bde4f8b2830",
      "Memory",
      {
        "content": "A meeting with JPMorgan Chase's CDAO leadership, including Teresa, has been confirmed for March 3rd, 2024 at 2:15 PM at their Mission Street office. JPMC is a significant potential customer requiring enthusiastic engagement. Key contact: Joe Kane (Managing Director - Global Technology Strategy, Innovation & Partnerships), contact: joseph.f.kane@jpmorgan.com, T: 212-622-9651, M: 917-841-3888. His EA is Liz Persaud (elizabeth.x.persaud@jpmchase.com)."
      }
    ],
    [
      "30596b36-a43c-4c84-8a69-13d7f16fdcde",
      "Memory",
      {
        "content": "Maintain an enthusiastic and warm tone in professional communications, especially with important clients like JPMorgan Chase. This includes expressing excitement about meetings and proactively suggesting next steps like calendar placeholders."
      }
    ]
  ]
}
#%%
from langmem import create_memory_enricher

enricher = create_memory_enricher(
    "gpt-4o",
)
results = await enricher(inputs["messages"], existing=inputs["existing"])

#%%

#%%
results[0]
#%%
consolidator = create_memory_enricher(
    "gpt-4o",
    instructions="Your task is to deduplicate, conslidate,"
    " and enrich the provided memories for future use. Try to compress without loss of information."
    " Extract new memories if you see patterns to generalize.",
    enable_deletes=True,
)
consolidated = await consolidator([], existing=inputs["existing"] + results)
#%%
import json
print(json.dumps([m.model_dump_json() for _, m in consolidated], indent=2)) #print(json.dumps(results, indent=2))
#%%
print(json.dumps(inputs, indent=2))
#%%
# # %% [markdown]
# # # LangMem Examples
# # This notebook demonstrates the key functionality of LangMem, a library for managing conversational memory.

# # %% [markdown]
# # ## Setup
# # First, let's set up our environment and import dependencies
# # %%
# %reload_ext autoreload
# %autoreload 2
# #rom pydantic import BaseModel


# from langchain.chat_models import init_chat_model
# from langchain_openai import OpenAIEmbeddings
# from langgraph.store.memory import InMemoryStore

# # Create a store with embeddings for semantic search
# embed = OpenAIEmbeddings(model="text-embedding-3-small")
# store = InMemoryStore(index={"embed": embed})

# # Sample conversation we'll use throughout the examples
# conversation = [
#     {
#         "role": "system",
#         "content": "You are a helpful customer support agent at LangChain."
#     },
#     {
#         "role": "user",
#         "content": "Could you please confirm if the LangGraph Platform Cloud is still in beta? This is unclear from the information provided on your website. I subscribed to the Plus plan under the assumption that it is still in beta, as in the platform."
#     },
#     {
#         "role": "assistant",
#         "content": "Hello Teodor, Thank you for contacting LangChain Support. We have received your request regarding the LangGraph Platform Cloud beta status and billing concerns. Our team is reviewing your inquiry and will provide clarification on the current status of the platform, your subscription, and any applicable adjustments needed."
#     },
#     {
#         "role": "assistant",
#         "content": "Hi Teodor, Thank you for writing in to LangChain Support. I apologize for the confusion here. The pending charge was for the LangSmith Plus plan which is $39.00 per seat. The page you're referencing is for our LangGraph product which is free while in beta, but may have a separate charge in the future. I understand the confusion since the naming conventions for the plans are similar, I'll follow up internally to see if we can make an adjustment/clarification to make it more clear. I issued a credit for the LangSmith seat to cover the $39.00 charge that was pending so you should see a $0 invoice now. Please log into LangSmith and ensure you're switched to the free Developer plan instead of the Plus plan to avoid any future seat charges until your needs change and Plus better suits you. If you have any additional questions please let me know. Best, Chad"
#     },
#     {
#         "role": "user",
#         "content": "Hello Chad, Thank you very much for the credit, but it looks like I still got charged $18.08. I just received the receipt."
#     },
#     {
#         "role": "assistant",
#         "content": "Hi Teodor, Sorry about that - missed the prorated seat charge for this month. I issued a refund for the $18.08. Best, Chad"
#     },
#     {
#         "role": "user",
#         "content": "Hello again Chad, Thank you for the refund but it looks like I just got charged again for $13.00."
#     },
#     {
#         "role": "assistant",
#         "content": "Hi Teodor, Thanks for letting me know - sorry about that. I had mentioned previously to please log into LangSmith and ensure you're switched to the free Developer plan instead of the Plus plan to avoid any future seat charges until your needs change and Plus better suits you. I didn't catch that wasn't completed, so the system is trying to charge a prorated seat charge for the rest of the month. Can you please make that change? Once the plan is switched I'll refund that charge. Best, Chad"
#     },
#     {
#         "role": "user",
#         "content": "Thank you for getting back to me. I've just switched to the free plan."
#     },
#     {
#         "role": "assistant",
#         "content": "Hi Teodor, Thank you - I sent the $13.00 refund. I don't see any additional pending charges but please let me know if anything else comes up. Sorry for the back and forth, and I appreciate your patience here. Best, Chad"
#     }
# ]

# # %% [markdown]
# # ## 1. Conversation Summarization
# # The `create_thread_extractor` creates a function that can summarize conversations or other interactions.
# # You can customize the output schema and instructions to control the summary format.

# # %%
# from langmem import create_thread_extractor
# from pydantic import BaseModel, Field

# # Default usage
# summarizer = create_thread_extractor("gpt-4o-mini")
# result = await summarizer(conversation)
# print("Default Summary:", result)

# # Custom schema for structured summaries
# class DetailedSummary(BaseModel):
#     """A detailed summary of a customer support conversation."""
#     title: str = Field(description="A brief title capturing the main topic")
#     customer_issue: str = Field(description="The main issue or request from the customer")
#     agent_actions: list[str] = Field(description="Key actions taken by the support agent")
#     resolution: str = Field(description="How the issue was resolved")
#     follow_up_needed: bool = Field(description="Whether further follow-up is required")

# # Custom instructions for support-specific summaries
# support_instructions = """Analyze this customer support conversation and extract:
# 1. The core customer issue
# 2. Actions taken by the support agent
# 3. Whether the issue was fully resolved
# 4. Any needed follow-up actions"""

# detailed_summarizer = create_thread_extractor(
#     "gpt-4o-mini",
#     schema=DetailedSummary,
#     instructions=support_instructions
# )
# detailed_result = await detailed_summarizer(conversation)
# print("\nDetailed Summary:", detailed_result)

# # %% [markdown]
# # ## 2. Memory Extraction and Updates
# # The `create_memory_enricher` can be customized with schemas to extract specific types of information
# # and instructions to guide the extraction process.

# # %%
# from langmem import create_memory_enricher

# # Default string-based memories
# basic_enricher = create_memory_enricher("gpt-4o-mini")
# basic_memories = await basic_enricher(conversation[:4])
# print("Basic Memories:", basic_memories)

# # Custom schema for structured memories
# class CustomerPreference(BaseModel):
#     """Customer preferences and important details."""
#     topic: str = Field(description="The subject of the preference")
#     preference: str = Field(description="The actual preference or requirement")
#     confidence: float = Field(description="Confidence in this preference (0-1)")
#     source: str = Field(description="Where this preference was inferred from")

# class BillingEvent(BaseModel):
#     """Billing-related events and issues."""
#     amount: float = Field(description="The amount involved")
#     action: str = Field(description="The billing action (charge/refund/credit)")
#     reason: str = Field(description="Why this billing event occurred")
#     status: str = Field(description="Current status of the billing event")

# # Custom instructions for support-focused memory extraction
# support_memory_instructions = """Extract and maintain:
# 1. Customer preferences and requirements
# 2. Billing events and their resolutions
# 3. Update existing memories if new information contradicts or clarifies them"""

# structured_enricher = create_memory_enricher(
#     "gpt-4o-mini",
#     schemas=[CustomerPreference, BillingEvent],
#     instructions=support_memory_instructions,
#     enable_inserts=True  # Allow creating new memories
# )

# structured_memories = await structured_enricher(conversation)
# print("\nStructured Memories:", structured_memories)

# # %% [markdown]
# # ## 3. Memory Management Tools
# # The memory tools can be configured with custom instructions and namespace prefixes
# # to control how and where memories are stored.

# # %%
# from langgraph.prebuilt import create_react_agent
# from langmem import create_manage_memory_tool, create_search_memory_tool

# # Custom instructions for memory management
# manage_instructions = """Manage memories with these guidelines:
# 1. Store clear, actionable preferences
# 2. Update contradictory information
# 3. Remove outdated or superseded memories
# 4. Tag memories by category (billing, product, support)"""

# # Custom instructions for memory search
# search_instructions = """Search memories considering:
# 1. Semantic similarity to the query
# 2. Recency of the memory
# 3. Confidence in the memory
# 4. Relevance to the current context"""

# # Create tools with custom configuration
# manager = create_manage_memory_tool(
#     instructions=manage_instructions,
#     namespace_prefix=("support_memories", "{langgraph_user_id}", "{category}")
# )

# searcher = create_search_memory_tool(
#     instructions=search_instructions,
#     namespace_prefix=("support_memories", "{langgraph_user_id}", "{category}")
# )

# # Create agent with configured tools
# llm = init_chat_model("gpt-4o-mini")
# agent = create_react_agent(llm, tools=[manager, searcher], store=store)

# # Example with configured namespace
# result = await agent.ainvoke({
#     "messages": [{"role": "user", "content": "I prefer email support over chat."}],
#     "configurable": {
#         "langgraph_user_id": "user123",
#         "category": "preferences"
#     }
# })

# # %% [markdown]
# # ## 4. Full Memory Pipeline
# # The memory pipeline can be configured with custom schemas and instructions
# # to create a complete memory management system.

# # %%
# from langmem import create_memory_store_enricher
# from langgraph.func import entrypoint

# # Define comprehensive memory schemas
# class SupportMemory(BaseModel):
#     """Base class for support-related memories."""
#     category: str = Field(description="Type of memory (preference/billing/product)")
#     confidence: float = Field(description="Confidence in this memory (0-1)")
#     last_updated: str = Field(description="When this memory was last updated")
#     content: dict = Field(description="The actual memory content")

# # Create pipeline with custom configuration
# processor = create_memory_store_enricher(
#     "gpt-4o-mini",
#     schemas=[SupportMemory],
#     instructions="""Extract and maintain support-related memories:
# 1. Customer preferences and requirements
# 2. Billing history and issues
# 3. Product usage patterns
# 4. Update existing memories with new information
# 5. Maintain confidence scores for each memory""",
#     enable_inserts=True,
#     namespace_prefix=("support_memories", "{langgraph_user_id}", "{category}")
# )

# @entrypoint(store=store)
# async def support_memory_pipeline(state: dict):
#     return await processor(state["messages"])

# # Process with custom configuration
# await support_memory_pipeline.ainvoke({
#     "messages": conversation,
#     "configurable": {
#         "langgraph_user_id": "user123",
#         "category": "support"
#     }
# })

# # %% [markdown]
# # ## 5. Prompt Optimization
# # The prompt optimizer can be configured to focus on specific aspects
# # of prompt improvement.

# # %%


# # Original prompt
# support_prompt = """You are a helpful customer support agent at LangChain."""

# feedback = """Areas for improvement:
# 1. Tone consistency
# 2. Proactive problem identification
# 3. Clear next steps
# 4. Billing expertise"""

# # Get optimized prompt with feedback
# improved_prompt = await optimizer(
#     conversation,
#     current_prompt=support_prompt,
#     feedback=feedback
# )
# print("Improved Prompt:", improved_prompt)

# # %%

# import langsmith as ls
# from typing import TypedDict, Literal


# model = init_chat_model(model="gpt-4o")


# prompt1 = """Classify the following email.

# Subject: {subject}
# Body: {body}"""


# class Output1(TypedDict):
#     classification: Literal['ignore', 'respond', 'notify']


# prompt2 = """Classify the following email.

# Subject: {subject}
# Body: {body}"""

# class Output2(TypedDict):
#     classification: Literal['ben', 'jerry', 'tom', 'sarah', 'will', 'ankush']


# class Classifier:
#     def __init__(self, prompts: dict):
#         self.prompts = prompts
#     async def __call__(self, inputs):
#         input1 = self.prompts["Triager"].format(**inputs)
#         response = await model.with_structured_output(Output1).ainvoke(input1)
#         if response['classification'] == "respond":
#             input2 = self.prompts["Router"].format(**inputs)
#             response = await model.with_structured_output(Output2).ainvoke(input2)
#             return response['classification']
#         else:
#             return response['classification']

# prompts = [
#     {
#         "name": "Triager",
#         "prompt": prompt1,
#         "when_to_update": "If the classification was supposed to be ignore/notify but instead we a person to tag for response."
#     },
#     {
#         "name": "Router",
#         "prompt": prompt2,
#         "when_to_update": "If, the classification is supposed to be route_to a particular person and that person is wrong or it was flagged as ignore or notify."
#         " This prompt is run AFTER the triager prompt, only if the triager prompt returns a \"respond\"."
#     }
# ]

# def correctness_evaluator(outputs, reference_outputs):
#     if outputs["output"] == reference_outputs["route_to"]:
#         return 1
#     else:
#         return 0

# classifier = Classifier({p["name"]: p["prompt"] for p in prompts})
# results = await ls.aevaluate(
#     classifier.__call__,
#     data="email_cs_multistep",
#     evaluators=[correctness_evaluator]
# )
# all_results = [r async for r in results]


# # %%

# from langchain_core.load import loa/
#     return (load(r['run'].child_runs[0].child_runs[0].inputs['messages'][0]), f"Expected: {r['example'].outputs}")

# convos = [_get_convo(r) for r in all_results]

# # %%
# from langmem import create_multi_prompt_optimizer
# optimizer = create_multi_prompt_optimizer("claude-3-5-sonnet-latest")

# results = await optimizer(convos, prompts = prompts)

# # %%
# new_classifier = Classifier({p["name"]: p["prompt"] for p in results})
# new_results = await ls.aevaluate(
#     new_classifier.__call__,
#     data="email_cs_multistep",
#     evaluators=[correctness_evaluator]
# )
# all_new_results = [r async for r in new_results]
# new_results
# # %%

# from langmem import create_multi_prompt_optimizer, Prompt


# prompts = [{
#     "name": "tone",
#     "prompt": "respond to user in a friendly and polite tone",
#     "update_instructions": "only update to include more information about the tone of the response",
#     "when_to_update": "If the feedback would contain info as to how the tone of the response should be"
# }]

# from langchain_core.messages import HumanMessage, AIMessage

# messages1 = [HumanMessage(content="hi!"), AIMessage(content="f u man")]

# optimizer = create_multi_prompt_optimizer("claude-3-5-sonnet-latest", kind="metaprompt", config={"max_reflection_steps": 1})
# result = await optimizer([(messages1, {"tone": "should be nice"})], prompts)


# print("Original Prompt:")
# print(prompts[0]["prompt"])
# print("Optimized Prompt:")
# print(result[0]["prompt"])
# # %%

# %%

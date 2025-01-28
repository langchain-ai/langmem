import asyncio
import typing
import uuid

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode
from langgraph.utils.config import get_store
from pydantic import BaseModel, Field, model_validator
from trustcall import create_extractor
from langmem import utils
from langmem.prompts.optimization import (
    create_prompt_optimizer,
    create_multi_prompt_optimizer,
    Prompt,
)
import langsmith as ls

## LangGraph Tools


def create_manage_memory_tool(
    instructions: str = """Proactively call this tool when you:
1. Identify a new USER preference.
2. Receive an explicit USER request to remember something or otherwise alter your behavior.
3. Are working and want to record important context.
4. Identify that an existing MEMORY is incorrect or outdated.""",
    namespace_prefix: tuple[str, ...] | utils.NamespaceTemplate = (
        "memories",
        "{user_id}",
    ),
):
    namespacer = (
        utils.NamespaceTemplate(namespace_prefix)
        if isinstance(namespace_prefix, tuple)
        else namespace_prefix
    )

    @tool
    async def manage_memory(
        action: typing.Literal["create", "update", "delete"],
        content: typing.Optional[str] = None,
        *,
        id: typing.Optional[uuid.UUID] = None,
    ):
        """Create, update, or delete persistent MEMORIES that will be carried over to future conversations.
        {instructions}"""
        store = get_store()

        if action == "create" and id is not None:
            raise ValueError(
                "You cannot provide a MEMORY ID when creating a MEMORY. Please try again, omitting the id argument."
            )

        if action in ("delete", "update") and not id:
            raise ValueError(
                "You must provide a MEMORY ID when deleting or updating a MEMORY."
            )
        if action == "delete":
            await store.adelete(namespace_prefix, key=str(id))
            return f"Deleted memory {id}"
        namespace = namespacer()
        id = id or uuid.uuid4()
        await store.aput(
            namespace,
            key=str(id),
            value={"content": content},
        )
        return f"{action}d memory {id}"

    manage_memory.__doc__.format(instructions=instructions)

    return manage_memory


_MEMORY_SEARCH_INSTRUCTIONS = ""


def create_search_memory_tool(
    instructions: str = _MEMORY_SEARCH_INSTRUCTIONS,
    namespace_prefix: tuple[str, ...] = ("memories", "{user_id}"),
):
    namespacer = utils.NamespaceTemplate(namespace_prefix)

    @tool(response_format="content_and_artifact")
    async def search_memory(
        query: str,
        *,
        limit: int = 10,
        offset: int = 0,
        filter: typing.Optional[dict] = None,
    ):
        """Search your long-term memories for information relevant to your current context. {instructions}"""
        store = get_store()
        namespace = namespacer()
        memories = await store.asearch(
            namespace,
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
        )
        return [m.dict() for m in memories], memories

    search_memory.__doc__.format(instructions=instructions)  # type: ignore

    return search_memory

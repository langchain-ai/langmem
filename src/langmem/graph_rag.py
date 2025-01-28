# import uuid
# from datetime import datetime
# from langgraph.utils.config import get_store, get_config
# from langchain_core.language_models.chat_models import BaseChatModel


# # Assume we have:
# #   store: BaseStore
# #     store.aput(namespace: tuple[str, ...], key: str, value: dict)
# #     store.search(namespace: tuple[str, ...], query: str | list[float], filter: dict | None, limit: int | None)
# #
# #   llm: we can call llm.invoke(messages: list[dict]) -> dict


# class sessionDoc(dict):
#     """An 'session' doc stored in the system."""

#     pass


# class EntityDoc(dict):
#     """An 'entity' doc in the system."""

#     pass


# class EdgeDoc(dict):
#     """A relationship doc linking two entity keys."""

#     pass


# async def ingest_session(
#     llm: str | BaseChatModel,
#     messages: list,
#     reference_time: datetime,
#     session_id: str | None = None,
#     namespace: tuple[str, ...] = ("graph_rag",),
# ) -> dict:
#     """
#     Demonstration of a single-session ingestion:
#       1. Create or retrieve session doc
#       2. LLM extraction of entities
#       3. Node embedding + search for near-duplicates
#       4. Merge/dedupe nodes
#       5. Edge extraction + resolution
#       6. Store everything
#     Returns a summary of the new/updated nodes & edges for logging or further use.
#     """
#     N_RECENT = 10
#     store = get_store()
#     config = get_config()
#     if session_id is None:
#         session_id = config["configurable"]["thread_id"]

#     recent_sessions = await store.asearch(
#         (*namespace, "sessions"),
#         query=None,
#         filter={"type": {"$eq": "session"}},
#         limit=N_RECENT,
#     )

#     prompt = [
#         {"role": "system", "content": "You are an entity extraction assistant."},
#         {"role": "user", "content": f"Text:\n{session_text}\nExtract named entities."},
#         {"role": "user", "content": f"Similar sessions:\n{recent_sessions}"},
#     ]
#     llm_response = await llm.ainvoke(prompt)
#     new_entities = llm_response.get("entities", [])
#     newly_created_nodes = []
#     resolved_nodes = []
#     for entity_info in new_entities:
#         entity_text = entity_info["name"]
#         candidate_entities = await store.search(
#             namespace,
#             query=embedding_vector,
#             filter={"type": {"$eq": "entity"}, "group_id": {"$eq": group_id}},
#             limit=5,
#         )

#         # 3c. Decide if we merge or not. For a simple example, if the top candidate is above some threshold, we treat it as same entity
#         # We'll skip threshold logic for brevity, but in real code you'd compare embeddings or names
#         if candidate_entities:
#             top_candidate = candidate_entities[0]
#             # Suppose we do a naive check of name equality
#             if top_candidate["name"].lower() == entity_text.lower():
#                 # Same entity -> reuse key
#                 resolved_nodes.append(top_candidate)
#                 continue

#         # If not merged, create new entity doc
#         new_key = str(uuid.uuid4())
#         entity_doc = {
#             "type": "entity",
#             "group_id": group_id,
#             "name": entity_text,
#             "entity_type": entity_info.get("entity_type", "unknown"),
#             "description": entity_info.get("description", ""),
#             "created_at": str(datetime.utcnow()),
#             "embedding": embedding_vector,
#         }
#         await store.aput(namespace, new_key, entity_doc)
#         newly_created_nodes.append({**entity_doc, "key": new_key})
#         resolved_nodes.append({**entity_doc, "key": new_key})

#     edge_prompt = [
#         {"role": "system", "content": "You are a relationship extraction assistant."},
#         {
#             "role": "user",
#             "content": f"session text:\n{session_text}\nExtract relationships among these entities: {resolved_nodes}",
#         },
#     ]
#     edge_response = await llm.invoke(edge_prompt)
#     # Suppose it returns e.g. [{"source": "Alice", "target": "Bob", "type": "friends", "description":"..."}]
#     new_edges = edge_response.get("relationships", [])

#     # 5. For each edge, embed, deduplicate, store
#     newly_created_edges = []
#     for edge_info in new_edges:
#         # 5a. find the source/target keys
#         source_key = None
#         target_key = None
#         # naive approach: look up by matching name
#         for node_doc in resolved_nodes:
#             if node_doc["name"].lower() == edge_info["source"].lower():
#                 source_key = node_doc["key"]
#             if node_doc["name"].lower() == edge_info["target"].lower():
#                 target_key = node_doc["key"]
#         if not source_key or not target_key:
#             continue  # skip if we can't find them

#         # 5b. embed relationship/fact
#         fact_text = edge_info.get("description", "")
#         fact_embed = await llm.invoke(
#             [{"role": "user", "content": f"Embed: {fact_text}"}]
#         )
#         fact_vector = fact_embed.get("embedding", [])

#         # 5c. search for similar edges to deduplicate
#         candidate_edges = await store.search(
#             namespace,
#             query=fact_vector,
#             filter={"type": {"$eq": "edge"}, "group_id": {"$eq": group_id}},
#             limit=5,
#         )
#         # if we find near-duplicate edges, we might skip or merge. We'll just create new for demonstration
#         edge_key = f"{source_key}--{edge_info['type']}--{target_key}"
#         edge_doc = {
#             "type": "edge",
#             "group_id": group_id,
#             "source_key": source_key,
#             "target_key": target_key,
#             "edge_type": edge_info["type"],
#             "fact": fact_text,
#             "embedding": fact_vector,
#             "created_at": str(datetime.utcnow()),
#         }
#         await store.aput(namespace, edge_key, edge_doc)
#         newly_created_edges.append({**edge_doc, "key": edge_key})

#     # 6. Create “session → entity” edges if desired
#     #    e.g., store a doc of type=“edge”, edge_type=“MENTIONS”
#     #    For brevity, just do a quick loop
#     for node_doc in resolved_nodes:
#         mention_key = f"{session_key}--MENTIONS--{node_doc['key']}"
#         mention_edge = {
#             "type": "edge",
#             "group_id": group_id,
#             "source_key": session_key,  # the session doc
#             "target_key": node_doc["key"],  # an entity doc
#             "edge_type": "MENTIONS",
#             "fact": f"session mentions {node_doc['name']}",
#             "created_at": str(datetime.utcnow()),
#         }
#         await store.aput(namespace, mention_key, mention_edge)

#     return {
#         "session_key": session_key,
#         "new_entities": newly_created_nodes,
#         "resolved_entities": resolved_nodes,
#         "new_edges": newly_created_edges,
#     }

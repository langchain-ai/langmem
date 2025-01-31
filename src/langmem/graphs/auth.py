import logging
from typing import NotRequired

from langgraph_api.auth.langsmith.client import auth_client
from langgraph_sdk import Auth
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

auth = Auth()


class AuthDict(TypedDict):
    organization_id: str
    tenant_id: str
    user_id: NotRequired[str]
    user_email: NotRequired[str]


@auth.authenticate
async def ls_authenticate(
    headers: dict,
) -> dict:
    headers = [
        ("X-Api-Key", headers.get(b"x-api-key", b"").decode("utf-8")),
    ]
    if not any(h[1] for h in headers):
        raise ValueError("Missing authentication headers")
    async with auth_client() as auth:
        res = await auth.get(
            "/auth/public", headers=[h for h in headers if h[1] is not None]
        )
        if res.status_code == 401:
            raise ValueError("Invalid token")
        elif res.status_code == 403:
            raise ValueError("Forbidden")
        else:
            res.raise_for_status()
            auth_dict: AuthDict = res.json()

    return {
        **auth_dict,
        "identity": auth_dict.get("user_id"),
        "auth_type": "langsmith",
    }


@auth.on
async def block(
    ctx: Auth.types.AuthContext,
    value: dict,
):
    if isinstance(ctx.user, Auth.types.StudioUser):
        return True
    assert False


@auth.on.threads
async def accept(ctx: Auth.types.AuthContext, value: Auth.types.on.threads.value):
    if isinstance(ctx.user, Auth.types.StudioUser):
        return True
    logger.warning(f"Accepting {ctx.user.identity} with {ctx.resource} / {ctx.action}.")
    filters = {"owner": ctx.user.identity}
    metadata = value.setdefault("metadata", {})
    metadata.update(filters)
    return filters


@auth.on.store
async def filter_store_requests(
    ctx: Auth.types.AuthContext, value: Auth.types.on.store.value
):
    if isinstance(ctx.user, Auth.types.StudioUser):
        return True
    namespace = value.get("namespace") or ()
    if not namespace:
        value["namespace"] = (ctx.user.identity,)
    elif ctx.user.identity != namespace[0]:
        value["namespace"] = (ctx.user.identity, *namespace)
    return True

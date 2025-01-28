from langgraph_sdk import Auth
import logging

logger = logging.getLogger(__name__)

auth = Auth()


# Very permissive.
@auth.authenticate
def authenticate():
    return "Authenticated"


@auth.on
async def block(
    ctx: Auth.types.AuthContext,
    value: dict,
):
    assert False


@auth.on.threads
async def accept(ctx: Auth.types.AuthContext, value: Auth.types.on.threads.value):
    logger.warning(f"CHecking thread acceptance; {ctx.resource} {ctx.permissions}")
    # Permit
    return {}

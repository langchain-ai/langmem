from langgraph_sdk import Auth


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


@auth.on.threads.create_run
async def accept(ctx: Auth.types.AuthContext, value: Auth.types.on.threads.create_run):
    # Permit
    return {}

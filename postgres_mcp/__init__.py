def main():
    import asyncio
    import sys
    from . import server

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(server.main())


__all__ = ["main"]

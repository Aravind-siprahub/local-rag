"""Local development entry point.

`uvicorn app.main:app` on Windows creates its asyncio event loop (via
`asyncio.run()` inside `Server.run()`) *before* importing `app.main` — so
setting the Windows event loop policy inside `app/main.py` is too late;
psycopg3's async mode has already been handed a `ProactorEventLoop` it
can't use. Setting the policy here, before uvicorn is even imported,
guarantees it's in place before any event loop is created.

Run with:
    python run.py

(Equivalent to `uvicorn app.main:app --reload`, but with the Windows fix
applied first. In Docker/Linux production this file isn't needed — the
platform check in `app/main.py`'s import of this module is a no-op there —
but using `python run.py` everywhere keeps local dev and prod startup
symmetric.)
"""
import asyncio
import os
import sys

# Ensure immediate terminal output on Windows by forcing line-buffered stdout/stderr
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

if sys.platform == "win32" and sys.version_info < (3, 14):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore  # noqa

import uvicorn  # noqa: E402  (must follow the policy fix above)

import pathlib

if __name__ == "__main__":
    app_dir = str(pathlib.Path(__file__).resolve().parent / "app")
    # On Windows, Uvicorn's reload subprocess spawns a child worker that initializes
    # ProactorEventLoop before importing app.main, breaking psycopg3 async connection.
    # Running directly with reload=False on Windows preserves WindowsSelectorEventLoopPolicy.
    should_reload = os.getenv("RELOAD", "false" if sys.platform == "win32" else "true").lower() == "true"
    
    try:
        if should_reload and sys.platform != "win32":
            uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=[app_dir])
        else:
            from app.main import app
            uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    except (KeyboardInterrupt, SystemExit):
        print("\n[INFO] Server stopped by user.")
    except Exception as e:
        print(f"\n[ERROR] Server error: {e}")



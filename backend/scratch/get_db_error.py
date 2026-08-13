import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from dotenv import load_dotenv

async def main():
    backend_dir = os.path.dirname(os.path.dirname(__file__))
    load_dotenv(os.path.join(backend_dir, ".env"))
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Error: DATABASE_URL environment variable not set.")
        return
        
    db_url = db_url.replace("postgresql://", "postgresql+psycopg://")
    engine = create_async_engine(db_url)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT id, status, error_message, job_type FROM processing_jobs ORDER BY created_at DESC LIMIT 5;"))
            
            output_file = os.path.join(os.path.dirname(__file__), "db_error.txt")
            with open(output_file, "w") as f:
                for row in result:
                    f.write(f"ID: {row[0]}, Status: {row[1]}, Type: {row[3]}, Error: {row[2]}\n")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

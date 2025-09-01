import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from loguru import logger
from src.app.api.api import create_app


app: FastAPI = create_app()  


def main():
    """entrypoint"""
    load_dotenv()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info("serving...")
    uvicorn.run(
        "src.main:app",  # points uvicorn at the app defined above
        host=host,
        port=port,
        reload=True,      # auto-reload during dev
    )


if __name__ == "__main__":
    main()
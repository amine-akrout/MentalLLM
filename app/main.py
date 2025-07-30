"""
MentalLLM API
"""

import ollama
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="MentalLLM API",
    description="API for the locally hosted Mental Health Assistant LLM",
    version="1.0.0",
)


# Pydantic model for request body
class ChatRequest(BaseModel):
    prompt: str


# Pydantic model for response body
class ChatResponse(BaseModel):
    response: str


# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API is running."""
    return {"status": "healthy"}


# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a prompt to the MentalLLM model and get a response.
    """
    try:
        logger.info(f"Received prompt: {request.prompt}")

        # Use the ollama client to generate a response
        response = ollama.generate(
            model="mental-llm-phi3",
            prompt=request.prompt,
            stream=False,  # Get full response at once
        )
        # Extract the text response
        reply = response.get("response", "").strip()
        if not reply:
            logger.warning("Received empty response from Ollama.")
            raise HTTPException(status_code=500, detail="Empty response from model.")

        logger.info(f"Generated response: {reply}")
        return ChatResponse(response=reply)

    except ollama.ResponseError as e:
        logger.error(f"Ollama Response Error: {e}")
        raise HTTPException(status_code=502, detail=f"Ollama Error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during chat: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

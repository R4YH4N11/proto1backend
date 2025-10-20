from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.schemas import ChatRequest, ChatResponse
from app.services.llm import get_chat_service

app = FastAPI(
    title="Hospital Assistant Chatbot",
    description="Conversational receptionist assistant backed by Gemini and LangChain.",
    version="0.1.0",
)

# Allow the Lovable frontend to call the API.
origins = [
    "https://lovable.dev/projects/cfbe1d2e-36a0-4b53-96e5-deefa67b8c41",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Simple health-check endpoint."""
    return {"status": "ok", "model": settings.gemini_model}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Accept a user message and return the assistant reply."""
    try:
        reply = await run_in_threadpool(
            get_chat_service().generate_reply,
            request.user_message,
            request.history,
            request.conversation_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise HTTPException(
            status_code=500, detail="Failed to generate reply."
        ) from exc
    return ChatResponse(reply=reply)

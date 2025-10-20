# Hospital Assistant Chatbot

A FastAPI-based receptionist assistant that uses Gemini via LangChain to converse with patients and call live hospital APIs for doctor search, availability, appointment lookups, and bookings.

## Prerequisites

- Python 3.11+
- Google Generative AI key exported as `GOOGLE_API_KEY`
- Hospital backend client ID exported as `HOSPITAL_CLIENT_ID` (needed for booking)

### Environment variables

- `GEMINI_MODEL` (defaults to `gemini-2.5-flash`)
- `LLM_TEMPERATURE` (defaults to `0.2`)
- `HOSPITAL_API_BASE_URL` (defaults to `http://34.93.7.250:8001/api`)
- `HTTP_TIMEOUT_SECONDS` (defaults to `10`)

## Install & Run

```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows; use `source .venv/bin/activate` on POSIX
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Once running, POST to `http://localhost:8000/chat` with:

```json
{
  "user_message": "Hi, can you find a cardiologist in Pune next Tuesday?",
  "conversation_id": "session-123"
}
```

Provide a stable `conversation_id` to let the server remember context automatically between requests (up to the five most recent messages). If you omit it, the server falls back to a single shared development session, so supplying an explicit ID is recommended for multi-user scenarios; you can also pass explicit `history` entries if you prefer to manage context yourself. The LangChain agent keeps the supplied context, decides when to call each hospital API tool, and blends the raw tool output back into a natural reply so the conversation feels seamless. Only the five most recent messages are forwarded to Gemini to keep context focused, and common specialty synonyms (including Hindi/Marathi terms) are auto-normalised before calling the doctor search API.

## Next Steps

- Ground responses in the hospital knowledge base via retrieval (RAG).
- Persist conversation state per patient/session so the API no longer has to send prior turns.
- Add guardrails such as double-confirming appointments before booking and structured error recovery.*** End Patch

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.schemas import Message
from app.services.memory import ConversationMemory
from app.tools import get_hospital_tools


class GeminiChatService:
    """Gemini-powered assistant that can call hospital backend tools."""

    SYSTEM_PROMPT = (
        "You are MedAssist, a hospital front desk assistant. "
        "Greet patients warmly, gather intent, and answer questions using only the "
        "provided information. Confirm doctor, patient, and timing before booking. "
        "If details are missing, ask focused follow-up questions. "
        "Always verify doctor availability with the search tool before promising an appointment. "
        "If the search tool does not return the requested doctor, explain that they are unavailable "
        "and offer alternatives such as searching other specialists. Respond in the same language "
        "the patient uses, and when the patient writes in Hindi or Marathi, answer using Devanagari script. "
        "Keep responses concise, empathetic, and professional."
    )
    MAX_HISTORY_MESSAGES = 5
    DEFAULT_CONVERSATION_ID = "_default_session"

    def __init__(self) -> None:
        self._llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.require_google_api_key(),
            temperature=settings.llm_temperature,
        )
        self._tools = get_hospital_tools()
        self._tool_map = {tool.name: tool for tool in self._tools}
        self._llm_with_tools = self._llm.bind_tools(self._tools)
        self._system_message = SystemMessage(content=self.SYSTEM_PROMPT)
        self._max_tool_iterations = 3
        self._memory = ConversationMemory(self.MAX_HISTORY_MESSAGES)

    def _map_history(self, history: Iterable[Message]) -> List[BaseMessage]:
        """Convert API message schema into LangChain message objects."""
        mapped: List[BaseMessage] = []
        for item in history:
            if item.role == "user":
                mapped.append(HumanMessage(content=item.content))
            elif item.role == "assistant":
                mapped.append(AIMessage(content=item.content))
            # Ignore external system messages to avoid conflicts.
        return mapped

    @staticmethod
    def _stringify_content(message: AIMessage) -> str:
        """Extract textual content from Gemini responses."""
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for chunk in content:
                if isinstance(chunk, dict):
                    if chunk.get("type") == "text" and chunk.get("text"):
                        parts.append(chunk["text"])
                elif isinstance(chunk, str):
                    parts.append(chunk)
            return "\n".join(parts)
        return str(content)

    def _call_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str | None
    ) -> ToolMessage:
        tool = self._tool_map.get(tool_name)
        if tool is None:
            return ToolMessage(
                content=f"Requested tool '{tool_name}' is not available.",
                name=tool_name,
                tool_call_id=tool_call_id,
            )
        try:
            result = tool.invoke(tool_args or {})
        except Exception as exc:  # pragma: no cover - defensive
            result = f"Tool '{tool_name}' failed: {exc}"
        return ToolMessage(
            content=str(result),
            name=tool_name,
            tool_call_id=tool_call_id,
        )

    def generate_reply(
        self,
        user_message: str,
        history: Iterable[Message],
        conversation_id: Optional[str] = None,
    ) -> str:
        """Generate a response from Gemini given the prior conversation."""
        effective_id: Optional[str]
        if conversation_id:
            effective_id = conversation_id
        elif not history:
            effective_id = self.DEFAULT_CONVERSATION_ID
        else:
            effective_id = None

        if effective_id:
            if history:
                self._memory.set_history(effective_id, history)
            history_source = self._memory.get_history(effective_id)
        else:
            history_source = list(history)

        trimmed_history = history_source[-self.MAX_HISTORY_MESSAGES :]
        lc_history = self._map_history(trimmed_history)
        conversation: List[BaseMessage] = [
            self._system_message,
            *lc_history,
            HumanMessage(content=user_message),
        ]

        reply_text: Optional[str] = None
        for _ in range(self._max_tool_iterations):
            response = self._llm_with_tools.invoke(conversation)
            conversation.append(response)
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                reply_text = self._stringify_content(response)
                break

            for call in tool_calls:
                tool_name = call.get("name")
                tool_args = call.get("args") or {}
                tool_call_id = call.get("id")
                tool_message = self._call_tool(tool_name, tool_args, tool_call_id)
                conversation.append(tool_message)

        if reply_text is None:
            final_response = self._llm.invoke(conversation)
            reply_text = self._stringify_content(final_response)

        if effective_id:
            self._memory.append_messages(
                effective_id,
                [
                    Message(role="user", content=user_message),
                    Message(role="assistant", content=reply_text),
                ],
            )

        return reply_text


@lru_cache
def get_chat_service() -> GeminiChatService:
    """Return a singleton GeminiChatService instance."""
    return GeminiChatService()

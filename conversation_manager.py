"""
Conversation Manager
Handles multi-turn conversations with the AI model
"""

import asyncio
import logging
from typing import List, Dict, Optional
from datetime import datetime

from local_model_executor import LocalModelExecutor
from mcp_discovery import MCPDiscovery

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation history and context
    """

    def __init__(self, model: LocalModelExecutor, mcp_discovery: MCPDiscovery):
        self.model = model
        self.mcp_discovery = mcp_discovery
        self.history: List[Dict[str, str]] = []
        self.max_history = 10  # Keep last 10 messages

    async def send_message(self, message: str) -> str:
        """
        Send a message to the AI and get response

        Args:
            message: User message

        Returns:
            AI response
        """
        # Add to history
        self.history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })

        # Build context from history
        context = self._build_context()

        # System prompt
        system_prompt = self._get_system_prompt()

        # Get AI response
        try:
            response = await self.model.execute(
                prompt=f"{context}\n\nUser: {message}\n\nAssistant:",
                system_prompt=system_prompt,
                max_tokens=2048,
                temperature=0.7
            )

            # Add to history
            self.history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })

            # Trim history if needed
            if len(self.history) > self.max_history * 2:
                self.history = self.history[-self.max_history * 2:]

            return response

        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

    def _build_context(self) -> str:
        """Build conversation context from history"""
        if not self.history:
            return ""

        context_lines = []
        for msg in self.history[-10:]:  # Last 5 exchanges
            role = msg["role"].capitalize()
            content = msg["content"]
            context_lines.append(f"{role}: {content}")

        return "\n\n".join(context_lines)

    def _get_system_prompt(self) -> str:
        """Get system prompt for the AI"""
        # Get available MCP tools
        tools = self.mcp_discovery.get_all_tools()
        tools_desc = []

        for server, tool_list in tools.items():
            tools_desc.append(f"\n{server}:")
            for tool in tool_list:
                tools_desc.append(f"  - {tool}")

        tools_text = "\n".join(tools_desc) if tools_desc else "No MCP tools available yet"

        return f"""You are Code-God, an autonomous AI development assistant that helps developers build applications.

You have access to the following MCP tools:
{tools_text}

Your capabilities:
- Answer questions about software development
- Help design and architect applications
- Guide users through building projects
- Suggest best practices and patterns
- Explain code and concepts

When users want to build a project:
- Ask clarifying questions if needed
- Suggest appropriate technologies
- Help them use the /build command with a clear description

Be helpful, concise, and practical. Focus on getting things done."""

    def clear_history(self):
        """Clear conversation history"""
        self.history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.history.copy()

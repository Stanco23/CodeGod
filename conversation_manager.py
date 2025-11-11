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

            # Check for tool calls and execute them
            if "<TOOL_CALL>" in response:
                response = await self._execute_tool_calls(response)

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
            for tool in tool_list:
                tools_desc.append(f"  - {server}.{tool}")

        tools_text = "\n".join(tools_desc) if tools_desc else "  - No MCP tools installed yet"

        return f"""You are Code-God, an AI development assistant.

AVAILABLE TOOLS:
{tools_text}

HOW TO USE TOOLS:
To use a tool, include this in your response:
<TOOL_CALL>
server: tool_name
arguments:
  arg1: value1
  arg2: value2
</TOOL_CALL>

EXAMPLES:
User: "List files in my project"
You: Let me check the files.
<TOOL_CALL>
server: filesystem
tool: list_directory
arguments:
  path: .
</TOOL_CALL>

User: "Create a file called test.txt"
You: I'll create that file for you.
<TOOL_CALL>
server: filesystem
tool: write_file
arguments:
  path: test.txt
  content: Hello from Code-God!
</TOOL_CALL>

IMPORTANT:
- You CAN and SHOULD use these tools
- Use tools to help users with file operations, git commands, etc.
- Always respond with text first, then tool calls if needed
- For complex tasks, suggest using /build command

Your main job: Help developers build applications and use tools when helpful."""

    async def _execute_tool_calls(self, response: str) -> str:
        """
        Parse and execute tool calls from response

        Args:
            response: AI response with tool calls

        Returns:
            Response with tool results appended
        """
        import re

        # Extract all tool calls
        tool_calls = []
        pattern = r'<TOOL_CALL>(.*?)</TOOL_CALL>'
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            try:
                # Parse tool call
                lines = match.strip().split('\n')
                server_name = None
                tool_name = None
                arguments = {}

                current_key = None
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith('server:'):
                        server_name = line.split(':', 1)[1].strip()
                    elif line.startswith('tool:'):
                        tool_name = line.split(':', 1)[1].strip()
                    elif line == 'arguments:':
                        current_key = 'arguments'
                    elif current_key == 'arguments' and ':' in line:
                        key, value = line.split(':', 1)
                        arguments[key.strip()] = value.strip()

                if server_name and tool_name:
                    tool_calls.append({
                        'server': server_name,
                        'tool': tool_name,
                        'arguments': arguments
                    })

            except Exception as e:
                logger.error(f"Failed to parse tool call: {e}")
                continue

        # Execute tool calls
        results_text = []
        for call in tool_calls:
            try:
                logger.info(f"Executing {call['server']}.{call['tool']}")
                result = await self.mcp_discovery.call_tool(
                    server_name=call['server'],
                    tool_name=call['tool'],
                    arguments=call['arguments']
                )

                result_str = str(result)[:500]  # Limit output
                results_text.append(f"\n\n[Tool Result - {call['tool']}]\n{result_str}")

            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                results_text.append(f"\n\n[Tool Error - {call['tool']}]\n{str(e)}")

        # Remove tool call blocks from response and add results
        cleaned_response = re.sub(pattern, '', response, flags=re.DOTALL)

        if results_text:
            cleaned_response += '\n' + '\n'.join(results_text)

        return cleaned_response

    def clear_history(self):
        """Clear conversation history"""
        self.history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.history.copy()

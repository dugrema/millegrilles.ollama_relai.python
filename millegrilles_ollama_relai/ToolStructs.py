import asyncio

from typing import Any, Optional

from millegrilles_ollama_relai.OllamaContext import OllamaContext


class OllamaTool:
    """
    Ollama tool abstract. Use concepts from MCP (resources, tools, prompts).
    Resources and tools are returned using Ollama structure.
    """

    def __init__(self):
        self._tools: list[dict[str, Any]] = list()
        # List of tools to use by Ollama, removes the function calls
        self._tool_list_cache: Optional[list[dict[str, Any]]] = None

    @property
    def name(self):
        raise NotImplementedError()

    @property
    def tools(self) -> list[dict[str, Any]]:
        if self._tool_list_cache is None:
            # Make a list to return to Ollama, only keep type and function elements
            ollama_list = list()
            for tool in self._tools:
                ollama_list.append({'type': tool['type'], 'function': tool['function']})
            self._tool_list_cache = ollama_list

        return self._tool_list_cache

    async def run_tool(self, user_profile: dict, _context: OllamaContext, function_name: str, args: Optional[dict: [str, Any]]):
        if args is None:
            args = dict()

        args['user_profile'] = user_profile  # Inject user_profile in params

        try:
            tool = [t for t in self._tools if t['function']['name'] == function_name].pop()
        except (KeyError, IndexError):
            raise Exception(f'Unknown tool: {function_name}')

        try:
            function_to_call = tool['async_call']
        except KeyError:
            return await asyncio.to_thread(tool['call'], **args)
        else:
            return await function_to_call(**args)

import logging

from millegrilles_ollama_relai.AttachmentHandler import AttachmentHandler
from millegrilles_ollama_relai.OllamaContext import OllamaContext
from millegrilles_ollama_relai.OllamaToolsTime import ToolTime
from millegrilles_ollama_relai.ToolStructs import OllamaTool

class OllamaToolHandler:

    def __init__(self, context: OllamaContext, attachment_handler: AttachmentHandler):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context = context
        self.__attachment_handler = attachment_handler
        self.__tool_modules: dict[str, OllamaTool] = dict()

    async def setup(self):
        time_tool = ToolTime()
        self.__tool_modules[time_tool.name] = time_tool

    async def run(self):
        await self.__context.wait()

    def tools(self):
        tools = list()
        for m in self.__tool_modules.values():
            tools.extend(m.tools)
        return tools

    async def run_tool(self, user_profile: dict, tool_call) -> str:
        function_name = tool_call.function.name
        module_name = function_name.split('_')[0]
        try:
            module_info = self.__tool_modules[module_name]
        except KeyError:
            return f'Module {module_name} is not available'
        else:
            arguments = tool_call.function.arguments
            try:
                result: str = await module_info.run_tool(user_profile, self.__context, function_name, arguments)
                return result
            except Exception as e:
                self.__logger.exception("Error running tool")
                return f'ERROR during tool execution: {e}'

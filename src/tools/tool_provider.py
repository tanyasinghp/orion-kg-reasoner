from typing import Any, Self, get_type_hints
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field
from langchain_core.tools.structured import StructuredTool


class ToolInfo(BaseModel):
    """Information about a tool which could be used by the agent.

    Attributes:
        name: Name of the tool.
        description: Description of the tool.
        arguments: Arguments of the tool.
        optional: Flag indicating whether the tool is optional or mandatory. Mandatory tools will always be
            available to the agent, while optional tools are not considered by default but can be dynamically
            selected if desired.
    """
    name: str = Field(frozen=True)
    description: str = Field(frozen=True)
    arguments: dict[str, Any] = Field(frozen=True)
    optional: bool = Field(frozen=True, default=False, validate_default=True)


class ToolProvider:
    """Provider for all tools which could be used by the agent.

    Attributes:
        registered_tools: All tools that were registered with the tool provider. These are all mandatory and optional
            tools the agent knows about, but not all the tools are necessarily used within an agent session.
        tool_info: Information about all registered tools.
        tools_available_to_agent: Names of the tools which should be made available to the agent. All registered tools
            marked as mandatory will always be available to the agent. Optional tools are only made available to the
            agent if they explicitly have been marked for usage via the 'consider_tool' method.
    """

    def __init__(self) -> None:
        self.registered_tools: dict[str, StructuredTool] = {} # tool name -> structured tool
        self.tool_info: dict[str, ToolInfo] = {} # tool name -> tool info
        self.tools_available_to_agent: set[str] = set()

    def register_async_tool(self, func: Callable[..., Awaitable[Any]], optional: bool = False) -> Self:
        """Register a function as a tool with the tool provider.

        Registering a tool adds it to the set of all the mandatory and optional tools the agent knows about,
        but not all the tools are necessarily used within an agent session. All registered tools marked
        as mandatory will always be made available to the agent. Optional tools are only made available to
        the agent if they explicitly have been marked for usage via the 'consider_tool' method.

        Args:
            func: The async function from which to create a tool.
            optional: Flag indicating whether the tool is optional or mandatory.

        Returns:
            Tool provider instance to support method chaining.
        """
        tool = StructuredTool.from_function(coroutine=func, parse_docstring=True)

        # Validate that the tool has a return value of type 'ToolOutput' or is an "in-place" tool returning nothing
        return_type: Any | None = get_type_hints(func).get("return", None)
        if return_type is not None:
            allowed_return_types: set[str] = {"builtins.NoneType", "tools.tool_executor.ToolOutput"}
            return_type_name = f"{return_type.__module__}.{return_type.__qualname__}"
            if return_type_name not in allowed_return_types:
                raise TypeError(f"Function {func.__name__} must return a 'ToolOutput' to be registered as a tool.")

        if tool.name not in self.registered_tools:
            self.registered_tools[tool.name] = tool

            # Customize tool arguments to hide context in the signature
            tool_args = tool.args
            tool_args.pop("context", None)

            # Also, simplify the argument schema a bit by removing schema elements we are not interested in
            for argument_schema in tool_args.values():
                argument_schema.pop("title", None)

            self.tool_info[tool.name] = ToolInfo(
                name=tool.name, description=tool.description, arguments=tool_args, optional=optional
            )

            # All mandatory tools are always available to the agent
            if not optional:
                self.tools_available_to_agent.add(tool.name)

        else:
            raise ValueError(f"Function {tool.name} was already registered as a tool")

        return self

    def consider_tool(self, tool_name: str) -> None:
        """Mark a registered tool as to be made available to the agent.

        This method is typically used by the query parser to dynamically select optional tools
        which could be useful for a given query.

        Args:
            tool_name: Name of the (optional) tool which should be made available to the agent in
                the agent session this method is called in.

        Raises:
            ValueError: If a function with the given name was not registered as a tool beforehand.
        """
        if tool_name in self.registered_tools:
            self.tools_available_to_agent.add(tool_name)
        else:
            raise ValueError(f"Function {tool_name} was not registered as a tool")

    def available_tools(self) -> list[ToolInfo]:
        """Get information about all tools made available to the agent.

        Returns:
             A list of tool information about all registered tools that have been marked to be available to the agent.
        """
        return [tool for tool in self.tool_info.values() if tool.name in self.tools_available_to_agent]

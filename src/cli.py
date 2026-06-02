import asyncio
import cmd
import logging
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.markup import escape

from config import settings
from flavors import create_agent_session
from agent.agent_context import Context
from agent.agent_session import AgentSession, Answer
from agent.agent import AgentResponse, AgentResponseType, ToolCall, ToolOutput


def interactive_shell_intro() -> str:
    llm = settings.llm.ollama.model_name
    return (
        f"Knowledge Graph Reasoner for {settings.flavor.description()} "
        f"(LLM: {llm})\n"
        f"\n"
        f"Type help or ? to list commands."
    )


class AgentShell(cmd.Cmd):
    """An interactive shell for the agent."""
    intro = interactive_shell_intro()
    prompt = ">>> "

    console = Console()
    context: Context | None = None
    _agent_session: AgentSession | None = None

    # Define a single async Runner per shell session so that several
    # top-level async functions can be called in the same event loop.
    session_runner = asyncio.Runner()

    def _print_with_layout(
        self,
        emoji: str,
        step: str,
        duration: str,
        style: str="green",
        highlight: bool=True
    ) -> None:
        """Print arguments in a three-column layout.

        Args:
            emoji: An emoji to print in the first column.
            step: Output of an agent step to print in the second column.
            duration: Duration of an agent step to print in the third column.
            style: Style for the step column.
            highlight: Whether to highlight the step column.
        """
        table = Table(show_header=False, show_footer=False, show_lines=False, show_edge=False, box=None,
                      padding=(0, 1, 0, 0))
        table.add_column("Emoji", justify="center", width=2, highlight=False)
        table.add_column("Step", justify="left", width=120, highlight=highlight, style=style)
        table.add_column("Duration", justify="right", width=15, highlight=True, style="white")
        table.add_row(emoji, step, duration)
        self.console.print(table)

    def after_query_parser_callback(self, context: Context, duration: float, num_tokens: int) -> None:
        self._print_with_layout(
            ":compass:",
            f"Analyzing query consumed {num_tokens} LLM input tokens.",
            f"{duration} s"
        )
        # Show found intents
        self._print_with_layout("", "├── Intents:", "", style="yellow")
        for i, intent_id in enumerate(context.intents, start=1):
            intent = self.agent_session.query_parser.intent_provider.intents_lookup_table[intent_id]
            turnstile = "├──" if i < len(context.intents) else "└──"
            self._print_with_layout(
                "",
                f"│   {turnstile} [{intent_id}] {intent.description}",
                "",
                style="yellow",
                highlight=False
            )

        # Show query-specific background information
        self._print_with_layout("", "├── Query-specific Background Information:", "", style="yellow")
        for i, background_info_id in enumerate(context.query_specific_background_information_ids, start=1):
            background_info = context.background_information[background_info_id]
            turnstile = "├──" if i < len(context.query_specific_background_information_ids) else "└──"
            self._print_with_layout(
                "",
                f"│   {turnstile} [{background_info_id}] {background_info.topic}",
                "",
                style="yellow",
                highlight=False
            )

        self._print_with_layout("", f"└── Target Types: {context.target_types}", "", style="yellow")

    def after_reasoning_step_callback(self, agent_response: AgentResponse, duration: float, num_tokens: int) -> None:
        if agent_response.type == AgentResponseType.ANSWER:
            self._print_with_layout(
                ":light_bulb:",
                f"Reasoning step consumed {num_tokens} LLM input tokens. Answer found!",
                f"{duration} s"
            )
        elif agent_response.type == AgentResponseType.TOOL_CALL:
            self._print_with_layout(
                ":light_bulb:",
                f"Reasoning step consumed {num_tokens} LLM input tokens. Call tool [magenta]'{agent_response.tool_call.tool_name}'[/magenta].",
                f"{duration} s"
            )
            # Reason for tool call
            self._print_with_layout("", f"Reason: [italic]{agent_response.tool_call.reason}", "", style="thistle1")
            # Tool arguments
            self._print_with_layout("", "Arguments:", "", style="yellow")
            for i, (arg_name, arg_value) in enumerate(agent_response.tool_call.args.items(), start=1):
                turnstile = "├──" if i < len(agent_response.tool_call.args) else "└──"
                self._print_with_layout("", f"{turnstile} {arg_name}: {arg_value}", "", style="yellow")

    def after_tool_callback(self, tool_call: ToolCall, tool_output: ToolOutput, duration: float) -> None:
        self._print_with_layout(
            ":hammer_and_wrench:",
            f"Executed tool [magenta]'{tool_call.tool_name}'[/magenta].",
            f"{duration} s"
        )
        # Tool output
        self._print_with_layout("", "Tool output:", "", style="yellow")
        # Show subset of entities in tool output
        self._print_with_layout("", f"├── Entities ({len(tool_output.entities)} total)", "", style="yellow")
        entities_to_show = tool_output.entities[:10]
        not_all_entities_shown = len(entities_to_show) < len(tool_output.entities)
        for i, entity in enumerate(entities_to_show, start=1):
            turnstile = "├──" if i < len(entities_to_show) or not_all_entities_shown else "└──"
            self._print_with_layout("", f"│   {turnstile} [{entity.id}] {entity.name}", "", style="yellow")
        if not_all_entities_shown:
            self._print_with_layout("", "│   └── ...", "", style="yellow")
        # Show subset of relations in tool output
        self._print_with_layout("", f"└── Relations ({len(tool_output.relations)} total)", "", style="yellow")
        relations_to_show = tool_output.relations[:10]
        not_all_relations_shown = len(relations_to_show) < len(tool_output.relations)
        for i, relation in enumerate(relations_to_show, start=1):
            turnstile = "├──" if i < len(relations_to_show) or not_all_relations_shown else "└──"
            self._print_with_layout("", f"    {turnstile} {escape(relation.signature)}", "", style="yellow")
        if not_all_relations_shown:
            self._print_with_layout("", "    └── ...", "", style="yellow")

    @property
    def agent_session(self) -> AgentSession:
        """The agent session used throughout the CLI session."""
        if self._agent_session is None:
            start_time = datetime.now()
            self._agent_session: AgentSession = create_agent_session(
                after_query_parser_callback=self.after_query_parser_callback,
                after_reasoning_step_callback=self.after_reasoning_step_callback,
                after_tool_callback=self.after_tool_callback,
            )
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            self._print_with_layout(":robot:", "Established agent session.", f"{duration} s")

        return self._agent_session

    def emptyline(self) -> None:
        """Method called when an empty line is entered in response to the prompt."""
        # Do nothing. The default implementation of this method repeats
        # the last nonempty command entered, we don't want that.

    def default(self, line: str) -> None:
        """Run the agent with the given line as the query."""
        answer: Answer = self.session_runner.run(self.agent_session.generate_answer(query=line, top_n=10, debug=True))
        self.context = self.agent_session.agent.context
        # Show answer
        self._print_with_layout(
            ":checkered_flag:",
            f"Answer: [italic]{answer.augmented_answer.description}",
            "",
            style="white"
        )
        # Show entities linked in answer
        entities = answer.augmented_answer.links if answer.augmented_answer.links else answer.entities
        for i, entity in enumerate(entities, start=1):
            turnstile = "├──" if i < len(entities) else "└──"
            formatted_entity_types = ", ".join(entity.types)
            self._print_with_layout(
                "",
                f"{turnstile} [{entity.id}] {entity.name} ({formatted_entity_types})",
                "",
                style="white",
                highlight=False
            )

    def do_tools(self, arg: str) -> None:
        """List tools available to the agent."""
        table = Table(show_lines=True, header_style="bold")
        table.add_column("Tool Name", style="cyan", no_wrap=True)
        table.add_column("Description")

        for tool_name, tool_info in self.agent_session.tool_executor_blueprint.tool_provider.tool_info.items():
            table.add_row(tool_name, tool_info.description.replace("\n", " "))

        self.console.print(table)

    def do_parse_query(self, arg: str) -> None:
        """Analyze the given query and create an initial context."""
        start_time = datetime.now()
        self.context = self.session_runner.run(self.agent_session.query_parser.parse_query(query=arg))
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        self.console.print(f"Analyzed query in {duration} seconds.", style="green")

        self.console.print("Intents:", style="bold cyan")
        for intent_id in self.context.intents:
            intent = self.agent_session.query_parser.intent_provider.intents_lookup_table[intent_id]
            self.console.print(f":twisted_rightwards_arrows: [{intent_id}] {intent.description}", style="cyan")

        self.console.print("Query-specific Background Information:", style="bold magenta")
        for background_info_id in self.context.query_specific_background_information_ids:
            background_info = self.context.background_information[background_info_id]
            self.console.print(f":page_facing_up: [{background_info_id}] {background_info.topic}", style="magenta")

        self.console.print(f"[bold]Target Types:[/bold] {self.context.target_types}", style="yellow")

    def do_show_context(self, arg: str) -> None:
        """Show the current agent context."""
        if self.context is not None:
            self.console.print(self.context.model_dump(
                exclude={
                    "query_specific_background_information_ids",
                    "id_to_tool_log",
                    "uri_to_entity"
                }))
        else:
            self.console.print("No agent context present, yet. Run or analyze a query to create an agent context.")

    def do_exit(self, arg: str) -> bool:
        """Exit the interactive agent session."""
        self.console.print(
            f"LLM input tokens consumed in session: {self.agent_session.input_token_counter.num_tokens}",
            style="white"
        )

        # Close the event loop used by the CLI
        self.session_runner.close()

        return True

if __name__ == "__main__":
    # Explicitly set log level for root logger to make sure only error logs are shown
    logging.getLogger().setLevel(logging.ERROR)

    AgentShell().cmdloop()

from tools.generic_tools import GenericTools
from tools.tool_executor import ToolOutput


class AmazonTools:
    def __init__(self, generic_tools: GenericTools) -> None:
        self.generic_tools = generic_tools

    async def optional_tool_get_top_products(self, category_name: str, top_n: int = 5, sort_by: str = "rating") -> ToolOutput:
        """Get top N products in a category sorted by a criterion."""
        return ToolOutput(information=(
            f"To find the top {top_n} products in '{category_name}' sorted by {sort_by}, "
            f"first use 'tool_retrieve_entities' to find products, filter by category, "
            f"then use 'tool_select_entities' with sorting on {sort_by}."
        ))

    async def optional_tool_get_category_summary(self, category_name: str) -> ToolOutput:
        """Get summary of products in a category."""
        return ToolOutput(information=(
            f"To get a summary for category '{category_name}', "
            f"find all products belonging to that category using 'tool_navigate_path', "
            f"then aggregate their properties."
        ))

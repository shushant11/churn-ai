"""MCP (Model Context Protocol) server: exposes the churn tools to ANY MCP client (Claude Desktop, Cursor, VS Code...).
MCP is to AI tools what USB is to devices: one standard plug, so you write the tool once and every assistant can use it.
Run:  python -m churnlab.mcp_server        (needs: pip install mcp)
Claude Desktop config (claude_desktop_config.json):
  {"mcpServers": {"churn-ai": {"command": "python", "args": ["-m", "churnlab.mcp_server"], "cwd": "<path to churn_ai>"}}}"""
from __future__ import annotations
from mcp.server.fastmcp import FastMCP
from . import agent

mcp = FastMCP("churn-ai")


@mcp.tool()
def predict_churn(customer_id: int) -> dict:
    """Calibrated churn probability, risk band and top drivers for a customer in the hold-out set."""
    return agent.predict_churn(customer_id)


@mcp.tool()
def what_if(customer_id: int, changes: dict) -> dict:
    """Re-score a customer after changing fields, e.g. {"IsActiveMember": 1} or {"NumOfProducts": 2}."""
    return agent.what_if(customer_id, changes)


@mcp.tool()
def search_playbook(query: str) -> list[dict]:
    """Search the retention playbook and fair-treatment policy. Cite the returned source and section."""
    return agent.search_playbook(query)


@mcp.tool()
def top_risk(n: int = 5) -> list[dict]:
    """The n riskiest customers in the hold-out portfolio."""
    return agent.top_risk(n)


if __name__ == "__main__":
    mcp.run()

import json
import re
from typing import Optional, Dict

def extract_tool_call(response: str) -> Optional[Dict[str, Any]]:
    json_part = response.split('</think>')[-1].strip()

    try:
        data = json.loads(json_part.strip())
        if isinstance(data, dict) and "name" in data and "arguments" in data:
            return {
                "name": data["name"],
                "arguments": data["arguments"]
            }
    except json.JSONDecodeError:
        pass


def is_tool_schema(obj: dict) -> bool:
    if not isinstance(obj, dict):
        return False

    if 'name' not in obj or not isinstance(obj['name'], str):
        return False
    if 'description' not in obj or not isinstance(obj['description'], str):
        return False

    params = obj.get('parameters')
    if not isinstance(params, dict):
        return False
    if params.get('type') != 'object':
        return False
    if 'properties' not in params or not isinstance(params['properties'], dict):
        return False
    if 'required' not in params or not isinstance(params['required'], list):
        return False

    return True

def execute_tools(tool_call) -> list:
    # gs is tool class
    results = {}
    tool_name = tool_call["name"]
    args = tool_call["arguments"]

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception as e:
            results = {
                "tool": tool_name,
                "status": "error",
                "message": f"Failed to parse arguments: {e}"
            }
    try:
        if tool_name == "news_api_search":
            json_data, page_text = gs.call(args)
            results = {
                "tool": tool_name,
                "status": "success",
                "json_data": json_data,
                "text": page_text,
            }
        elif tool_name == "rag_search":
            rag_results = rt.call(args)
            results = {
                "tool": tool_name,
                "status": "success",
                "data": rag_results,
            }
        else:
            results = {
                "tool": tool_name,
                "status": "error",
                "message": f"Unknown tool: {tool_name}"
            }
    except Exception as e:
        results = {
            "tool": tool_name,
            "status": "error",
            "message": str(e)
        }

    return results
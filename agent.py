import uuid
import copy

class Agent:
    def __init__(self, llm, max_steps=3):
        self.llm = llm
        self.max_steps = max_steps

    def _build_context(self, memory_buff: List[Message]):
        return [msg.model_dump() for msg in memory_buff]

    def run(self, query: str) -> str:
        session_id = str(uuid.uuid4())
        memory_tracker = AgentSessionTracker(session_id=session_id)
        SYSTEM_PROMPT_MULTI_QUERY = None

        messages = [
            SystemMessage(content=SYSTEM_PROMPT, role="system"),
            UserMessage(content=query, role="user"),
        ]

        for step in range(self.max_steps):
            sys_message = []
            memory_tracker.add_message(messages)
            context = self._build_context(memory_tracker.memory_buffer)
            
            response = self.llm.create_chat_completion(
                messages=context,
                temperature=0,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )
            content = response['choices'][0]['message']['content']
            print(f"\nStep {step} response:\n{content}")
            
            if tool_call := extract_tool_call(content):
                print(f"🛠️ Executing tool: {tool_call['name']} with args: {tool_call['arguments']}")

                
                
                formatted_results = []
                if tool_call["name"] == "news_api_search":
                    tool_results = execute_tools(tool_call)

                    if tool_results["status"] == "success":
                        formatted = f"Article content:\n{tool_results['json_data']}"
                        formatted_results.append(formatted)

                        # Подготовка промпта для rag_search
                        MULTI_QUERY_CALL_PROMPT = """
                    You are an AI language model assistant. Your task is to generate five different versions of the given user query.

                    You must respond with ONLY the following JSON format — do not add explanations, markdown, or natural language:

                    {{
                    "name": "rag_search",
                    "arguments": {{
                        "queries": [
                        "reformulated_query_1",
                        "reformulated_query_2",
                        "reformulated_query_3",
                        "reformulated_query_4",
                        "reformulated_query_5"
                        ]
                    }}
                    }}

                    Original query: {query}
                    """.strip()


                        messages = [
                            LlmMessage(content=content, role="assistant", tool_calls=[tool_results]),
                            UserMessage(content="Tool execution results:\n" + "\n\n".join(formatted_results), role="user"),
                            SystemMessage(content=MULTI_QUERY_CALL_PROMPT.format(query=query), role="system")
                        ]
                        
                if tool_call["name"] == "rag_search":
                    rag_call = copy.deepcopy(tool_call)
                    rag_call['arguments']['page_text'] = tool_results
                    tool_results = execute_tools(rag_call)
                    if tool_results["status"] == "success":
                        formatted_results.append(tool_results['data'])
                
            else:
                return tool_results

        return tool_results
class AgentMemory(BaseModel):
    # Logicaly it seems better to store this class as an item in the session store.
    # We can have different configs for different memories.
    memory_buffer: List[Message] = Field(default_factory=list)
    max_memory_size: int = Field(default=10)

class AgentSessionTracker(BaseModel):
    session_id: str = Field(..., description="Unique session ID")
    user_id: str = Field(default_factory=str)
    session_store: Dict[str, List] = Field(default_factory=dict)
    
    # Initializing the agent memory
    agent_memory: AgentMemory = Field(default_factory=AgentMemory)
    
    @property
    def memory_buffer(self) -> List[Message]:
        if self.agent_memory.memory_buffer is None:
            print(f"[Session: {self.session_id}] Memory buffer was empty. Creating a new one.")
            self.agent_memory.memory_buffer = []
        return self.agent_memory.memory_buffer

    def _check_memory_size(self):
        while len(self.memory_buffer) > self.agent_memory.max_memory_size:
            self.memory_buffer.pop(0)
            raise Warning(
                f"[Session: {self.session_id}] Memory buffer size exceeded. "
                f"Removing the oldest message to maintain the limit of {self.agent_memory.max_memory_size}."
            )

    def add_message(self, messages: Union[Message, List[Message]]):
        if isinstance(messages, list):
            if not all(isinstance(m, Message) for m in messages):
                raise ValueError("All items in the list must be of type Message.")
        elif isinstance(messages, Message):
            messages = [messages]
        else:
            raise ValueError("'messages' must be of type Message or List[Message].")
        self.memory_buffer.extend(messages)
        self._check_memory_size()
    
    def clean_memory(self):
        self.agent_memory.memory_buffer = []
    
    def get_by_session_id(self, session_id: str) -> Optional[AgentMemory]:
        if session_id not in self.session_store:
            self.session_store[session_id] = self.memory_buffer
        return self.session_store[session_id]
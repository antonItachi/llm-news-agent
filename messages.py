from pydantic import BaseModel, Field
from typing import List, Dict, Any
import requests
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Union, Literal


class Message(BaseModel):
    name: str = "message"
    role: Literal["user", "assistant", "system"]
    content: str = Field(default_factory=str)

    def __repr__(self):
        return f"\n{self.content}\n"
    
class LlmMessage(Message):
    tool_calls: List[Dict] = Field(default_factory=list)

    def __repr__(self):
        return f"\n{self.content}\n"
    
    def model_dump(self, **kwargs):
        base = super().model_dump(**kwargs)
        base["formatted"] = {"role": self.role, "content": self.content}
        return base["formatted"]
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        yield from self.model_dump().items()

class UserMessage(Message):
    user_id: str = Field(default_factory=str)

    def __repr__(self):
        return f"\n{self.content}\n"
    
    def model_dump(self, **kwargs):
        base = super().model_dump(**kwargs)
        base["formatted"] = {"role": self.role, "content": self.content}
        return base["formatted"]
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        yield from self.model_dump().items()

class SystemMessage(Message):
    def __repr__(self):
        return f"\n{self.content}\n"
    
    def model_dump(self, **kwargs):
        base = super().model_dump(**kwargs)
        base["formatted"] = {"role": self.role, "content": self.content}
        return base["formatted"]
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        yield from self.model_dump().items()
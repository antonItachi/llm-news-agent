import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain.text_splitter import RecursiveCharacterTextSplitter


class BaseTool(ABC):
    name: str = ''
    description: str = ''
    parameters: Union[List[dict], dict] = []

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        if not self.name:
            raise ValueError(
                f'You must set {self.__class__.__name__}.name, either by @register_tool(name=...) or explicitly setting {self.__class__.__name__}.name'
            )
        if isinstance(self.parameters, dict):
            if not is_tool_schema({'name': self.name, 'description': self.description, 'parameters': self.parameters}):
                raise ValueError(
                    'The parameters, when provided as a dict, must confirm to a valid openai-compatible JSON schema.')

    @abstractmethod
    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict]:
        """The interface for calling tools.

        Each tool needs to implement this function, which is the workflow of the tool.

        Args:
            params: The parameters of func_call.
            kwargs: Additional parameters for calling tools.

        Returns:
            The result returned by the tool, implemented in the subclass.
        """
        raise NotImplementedError

    @property
    def function_info(self) -> dict:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters
        }


class NewsAPITool(BaseTool):
    name: str = 'news_api_search'
    description: str = 'Search for news using NewsAPI. Returns latest news articles with direct links.'

    class Config:
        api_key: str = "9efd68b03f504c759af44000a347b287"
        max_retries: int = 2

    parameters: dict = {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': 'Search query (e.g. "Ukraine Russia ceasefire")'
            },
            'language': {
                'type': 'string',
                'description': 'Language code (e.g. "ru", "en")',
                'default': 'en'
            },
            'max_results': {
                'type': 'integer',
                'description': 'Number of results (1-30)',
                'default': 3
            },
            'sort_by': {
                'type': 'string',
                'description': 'Sorting method: "relevancy", "popularity", or "publishedAt"',
                'default': 'publishedAt'
            }
        },
        'required': ['query'],
    }

    def call(self, params: dict) -> List[Dict[str, Any]]:
        """Execute news search via NewsAPI"""
        query = params['query']
        language = params.get('language', 'en')
        max_results = min(params.get('max_results', 3), 30)
        sort_by = params.get('sort_by', 'publishedAt')

        for attempt in range(self.Config.max_retries + 1):
            try:
                url = (
                    f"https://newsapi.org/v2/everything?"
                    f"q={query}&"
                    f"language={language}&"
                    f"pageSize={max_results}&"
                    f"sortBy={sort_by}&"
                    f"apiKey={self.Config.api_key}"
                )

                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()

                if data['status'] == 'ok':
                    return self._format_results(data['articles'])

                raise Exception(f"API error: {data.get('message', 'Unknown error')}")

            except Exception as e:
                if attempt == self.Config.max_retries:
                    return [{
                        'error': f"NewsAPI search failed after {self.Config.max_retries} attempts",
                        'details': str(e)
                    }]
                continue

    def fetch_full_article_text(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=10, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            })
            soup = BeautifulSoup(response.text, "html.parser")

            # Try main content blocks
            for tag in ["article", "main"]:
                container = soup.find(tag)
                if container:
                    paragraphs = container.find_all("p")
                    if paragraphs:
                        return "\n".join(p.get_text(strip=True) for p in paragraphs)

            # Fallback to all <p> tags
            paragraphs = soup.find_all("p")
            return "\n".join(p.get_text(strip=True) for p in paragraphs)

        except Exception as e:
            return f"[Failed to fetch article text: {e}]"

    def _format_results(self, articles: List[Dict]) -> List[Dict[str, Any]]:
        """Standardize NewsAPI response and fetch full article text"""
        formatted_json = []
        formatted_text = []
        for article in articles:
            full_text = self.fetch_full_article_text(article["url"])

            formatted_json.append({
                "title": article["title"],
                "source": article["source"]["name"],
                "url": article["url"],
                "published_at": article["publishedAt"],
                "description": article.get("description"),
                "image_url": article.get("urlToImage")
            })
            formatted_text.append(full_text)

        return formatted_json, formatted_text

    def format_for_display(self, results: List[Dict[str, Any]]) -> str:
        """User-friendly results formatting"""
        if results and 'error' in results[0]:
            return f"Error: {results[0]['error']}\n{results[0].get('details', '')}"

        return '\n\n'.join(
            f"{item['title']}\n"
            f"Source:   {item['source']}\n"
            f"Date:     {item['published_at']}\n"
            f"URL:      {item['url']}\n"
            f"{item.get('description', 'No description')}"
            for item in results
        )

class RagTool(BaseTool):
    name: str = 'rag_search'
    description: str = 'RAG search with NewsAPI data. Returns most relevant chunks.'

    class RagConfig:
        embed_model = model
        tokenizer = tokenizer
        dims = 768
        chunk_size = 256
        chunk_overlap = 20

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        qdrant_client = QdrantClient(host="localhost", port=6333)
        collection_name = "news_articles"

    parameters: dict = {
        'type': 'object',
        'properties': {
            'queries': {
                'type': 'list',
                'description': 'New generated queries for multi-rag.',
            },
        },
        'required': ['queries'],
    }
    
    def call(self, args):
        docs = []
        rag_results = []
        data = args['page_text']
        multi_query = args['queries']

        content = data

        for idx, item_content in enumerate(content['text']):
            text = self._join_words(item_content)
            chunks = self.RagConfig.text_splitter.create_documents(
                [text],
                metadatas=[content['json_data'][idx]]
            )
            docs.extend(chunks)

        embeddings = self._embed_documents(docs)

        self.RagConfig.qdrant_client.recreate_collection(
            collection_name=self.RagConfig.collection_name,
            vectors_config=VectorParams(
                size=self.RagConfig.dims,
                distance=Distance.COSINE
            )
        )

        self.RagConfig.qdrant_client.upsert(
            collection_name=self.RagConfig.collection_name,
            points=[
                {
                    "id": i,
                    "vector": emb.tolist(),
                    "payload": {"text": doc.page_content, **doc.metadata}
                }
                for i, (emb, doc) in enumerate(embeddings)
            ]
        )
        for query in multi_query:
            rag_results.append(self.search(query=query, top_k=4))
            
        return rag_results
    
    def search(self, query: str, top_k: int = 5):
        """Perform a vector similarity search for a given query string."""
        inputs = self.RagConfig.tokenizer(
            query,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors='pt'
        ).to(device)

        with torch.inference_mode():
            output = self.RagConfig.embed_model(**inputs)

        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            query_vector = output.pooler_output.squeeze(0).cpu().numpy()
        else:
            query_vector = output.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

        results = self.RagConfig.qdrant_client.search(
            collection_name=self.RagConfig.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )

        return [
            {
                "score": hit.score,
                "text": hit.payload.get("text"),
                "meta": {k: v for k, v in hit.payload.items() if k != "text"}
            }
            for hit in results
        ]


    def _embed_documents(self, docs):
        results = []
        for doc in docs:
            inputs = self.RagConfig.tokenizer(
                doc.page_content,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors='pt'
            ).to(device)

            with torch.inference_mode():
                output = self.RagConfig.embed_model(**inputs)

            if hasattr(output, "pooler_output") and output.pooler_output is not None:
                emb = output.pooler_output.squeeze(0)
            else:
                emb = output.last_hidden_state.mean(dim=1).squeeze(0)


            results.append((emb.cpu().numpy(), doc))
        return results

    def _join_words(self, sequence):
        if isinstance(sequence, list):
            return " ".join(sequence)
        return sequence
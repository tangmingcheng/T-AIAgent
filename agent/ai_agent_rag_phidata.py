from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.pgvector import PgVector, SearchType
from phi.embedder.ollama import OllamaEmbedder
from phi.assistant import Assistant
from phi.llm.groq import Groq

# Configure the language model
llm = Groq(model="qwen-qwq-32b")

# Configure embedder model
embedder = OllamaEmbedder(model="nomic-embed-text", dimensions=768)


# Use PgVector as the vector database and store embeddings in the `pdf_rag_collection` table
vector_db = PgVector(
    db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    table_name="pdf_rag_collection",
    search_type=SearchType.hybrid,
    embedder=embedder
)

# Create a knowledge base of PDFs
pdf_path = "D:/Chrome/Downloads/ThaiRecipes.pdf"
knowledge_base = PDFKnowledgeBase(
    path=pdf_path,
    vector_db=vector_db
)

# Load the knowledge base: Comment after first run as the knowledge base is already loaded
try:
    knowledge_base.load(recreate=True)
    print("知识库加载完成")
except Exception as e:
    print(f"加载知识库失败: {e}")
    exit()


# Create assistant
assistant = Assistant(
    run_id="run_id",  # use any unique identifier to identify the run
    user_id="user",  # user identifier to identify the user
    llm=llm,
    knowledge_base=knowledge_base,
    use_tools=True,
    show_tool_calls=True,
    search_knowledge=True,
    add_references_to_prompt=True,  # Use traditional RAG (Retrieval-Augmented Generation)
    debug_mode=False,  # Enable debug mode for additional information

)

# Test RAG
questions = [
    "ThaiRecipes.pdf里说了什么？",
    "PDF 中提到了什么菜？"
]

for question in questions:
    print(f"\n问题: {question}")
    assistant.print_response(question, markdown=True,stream = True)
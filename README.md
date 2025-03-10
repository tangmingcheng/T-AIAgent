# T-AIAgent

T-AIAgent is a powerful AI Agent built on Ollama, Phidata, Groq, and Omniparser v2. It features a Retrieval-Augmented Generation (RAG) system, tool calling capabilities, and the ability to directly operate a computer through image parsing.

## Features

### 1. Retrieval-Augmented Generation (RAG)
- Uses pgvector for vector search to enhance the accuracy of AI-generated content.
- Supports custom knowledge bases for local knowledge enhancement.

### 2. Tool Calling
- Supports internet search, email sending, weather queries, and more.
- Expandable API allowing integration of custom plugins.

### 3. Vision-Driven Interaction
- Uses Omniparser v2 for image parsing and automated actions.
- Can be applied to GUI automation, screen analysis, and more.

### 4. Local Deployment and Optimization
- Runs local AI models with Ollama, supporting GPU acceleration.
- Utilizes Phidata for AI workflow management.
- Compatible with Groq for accelerated inference, providing faster processing speeds.

## Installation & Usage

### 1. System Requirements
- Windows 11 (WSL2 recommended) or Linux
- Python 3.8+
- PostgreSQL (required for pgvector)
- Ollama, Phidata, Omniparser v2, Groq SDK

### 2. Installation Steps
```bash
# Clone the repository
git clone https://github.com/your-repo/T-AIAgent.git
cd T-AIAgent

# Install dependencies
pip install -r requirements.txt

# Run the AI Agent example
python ai_agent_rag.py
```

## Configuration

### 1. Ollama Configuration
Ensure Ollama is installed and ready with a model (e.g., `qwen-qwq-32b` or `llama3`).
```bash
ollama run qwen-qwq-32b
```

### 2. Omniparser Configuration
Install Omniparser v2 and configure parsing rules.

## Usage Example
T-AIAgent provides both an interactive CLI and a web interface:
```bash
streamlit run  app.py
```
Then access `http://localhost:8501` to interact with the system.

## Future Plans
- Enhance RAG capabilities and document retrieval optimization.
- Expand tool-calling functionality with more API integrations.
- Incorporate multi-modal AI to handle more complex tasks.

## Contributions
Contributions are welcome! Feel free to submit issues and pull requests.

## License
This project is licensed under the MIT License.



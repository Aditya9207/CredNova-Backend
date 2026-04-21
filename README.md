# CredNova Backend

The backend for CredNova is a FastAPI-based service that handles credit scoring, bank statement analysis, and identity verification. It integrates with MongoDB, external ML models, and LLMs (Ollama/OpenAI) to provide comprehensive credit insights.

## Prerequisites

- **Python**: 3.10 or higher
- **MongoDB**: A running instance of MongoDB
- **Ollama** (Optional): For local LLM processing (default model: `mistral`)
- **OpenAI API Key** (Optional): For advanced AI spending narratives

## Installation & Setup

1. **Navigate to the backend directory**:
   ```bash
   cd backend
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment variables**:
   Copy the example environment file and update it with your settings:
   ```bash
   cp .env.example .env
   # On Windows PowerShell:
   # cp .env.example .env
   ```

## Running the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000`. You can access the automatic API documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.

## Key Components

- **`app.py`**: The main entry point and core API endpoints.
- **`credit_flow.py`**: Handles the logic for loan applications and ML model integration.
- **`statement_service.py`**: Logic for parsing and analyzing bank statements (PDFs and CSVs).
- **`insights_service.py`**: Generates AI-driven spending insights and credit tips.
- **`artifacts/`**: Contains pre-trained model bundles and scalers.

## Backend Technical Stack

- **Framework**: FastAPI
- **Database**: MongoDB (motor/pymongo)
- **ML/Calculations**: Scikit-learn, Pandas, SHAP, Joblib
- **LLM Integration**: OpenAI HTTP API & Local Ollama

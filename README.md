# Text Classifier Backend

A FastAPI-based backend service for hierarchical text classification using OpenAI's language models. This service processes text input and categorizes it according to a predefined hierarchical category structure.

## Features

- **Hierarchical Text Classification**: Categorizes text into multiple layers of categories
- **Sentiment Analysis**: Extracts sentiment information from the text
- **Summary Generation**: Provides summaries for each categorization
- **Flag Detection**: Identifies general or empty comments
- **RESTful API**: Easy integration with frontend applications

## Requirements

- Python 3.9+
- OpenAI API key

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd text-classifier-backend
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   FRONTEND_URL=your_frontend_url_here
   ```

## Usage

### Running the server

Start the server with Uvicorn:

```bash
python app.py
```

This will start the server on `http://0.0.0.0:5001`.

### API Endpoints

#### Classify Text

**Endpoint**: `POST /api/classify`

**Request Body**:
```json
{
  "text": "Your text to classify",
  "categories": {
    "categories": [
      {
        "id": "uuid-1",
        "name": "Category Name",
        "description": "Category Description",
        "user_id": "user-id"
      }
    ],
    "relations": [
      {
        "parent_id": "parent-uuid",
        "child_id": "child-uuid",
        "user_id": "user-id"
      }
    ]
  }
}
```

**Response**:
```json
{
  "classification": "Text summary of the classification",
  "graph_data": {
    "nodes": [
      {
        "id": "uuid-1",
        "name": "Category Name",
        "layer": 1,
        "summary": "Summary of the text for this category",
        "sentiment": "Positive",
        "flag": "General comment",
        "parent_id": null
      }
    ],
    "text": "Original text"
  }
}
```

## How It Works

1. **Category Hierarchy**: The system uses a hierarchical structure of categories, with parent-child relationships.
2. **Layer-by-Layer Processing**: Text is first classified at the top layer, then progressively classified into subcategories.
3. **OpenAI Integration**: Uses OpenAI's models to determine which categories the text belongs to.
4. **Sentiment Analysis**: Extracts sentiment information from the text for each category.
5. **Graph Data Generation**: Formats the output for easy visualization in a graph structure.

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes |
| `FRONTEND_URL` | URL of the frontend application | No |

## Dependencies

- fastapi: Web framework for building APIs
- uvicorn: ASGI server implementation
- openai: Client for the OpenAI API
- python-dotenv: Environment variable management
- pydantic: Data validation and settings management
- pandas: Data manipulation and analysis
- tqdm: Progress bar utility

## Error Handling

The API handles various error scenarios:
- OpenAI API key configuration issues
- Rate limit exceeded
- Regional restrictions (unsupported regions)
- General processing errors

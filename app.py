from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from typing import List, Dict, Optional, Tuple
import os
from dotenv import load_dotenv
import pandas as pd
from classify import process_verbatim_layers, VerbatimClassification

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:5173",  # Local development
    os.getenv("FRONTEND_URL", ""),  # Production frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class Category(BaseModel):
    id: str
    name: str
    description: str
    user_id: str

class CategoryRelation(BaseModel):
    parent_id: str
    child_id: str
    user_id: str

class CategoryData(BaseModel):
    categories: List[Category]
    relations: List[CategoryRelation]

class ClassificationNode(BaseModel):
    id: str
    name: str
    layer: int
    summary: str
    sentiment: str
    flag: str
    parent_id: Optional[str] = None

class ClassificationResult(BaseModel):
    nodes: List[ClassificationNode]
    text: str

# Define request model
class ClassificationRequest(BaseModel):
    text: str
    categories: CategoryData

# Define response model
class ClassificationResponse(BaseModel):
    classification: str
    graph_data: ClassificationResult

def convert_to_dataframes(categories_data: CategoryData) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert the CategoryData to two DataFrames expected by process_verbatim_layers"""
    # Create categories dataframe
    categories_df = pd.DataFrame([
        {
            'uuid': cat.id,  # Using 'uuid' as the column name
            'name': cat.name,
            'description': cat.description
        }
        for cat in categories_data.categories
    ])
    
    # Create relationships dataframe
    relationships_df = pd.DataFrame([
        {
            'parent_uuid': rel.parent_id,  # Using 'parent_uuid' as the column name
            'child_uuid': rel.child_id     # Using 'child_uuid' as the column name
        }
        for rel in categories_data.relations
    ])
    
    return categories_df, relationships_df

@app.post("/api/classify", response_model=ClassificationResponse)
async def classify_text(request: ClassificationRequest):
    try:
        # Convert the categories data to DataFrame format
        df_categories, df_relationships = convert_to_dataframes(request.categories)
        
        # Process the text using process_verbatim_layers
        classifications = process_verbatim_layers(
            client=client,
            verbatim=request.text,
            survey="",  
            df_categories=df_categories,
            df_relationships=df_relationships,
            layer=1,  
            parent_classification=None,
            model_name="gpt-4o-mini",
            temperature=0.7,
            api_key=os.getenv('OPENAI_API_KEY'),
            rate_limit_sleep=1,
            max_retries=3,
            max_trials=1
        )
        
        # Format the classification result for graph visualization
        nodes = []
        for layer, layer_classifications in sorted(classifications.items()):
            for classification in layer_classifications:
                # Skip nodes flagged as "No comment"
                if classification.flag == "No comment":
                    continue
                    
                nodes.append(ClassificationNode(
                    id=classification.category_uuid,
                    name=classification.category_name,
                    layer=classification.layer,
                    summary=classification.summary,
                    sentiment=classification.sentiment,
                    flag=classification.flag,  # Will be "General comment" or empty
                    parent_id=classification.parent_uuid
                ))
        
        # Format the text response
        classification_text = ""
        for node in nodes:
            classification_text += f"Category: {node.name} (Layer {node.layer})\n"
            classification_text += f"Summary: {node.summary}\n"
            classification_text += f"Sentiment: {node.sentiment}\n"
            if node.flag:
                classification_text += f"Flag: {node.flag}\n"
            classification_text += "\n"
        
        return ClassificationResponse(
            classification=classification_text,
            graph_data=ClassificationResult(
                nodes=nodes,
                text=request.text
            )
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        error_message = str(e)
        if "api_key" in error_message.lower():
            return {"classification": "", "error": "OpenAI API key not configured properly"}
        elif "rate limit" in error_message.lower():
            return {"classification": "", "error": "OpenAI rate limit exceeded"}
        elif "unsupported_country_region_territory" in error_message:
            return {"classification": "", "error": "This service is not available in your region. Please use a VPN to access OpenAI services."}
        else:
            return {"classification": "", "error": f"Error classifying text: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)

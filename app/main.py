import os
from fastapi import FastAPI, Query, HTTPException
import pandas as pd
import pyterrier as pt
from pyterrier.measures import *
# from sentence_transformers import CrossEncoder
import re
from typing import List
from pydantic import BaseModel
import pickle

app = FastAPI()

# Initialize pyterrier if not started
if not pt.java.started():
    pt.java.init()

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load necessary files
recipes_df = pd.read_pickle(os.path.join(current_dir, "clean_recipe.pkl"))
indexref = pt.IndexRef.of(os.path.join(current_dir, "recipe_index"))
bm25 = pt.BatchRetrieve(indexref, wmodel="BM25")

# Load pickled crossencoder model
with open(os.path.join(current_dir, 'crossencoder_model.pkl'), 'rb') as f:
    crossmodel = pickle.load(f)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    pattern = re.compile('[\\W_]+')
    text = pattern.sub(' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Cross encoder apply function
def _crossencoder_apply(df: pd.DataFrame):
    return crossmodel.predict(list(zip(df['query'].values, df['text_raw'].values)))

cross_encT = pt.apply.doc_score(_crossencoder_apply, batch_size=128)

# Create pipeline
bm25_reranker_pipeline = (
    bm25 % 50 >>
    pt.text.get_text(indexref, ['text_raw']) >>
    cross_encT % 30
)

class RecipePreview(BaseModel):
    rank: int
    id: int
    name: str
    minutes: int
    description: str | None

class SearchResponse(BaseModel):
    recipes: List[RecipePreview]
    total_results: int

class RecipeDetail(BaseModel):
    id: int
    name: str
    minutes: int
    ingredients: List[str]
    steps: List[str]
    description: str | None

@app.get("/search/", response_model=SearchResponse)
async def search_recipes(
    query: str = Query(..., description="Search query for recipes"),
):
    # Preprocess query
    processed_query = preprocess_text(query)
    
    # Search using pipeline
    results = bm25_reranker_pipeline.search(processed_query)
    results['docno'] = results['docno'].astype(int)
    
    # Merge with recipes dataframe
    matching_recipes = pd.merge(
        results, 
        recipes_df,
        left_on='docno',
        right_on='id',
        how='left'
    )
    
    # Convert to list of dictionaries and process
    recipes_list = []
    for _, row in matching_recipes.iterrows():
        recipe = RecipePreview(
            rank=int(row['rank']),
            id=int(row['id']),
            name=row['name'],
            minutes=int(row['minutes']),
            description=row['description'] if pd.notnull(row['description']) else None
        )
        recipes_list.append(recipe)

    return SearchResponse(
        recipes=recipes_list,
        total_results=len(recipes_list)
    )

@app.get("/recipe/{recipe_id}", response_model=RecipeDetail)
async def get_recipe_details(recipe_id: int):
    # Find recipe by id
    recipe = recipes_df[recipes_df['id'] == recipe_id]
    
    if recipe.empty:
        raise HTTPException(status_code=404, detail="Recipe not found")
        
    row = recipe.iloc[0]
    
    return RecipeDetail(
        id=int(row['id']),
        name=row['name'],
        minutes=int(row['minutes']),
        ingredients=eval(row['ingredients']),
        steps=eval(row['steps']),
        description=row['description'] if pd.notnull(row['description']) else None
    )

@app.get("/")
def read_root():
    return {"Hello": "World 1233"}
from fastapi import FastAPI
import pyterrier as pt
from pyterrier.measures import *

app = FastAPI()

if not pt.java.started():
    pt.java.init()

@app.get("/")
def read_root():
    return {"Hello": "World 1233"}
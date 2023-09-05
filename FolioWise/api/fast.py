import pandas as pd
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return dict(greeting="Hello")

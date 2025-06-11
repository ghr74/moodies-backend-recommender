from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3

app = FastAPI(); 

class MovieSuggestionRequest(BaseModel):
    mood: str


class MovieSuggestionResponse(BaseModel):
    idx: int
    title: str


@app.get("/")
def getMovieSuggestions(MovieSuggestionRequest: req):
    conn = sqlite3.connect('search.db')
    cursor = conn.cursor()

    cursor.execute(
        "SELECT idx, title FROM movies WHERE title LIKE ? LIMIT ? OFFSET ?",
        (mood, take, skip)
    )
    results = cursor.fetchall()
    conn.close()

    return {"Hello": mood};

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from enum import Enum
from typing import List
from contextlib import asynccontextmanager
import lancedb
import numpy as np
import httpx as hx
import json
import polars as pl
import os


class MoodEnum(str, Enum):
    funny = "funny"
    romantic = "romantic"
    scary = "scary"
    inspiring = "inspiring"
    sad = "sad"
    exciting = "exciting"
    suspenseful = "suspenseful"
    comforting = "comforting"
    disgusting = "disgusting"


emotion_vectors = {
    MoodEnum.funny: [1, 0, 0, 0, 0, 0, 0, 0, 0],
    MoodEnum.romantic: [0, 1, 0, 0, 0, 0, 0, 0, 0],
    MoodEnum.scary: [0, 0, 1, 0, 0, 0, 0, 0, 0],
    MoodEnum.inspiring: [0, 0, 0, 1, 0, 0, 0, 0, 0],
    MoodEnum.sad: [0, 0, 0, 0, 1, 0, 0, 0, 0],
    MoodEnum.exciting: [0, 0, 0, 0, 0, 1, 0, 0, 0],
    MoodEnum.suspenseful: [0, 0, 0, 0, 0, 0, 1, 0, 0],
    MoodEnum.comforting: [0, 0, 0, 0, 0, 0, 0, 1, 0],
    MoodEnum.disgusting: [0, 0, 0, 0, 0, 0, 0, 0, 1],
}

mood_keys = [mood.value for mood in MoodEnum]


class RecommendationRequest(BaseModel):
    mood: MoodEnum = Field(..., description="Mood for movie recommendation")
    movie_ids: List[int]
    use_llm: bool = True


class RecommendedMovie(BaseModel):
    title: str
    poster: str


class RecommendationResponse(BaseModel):
    first: RecommendedMovie
    second: RecommendedMovie
    third: RecommendedMovie


class LlmResponse(BaseModel):
    first_place: str
    second_place: str
    third_place: str


class LlmResponseWrapper(BaseModel):
    chosenMovies: str


db_uri = os.getenv("LANCEDB_PATH", "lance")
api_key = os.getenv("GEMINI_API_KEY")


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with hx.AsyncClient() as client:
        app.state.http_client = client
        app.state.conn = lancedb.connect(db_uri)
        yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.post("/recommendation", response_model=RecommendationResponse)
async def get_recommendation(req: RecommendationRequest, request: Request):
    tbl = app.state.conn.open_table("movies")
    search_str = ",".join(str(idx) for idx in req.movie_ids)
    emotion_vector = emotion_vectors[req.mood]

    seed_vectors = (
        tbl.search()
        .where(f"idx IN ({search_str})")
        .limit(3)
        .select(["mood_vector", "word_embedding", "title"])
        .to_polars()
    )

    word_embeddings = seed_vectors["word_embedding"].to_list()
    mean_word_embedding = np.stack(word_embeddings).mean(axis=0)

    mood_vectors = seed_vectors["mood_vector"].to_list()
    mood_vectors.append(emotion_vector)
    mood_vectors.append(emotion_vector)
    mood_vectors.append(emotion_vector)
    mood_vectors.append(emotion_vector)
    mean_mood_vector = np.stack(mood_vectors).mean(axis=0)

    mood_results = (
        tbl.search(mean_mood_vector, vector_column_name="mood_vector")
        .distance_type("cosine")
        .where(f"idx NOT IN ({search_str}) AND score > 75000", prefilter=True)
        .limit(300)
        .to_polars()
    )
    mood_ids: List[int] = mood_results["idx"].to_list()

    results = (
        tbl.search(mean_word_embedding, vector_column_name="word_embedding")
        .distance_type("cosine")
        .where(f"idx IN ({",".join(map(str, mood_ids))})", prefilter=True)
        .limit(50)
        .to_polars()
    )

    if not req.use_llm:
        r = results.head(10).sample(3).to_dicts()
        return {
            "first": {"title": r[0]["title"], "poster": r[0]["poster_path"]},
            "second": {"title": r[1]["title"], "poster": r[1]["poster_path"]},
            "third": {"title": r[2]["title"], "poster": r[2]["poster_path"]},
        }

    try:
        if api_key == None or api_key == "":
            print("NO api Key")
            raise ValueError("Api Key empty, can't request LLM")

        prompt = f"""
    You are an expert movie recommender. Pick the top three movie recommendations for someone looking for a {req.mood.value} movie out of the following list:
    ========
    {'\n'.join(results["title"].to_list())}
    ========
    Reply with only a JSON object in the format: first_place: string; second_place: string; third_place: string;
        """

        client: hx.AsyncClient = request.app.state.http_client

        llm_response = await client.post(
            "https://run.chayns.codes/a1e60852",
            json={"validate": api_key, "prompt": prompt},
        )
        llm_result = llm_response.json()

        llm_respose_wrapper = LlmResponseWrapper.parse_obj(llm_result)

        chosen_movies_json = json.loads(llm_respose_wrapper.chosenMovies)

        llm_placement = LlmResponse.parse_obj(chosen_movies_json)

        print(llm_placement)
        first = results.filter(pl.col("title") == llm_placement.first_place).head(1)
        second = results.filter(pl.col("title") == llm_placement.second_place).head(1)
        third = results.filter(pl.col("title") == llm_placement.third_place).head(1)

        if first.height <= 0 or second.height <= 0 or third.height <= 0:
            raise ValueError

        first_item = first.to_dicts()[0]
        second_item = second.to_dicts()[0]
        third_item = third.to_dicts()[0]

        return {
            "first": {
                "title": first_item["title"],
                "poster": first_item["poster_path"],
            },
            "second": {
                "title": second_item["title"],
                "poster": second_item["poster_path"],
            },
            "third": {
                "title": third_item["title"],
                "poster": third_item["poster_path"],
            },
        }
    except:
        print("falling back to manual results")
        r = results.head(10).sample(3).to_dicts()
        return {
            "first": {"title": r[0]["title"], "poster": r[0]["poster_path"]},
            "second": {"title": r[1]["title"], "poster": r[1]["poster_path"]},
            "third": {"title": r[2]["title"], "poster": r[2]["poster_path"]},
        }

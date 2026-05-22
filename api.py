import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from story2star import build_story2star_coaching

app = FastAPI(title="Story2STAR API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


class StoryRequest(BaseModel):
    story: str
    skill_focus: str
    target_role: str = "General / Any Role"
    answer_length: str = "Detailed (3-4 min)"


@app.post("/generate")
def generate(req: StoryRequest):
    if not req.story.strip():
        raise HTTPException(status_code=400, detail="story is required")
    return build_story2star_coaching(
        req.story,
        req.skill_focus,
        req.target_role,
        req.answer_length,
        save=False,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"app": "Story2STAR API", "docs": "/docs"}

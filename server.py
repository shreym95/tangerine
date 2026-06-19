import json
import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

app = FastAPI()

OLLAMA_BASE = "http://localhost:11434"
MODEL = "gemma4:e4b"
MAX_CONTEXT_CHARS = 400_000


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    think: bool = False


def trim_to_context(messages: list[dict]) -> tuple[list[dict], bool]:
    trimmed = False
    while len(messages) > 1:
        if sum(len(m["content"]) for m in messages) <= MAX_CONTEXT_CHARS:
            break
        messages = messages[1:]
        trimmed = True
    return messages, trimmed


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    messages, trimmed = trim_to_context(
        [{"role": m.role, "content": m.content} for m in req.messages]
    )

    async def generate():
        if trimmed:
            yield f"data: {json.dumps({'type': 'trimmed'})}\n\n"
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE}/api/chat",
                json={"model": MODEL, "messages": messages, "stream": True, "think": req.think},
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = data.get("message", {})
                    if msg.get("content"):
                        yield f"data: {json.dumps({'type': 'content', 'text': msg['content']})}\n\n"
                    if data.get("done"):
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                        break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

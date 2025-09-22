# app.py
import json
import time
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from transformers import pipeline

app = FastAPI()

# Simple emotion classifier (text only)
# Model choice: replace if you have a better domain model
emotion_pipe = pipeline("text-classification",
                        model="j-hartmann/emotion-english-distilroberta-base",
                        return_all_scores=False)

# connected supervisor websockets
supervisors: List[WebSocket] = []

def make_alert(agent_id: str, text: str, label: str, score: float):
    return {
        "type": "alert",
        "agentId": agent_id,
        "text": text,
        "emotion": label,
        "score": float(score),
        "ts": int(time.time() * 1000)
    }

@app.websocket("/ws/agent")
async def ws_agent(websocket: WebSocket):
    """
    Agent will send plain-text transcripts (one message per recognized chunk).
    Example message: "Hello, I need help with my bill"
    """
    await websocket.accept()
    # create a simple agent id for demo; in prod, authenticate and pass real id
    agent_id = f"agent-{int(time.time()*1000) % 100000}"
    try:
        while True:
            data = await websocket.receive_text()  # agent sends transcript text
            text = data.strip()
            if not text:
                continue

            # run the emotion classifier
            try:
                res = emotion_pipe(text, truncation=True, max_length=256)
                # res could be a list like [{'label': 'anger', 'score': 0.9}]
                if isinstance(res, list) and len(res) > 0:
                    label = res[0]["label"]
                    score = res[0]["score"]
                else:
                    label = "neutral"
                    score = 0.0
            except Exception:
                label = "unknown"
                score = 0.0

            # decide threshold for alerting (tune per environment)
            if label in ["anger", "fear", "sadness"] and score > 0.6:
                alert = make_alert(agent_id, text, label, score)
                # broadcast to all supervisors
                dead = []
                for sup in supervisors:
                    try:
                        await sup.send_text(json.dumps(alert))
                    except Exception:
                        dead.append(sup)
                for d in dead:
                    supervisors.remove(d)
            # optionally also send lightweight transcript broadcast (non-alert)
            # we send a small transcript message so supervisors can see live text
            transcript_msg = {
                "type": "transcript",
                "agentId": agent_id,
                "text": text,
                "emotion": label,
                "score": float(score),
                "ts": int(time.time() * 1000)
            }
            dead = []
            for sup in supervisors:
                try:
                    await sup.send_text(json.dumps(transcript_msg))
                except Exception:
                    dead.append(sup)
            for d in dead:
                supervisors.remove(d)

    except WebSocketDisconnect:
        # agent disconnected
        return

@app.websocket("/ws/supervisor")
async def ws_supervisor(websocket: WebSocket):
    """
    Supervisors connect here and will receive JSON messages:
     - type: 'transcript'  -> normal transcript with emotion metadata
     - type: 'alert'       -> highlighted alert requiring attention
    """
    await websocket.accept()
    supervisors.append(websocket)
    try:
        while True:
            # keep the connection alive; supervisors might send pings or commands later
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in supervisors:
            supervisors.remove(websocket)
        return

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

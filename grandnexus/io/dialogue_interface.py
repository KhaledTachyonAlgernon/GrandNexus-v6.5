# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 62453-62646
# grandnexus/io/dialogue_interface.py
# ─────────────────────────────────────────────────────────────────────────
"""
DialogueInterface — REST + WebSocket Gateway for GrandNexus
===========================================================

Dépendances :
    pip install fastapi uvicorn[standard] pydantic sse-starlette
    # audio (optionnels) :
    pip install openai-whisper TTS
"""

from __future__ import annotations

import logging, asyncio, uuid, time, json, os, tempfile, threading
from dataclasses import dataclass
from typing import Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
import uvicorn

from grandnexus.core.nexus_core import NexusCore
from grandnexus.cognition.cognition_manager import CognitiveTask   # type: ignore

# ─────────────────────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class DIConfig:
    host: str = "0.0.0.0"
    port: int = 8089
    ws_route: str = "/ws"
    whisper_enabled: bool = False
    tts_enabled: bool = False
    system_prompt: str = "You are GrandNexus."
    max_pending: int = 100


# ─────────────────────────────────────────────────────────────────────────
# 2. DialogueInterface
# ─────────────────────────────────────────────────────────────────────────
class DialogueInterface:
    MODULE_NAME = "dialogue_interface"
    MSG_OUT = "dialogue_response"

    def __init__(self, nexus: NexusCore, cfg: Optional[DIConfig] = None):
        self.logger = logging.getLogger("GrandNexus.IO.DialogueIF")
        self.nexus = nexus
        self.cfg = cfg or DIConfig()

        # FastAPI app
        self.app = FastAPI(title="GrandNexus Dialogue API")

        # Pending responses {user_id: asyncio.Queue}
        self._queues: Dict[str, asyncio.Queue] = {}

        # Runtime
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Optional ASR / TTS
        self._whisper = None
        if self.cfg.whisper_enabled:
            import whisper  # lazy
            self._whisper = whisper.load_model("base")

        self._tts = None
        if self.cfg.tts_enabled:
            from TTS.api import TTS
            self._tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=False)

        self._configure_routes()

    # ─────────────────────────────────────────────────────────────────────
    # Routes
    # ─────────────────────────────────────────────────────────────────────
    def _configure_routes(self):
        @self.app.post("/chat")
        async def chat(input: Dict[str, Any]):
            user_id = input.get("user_id", "anon")
            text = input.get("text")
            meta = input.get("meta", {})
            if not text:
                return JSONResponse({"error": "text required"}, status_code=400)
            self._enqueue_task(user_id, text, meta)
            # Stream response as SSE
            return EventSourceResponse(self._iter_events(user_id))

        @self.app.websocket(self.cfg.ws_route)
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()
            user_id = str(uuid.uuid4())[:8]
            q = asyncio.Queue(maxsize=self.cfg.max_pending)
            self._queues[user_id] = q
            try:
                while True:
                    data = await ws.receive_text()
                    self._enqueue_task(user_id, data, {})
                    # relay answers
                    resp = await q.get()
                    await ws.send_text(json.dumps(resp))
            except WebSocketDisconnect:
                pass
            finally:
                self._queues.pop(user_id, None)

        @self.app.post("/speech")
        async def speech(file: UploadFile = File(...)):
            if not self._whisper:
                return JSONResponse({"error": "Whisper disabled"}, status_code=403)
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(await file.read())
            text = self._whisper.transcribe(tmp.name)["text"]
            os.remove(tmp.name)
            user_id = str(uuid.uuid4())[:8]
            self._enqueue_task(user_id, text, {"asr": True})
            return EventSourceResponse(self._iter_events(user_id))

    # ─────────────────────────────────────────────────────────────────────
    # Helper : SSE generator
    # ─────────────────────────────────────────────────────────────────────
    async def _iter_events(self, user_id: str):
        q = self._queues.setdefault(user_id, asyncio.Queue(maxsize=self.cfg.max_pending))
        while True:
            data = await q.get()
            yield {"event": "message", "data": json.dumps(data)}
            if data.get("done"): break

    # ─────────────────────────────────────────────────────────────────────
    # Enqueue Cognition task
    # ─────────────────────────────────────────────────────────────────────
    def _enqueue_task(self, user_id: str, text: str, meta: Dict[str, Any]):
        task = CognitiveTask(
            task_id=str(uuid.uuid4())[:8],
            task_type="user_dialogue",
            parameters={"text": text, "user_id": user_id, **meta},
            priority=0.4,
        )
        self.nexus.send_message(
            source=self.MODULE_NAME,
            target="executive_planner",
            message_type="cognitive_task",
            content=task.__dict__,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────
    def start(self):
        if self._running: return True
        self._running = True
        self.nexus.register_module(name=self.MODULE_NAME, module=self, dependencies=["executive_planner"])
        self._thread = threading.Thread(target=self._run_uvicorn, daemon=True)
        self._thread.start()
        self.logger.info(f"DialogueInterface listening on {self.cfg.host}:{self.cfg.port}")
        return True

    def _run_uvicorn(self):
        uvicorn.run(self.app, host=self.cfg.host, port=self.cfg.port, log_level="warning")

    def stop(self):
        self._running = False
        # Uvicorn shutdown handled by signal; for brevity skipped here.

    # ─────────────────────────────────────────────────────────────────────
    # Receive answers from system
    # ─────────────────────────────────────────────────────────────────────
    def handle_message(self, msg: Dict[str, Any]):
        if msg.get("type") != self.MSG_OUT: return
        user_id = msg["content"].get("user_id")
        if not user_id: return
        queue = self._queues.get(user_id)
        if not queue: return
        asyncio.run_coroutine_threadsafe(queue.put(msg["content"]), asyncio.get_event_loop())

    # ─────────────────────────────────────────────────────────────────────
    # Health / status
    # ─────────────────────────────────────────────────────────────────────
    def health_check(self) -> Dict[str, Any]:
        return {"queues": len(self._queues), "healthy": True}

    def get_status(self) -> Dict[str, Any]:
        return {"active_users": len(self._queues)}

# curl -N -H "Content-Type: application/json" \
#      -X POST --data '{"text":"Hello GrandNexus!"}' \
#      http://localhost:8089/chat
#
# const ev=new EventSource("http://localhost:8089/chat",{headers:{},method:"POST",
#  body:JSON.stringify({text:"Plan a route to the Moon"})});
# ev.onmessage=e=>console.log(JSON.parse(e.data));
#

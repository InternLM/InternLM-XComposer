import base64
import asyncio
import time
from typing import Dict, List, cast, get_args
from client import Client
from fastapi import APIRouter, WebSocket

router = APIRouter()

# 发送部署信号
@router.get('/deploy')
async def deploy_code():
    print('accept deploy sign')
    return 'ok'


@router.websocket("/chat")
async def stream_code(websocket: WebSocket):
    await websocket.accept()
    print("Incoming websocket connection...")
    data = await websocket.receive_json()
    print(data)
    if 'session_id' in data:
        session_id = data['session_id']
        print(session_id)
        await websocket.send_text("@@socket_ready")

        manager = Client(session_id)
        await manager.run(websocket)
import base64
import asyncio
import time
from typing import Dict, List, cast, get_args
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

        from main import app
        print(f'waiting for closing last main threading......')
        print(f'last main threading closed.')
        app.client.initiate(session_id)
        await app.client.run(websocket, session_id)

        app.client.stop_event.set()
        app.client.finished_closed = True
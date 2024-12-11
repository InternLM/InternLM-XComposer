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
        try:
            print(f'waiting for closing last main threading......')
            while not app.client.finished_closed:
                await websocket.send_text('waiting for closing last main threading......')
                await asyncio.sleep(0.5)
            app.client.finished_closed = False
            print(f'last main threading closed.')
            app.client.initiate(session_id)
            await app.client.run(websocket)

        except Exception as e:
            print(f'error type is {type(e)}')
            print(f'error is {e}')

        app.client.stop_event.set()
        app.client.vs_dict['break'] = True
        app.client.finished_closed = True
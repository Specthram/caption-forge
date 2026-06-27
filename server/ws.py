"""WebSocket endpoint streaming live job-queue events to the front-end.

On connect the client receives the current registry as one ``snapshot``
message, then one ``job`` message per subsequent update. The front-end
mirrors these into its jobs store (the sidebar Jobs button and the jobs
drawer).
"""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.jobs import manager

router = APIRouter()


@router.websocket("/ws/jobs")
async def jobs_socket(websocket: WebSocket) -> None:
    """Stream the initial job list then every job update to one client."""
    await websocket.accept()
    await websocket.send_json(
        {"kind": "snapshot", "jobs": manager.list_jobs()}
    )
    queue = manager.subscribe()
    try:
        while True:
            snapshot = await queue.get()
            await websocket.send_json({"kind": "job", "job": snapshot})
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        manager.unsubscribe(queue)

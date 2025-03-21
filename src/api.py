import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from .realtime_pipeline import RealTimeDataPipeline
from .prediction import predict_trajectory

logger = logging.getLogger("SatelliteWazePro")

def create_app() -> FastAPI:
    app = FastAPI(title="SatelliteWaze Pro API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.websocket("/ws/trajectory/{satellite_id}")
    async def websocket_trajectory(websocket: WebSocket, satellite_id: int):
        """
        Endpoint WebSocket para transmitir trayectorias en tiempo real.
        """
        await websocket.accept()
        logger.info("Cliente conectado para el satélite %s", satellite_id)
        rtp = RealTimeDataPipeline()
        try:
            tle_generator = rtp.stream_tle_data()
            while True:
                tle_data = await asyncio.to_thread(next, tle_generator)
                prediction = predict_trajectory(tle_data)
                await websocket.send_json(prediction)
        except WebSocketDisconnect:
            logger.info("Cliente desconectado del satélite %s", satellite_id)
        except Exception as e:
            logger.error("Error en el WebSocket: %s", str(e))
            await websocket.close(code=1011)

    return app

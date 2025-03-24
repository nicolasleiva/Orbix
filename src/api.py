import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .realtime_pipeline import RealTimeDataPipeline
from .prediction import predict_trajectory
from .ssc_api import SSCApi
from .spacex_api import SpaceXApi

logger = logging.getLogger("Orbix")

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
    
    @app.get("/ssc/satellites")
    async def get_available_satellites():
        """
        Endpoint para obtener la lista de satélites disponibles en la API SSC.
        """
        try:
            ssc_api = SSCApi()
            satellites = await asyncio.to_thread(ssc_api.get_available_satellites)
            return {"satellites": satellites}
        except Exception as e:
            logger.error("Error al obtener satélites disponibles: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/ssc/satellite/{satellite_id}/data")
    async def get_satellite_data(
        satellite_id: str,
        days: Optional[int] = Query(1, description="Número de días hacia atrás para obtener datos"),
        resolution: Optional[int] = Query(1, description="Factor de resolución (1=máxima resolución)")
    ):
        """
        Endpoint para obtener datos de un satélite específico de la API SSC.
        """
        try:
            ssc_api = SSCApi()
            now = datetime.now()
            start_time = now - timedelta(days=days)
            
            satellite_data = await asyncio.to_thread(
                ssc_api.get_satellite_data,
                satellites=[satellite_id],
                start_time=start_time,
                end_time=now,
                resolution_factor=resolution
            )
            
            if "error" in satellite_data:
                raise HTTPException(status_code=500, detail=satellite_data["error"])
                
            return satellite_data
        except Exception as e:
            logger.error("Error al obtener datos del satélite %s: %s", satellite_id, str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/spacex/launches")
    async def get_spacex_launches(
        limit: Optional[int] = Query(None, description="Número máximo de lanzamientos a obtener"),
        offset: Optional[int] = Query(None, description="Número de lanzamientos a saltar")
    ):
        """
        Endpoint para obtener información sobre lanzamientos de SpaceX.
        """
        try:
            spacex_api = SpaceXApi()
            launches = await asyncio.to_thread(spacex_api.get_launches, limit=limit, offset=offset)
            if "error" in launches:
                raise HTTPException(status_code=500, detail=launches["error"])
            return launches
        except Exception as e:
            logger.error("Error al obtener lanzamientos de SpaceX: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/spacex/launches/{flight_number}")
    async def get_spacex_launch(flight_number: int):
        """
        Endpoint para obtener información sobre un lanzamiento específico de SpaceX.
        """
        try:
            spacex_api = SpaceXApi()
            launch = await asyncio.to_thread(spacex_api.get_launch, flight_number=flight_number)
            if "error" in launch:
                raise HTTPException(status_code=500, detail=launch["error"])
            return launch
        except Exception as e:
            logger.error("Error al obtener lanzamiento %s de SpaceX: %s", flight_number, str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/spacex/rockets")
    async def get_spacex_rockets():
        """
        Endpoint para obtener información sobre cohetes de SpaceX.
        """
        try:
            spacex_api = SpaceXApi()
            rockets = await asyncio.to_thread(spacex_api.get_rockets)
            if "error" in rockets:
                raise HTTPException(status_code=500, detail=rockets["error"])
            return rockets
        except Exception as e:
            logger.error("Error al obtener cohetes de SpaceX: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/spacex/rockets/{rocket_id}")
    async def get_spacex_rocket(rocket_id: str):
        """
        Endpoint para obtener información sobre un cohete específico de SpaceX.
        """
        try:
            spacex_api = SpaceXApi()
            rocket = await asyncio.to_thread(spacex_api.get_rocket, rocket_id=rocket_id)
            if "error" in rocket:
                raise HTTPException(status_code=500, detail=rocket["error"])
            return rocket
        except Exception as e:
            logger.error("Error al obtener cohete %s de SpaceX: %s", rocket_id, str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/spacex/capsules")
    async def get_spacex_capsules():
        """
        Endpoint para obtener información sobre cápsulas de SpaceX.
        """
        try:
            spacex_api = SpaceXApi()
            capsules = await asyncio.to_thread(spacex_api.get_capsules)
            if "error" in capsules:
                raise HTTPException(status_code=500, detail=capsules["error"])
            return capsules
        except Exception as e:
            logger.error("Error al obtener cápsulas de SpaceX: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/spacex/capsules/{capsule_serial}")
    async def get_spacex_capsule(capsule_serial: str):
        """
        Endpoint para obtener información sobre una cápsula específica de SpaceX.
        """
        try:
            spacex_api = SpaceXApi()
            capsule = await asyncio.to_thread(spacex_api.get_capsule, capsule_serial=capsule_serial)
            if "error" in capsule:
                raise HTTPException(status_code=500, detail=capsule["error"])
            return capsule
        except Exception as e:
            logger.error("Error al obtener cápsula %s de SpaceX: %s", capsule_serial, str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    return app

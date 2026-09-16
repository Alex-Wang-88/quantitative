from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .domain import RunMode, jsonable
from .runtime import RuntimeService
from .terminal_feed import TerminalFeedStore


class ModeRequest(BaseModel):
    mode: RunMode


class ModifyOrderRequest(BaseModel):
    quantity: int = Field(gt=0)


class ExternalFillRequest(BaseModel):
    symbol: str
    side: str
    quantity: int = Field(gt=0)
    price: float = Field(gt=0)
    executed_at: str | None = None
    note: str = ""


class TerminalQuoteRequest(BaseModel):
    """Market-data-only payload emitted by the Eastmoney terminal adapter."""

    symbol: str
    timestamp: datetime
    last: float = Field(gt=0)
    previous_close: float | None = Field(default=None, ge=0)
    bid1: float | None = Field(default=None, ge=0)
    ask1: float | None = Field(default=None, ge=0)
    bid1_volume: int = Field(default=0, ge=0)
    ask1_volume: int = Field(default=0, ge=0)
    bar_volume: int = Field(default=0, ge=0)
    upper_limit: float | None = Field(default=None, ge=0)
    lower_limit: float | None = Field(default=None, ge=0)
    source: str = "EASTMONEY_TERMINAL_ADAPTER"
    frequency: str = "snapshot"


class TerminalQuoteBatch(BaseModel):
    quotes: list[TerminalQuoteRequest] = Field(min_length=1, max_length=1000)


runtime = RuntimeService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await runtime.stop()


app = FastAPI(title="A股量化纸盘控制台 API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"
if (FRONTEND_DIST / "assets").is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIST / "assets")),
        name="frontend-assets",
    )


@app.get("/", tags=["system"])
def root() -> Any:
    if FRONTEND_INDEX.is_file():
        return FileResponse(FRONTEND_INDEX)
    return {"name": "A股量化纸盘控制台", "api": "/api/v1", "paper_only": True}


@app.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    runtime.refresh_from_ledger()
    return {"ok": True, "paper_only": True, "status": runtime.status()}


@app.get("/status", tags=["system"])
def root_status() -> dict[str, Any]:
    runtime.refresh_from_ledger()
    return runtime.status()


@app.get("/api/v1/status", tags=["runtime"])
def get_status() -> dict[str, Any]:
    runtime.refresh_from_ledger()
    return runtime.status()


@app.get("/api/v1/dashboard", tags=["dashboard"])
def get_dashboard() -> dict[str, Any]:
    return runtime.dashboard()


def _terminal_feed_store() -> TerminalFeedStore:
    provider = getattr(runtime, "provider", None)
    store = getattr(provider, "terminal_feed_store", None)
    if isinstance(store, TerminalFeedStore):
        return store
    return TerminalFeedStore()


@app.get("/api/v1/integrations/eastmoney/subscription", tags=["integrations"])
def eastmoney_subscription(limit: int = 500) -> dict[str, Any]:
    """Return a bounded, paper-only symbol list for the terminal adapter."""

    runtime.refresh_from_ledger()
    cap = max(1, min(int(limit), 500))
    symbols: list[str] = []
    seen: set[str] = set()

    def add(symbol: str) -> None:
        value = str(symbol).strip()
        security = runtime.securities.get(value)
        if not security or value in seen or len(symbols) >= cap:
            return
        seen.add(value)
        exchange = "SHSE" if security.exchange == "SSE" else "SZSE"
        symbols.append(f"{exchange}.{value}")

    for symbol in runtime.broker.positions:
        add(symbol)
    for order in runtime.broker.orders.values():
        if order.status.value in {"DRAFT", "CONFIRMED", "SUBMITTED", "PARTIAL_FILLED"}:
            add(order.symbol)
    for signal in runtime.signals:
        add(signal.symbol)
    for symbol in sorted(runtime.securities):
        if len(symbols) >= min(cap, 30):
            break
        add(symbol)

    return {
        "paper_only": True,
        "symbols": symbols,
        "count": len(symbols),
        "limit": cap,
        "updated_at": datetime.now().astimezone().isoformat(),
    }


@app.get("/api/v1/integrations/eastmoney/feed", tags=["integrations"])
def eastmoney_feed_status() -> dict[str, Any]:
    return _terminal_feed_store().status(
        stale_after_seconds=int(getattr(runtime.config.data, "stale_after_seconds", 180))
    )


@app.post("/api/v1/integrations/eastmoney/quotes", tags=["integrations"])
def ingest_eastmoney_quotes(payload: TerminalQuoteBatch) -> dict[str, Any]:
    """Accept terminal market data; deliberately exposes no order operation."""

    accepted = _terminal_feed_store().write_rows(
        [item.model_dump(mode="json") for item in payload.quotes]
    )
    return {
        "accepted": accepted,
        "received": len(payload.quotes),
        "paper_only": True,
        "feed": _terminal_feed_store().status(
            stale_after_seconds=int(getattr(runtime.config.data, "stale_after_seconds", 180))
        ),
    }


@app.post("/api/v1/runner/start", tags=["runtime"])
async def start_runner() -> dict[str, Any]:
    return await runtime.start()


@app.post("/api/v1/runner/stop", tags=["runtime"])
async def stop_runner() -> dict[str, Any]:
    return await runtime.stop()


@app.post("/api/v1/runner/tick", tags=["runtime"])
async def tick_runner() -> dict[str, Any]:
    return await runtime.tick()


@app.post("/api/v1/runner/mode", tags=["runtime"])
def set_runner_mode(payload: ModeRequest) -> dict[str, Any]:
    return runtime.set_mode(payload.mode)


@app.post("/api/v1/rebalance", tags=["research"])
def rebalance() -> dict[str, Any]:
    try:
        return runtime.rebalance()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/model/train", tags=["research"])
def train_model() -> dict[str, Any]:
    try:
        return runtime.train_local_model()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/reports/run", tags=["reports"])
def generate_report() -> dict[str, Any]:
    try:
        return runtime.generate_report()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/reports", tags=["reports"])
def list_reports() -> list[dict[str, Any]]:
    return runtime.ledger.list_reports()


@app.get("/api/v1/events", tags=["audit"])
def list_events(limit: int = 50) -> list[dict[str, Any]]:
    return runtime.events(max(1, min(limit, 500)))


@app.post("/api/v1/orders/{order_id}/confirm", tags=["orders"])
def confirm_order(order_id: str) -> dict[str, Any]:
    try:
        return jsonable(runtime.confirm_order(order_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="order not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/orders/{order_id}/reject", tags=["orders"])
def reject_order(order_id: str) -> dict[str, Any]:
    try:
        return jsonable(runtime.reject_order(order_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="order not found") from exc


@app.post("/api/v1/orders/{order_id}/cancel", tags=["orders"])
def cancel_order(order_id: str) -> dict[str, Any]:
    try:
        return jsonable(runtime.cancel_order(order_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="order not found") from exc


@app.post("/api/v1/orders/{order_id}/modify", tags=["orders"])
def modify_order(order_id: str, payload: ModifyOrderRequest) -> dict[str, Any]:
    try:
        return jsonable(runtime.modify_order(order_id, payload.quantity))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="order not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/manual-records", tags=["manual"])
def record_external_fill(payload: ExternalFillRequest) -> dict[str, Any]:
    return runtime.record_external_fill(payload.model_dump())


@app.websocket("/api/v1/stream")
async def dashboard_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(runtime.dashboard())
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        return

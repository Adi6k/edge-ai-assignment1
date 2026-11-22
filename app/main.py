"""
app/main.py -- STARTER battery API. Look for TODO.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

GET  /health          -> {"status": "ok", "team": 7, "params": 1281, "val_mae": 3.4}
GET  /model_info      -> the per-layer table: params, bytes, MACs, and whether it fits the doorbell
POST /predict         one reading  -> {"days_left": 52.1, "timings_ms": {...}}
POST /predict_batch   {"rows": [reading, ...]} -> {"days_left": [...], "timings_ms": {...}}
A reading that is out of range gets HTTP 422 with the reason, never a guess.
"""
import json
import time

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from app.model import FEATURES, BatteryNet

cfg = json.load(open("model/config.json"))
net = BatteryNet(tuple(cfg["hidden"]))
net.load_state_dict(torch.load("model/model.pt", weights_only=True))
net.eval()
torch.set_num_threads(1)
MU, SD = np.array(cfg["mu"], np.float32), np.array(cfg["sd"], np.float32)
N_PARAMS = sum(p.numel() for p in net.parameters())


class Reading(BaseModel):
    # Mission 3: physical sensor limits (not the data's min/max). allow_inf_nan=False refuses NaN/inf.
    battery_v: float = Field(ge=2.5, le=4.5, allow_inf_nan=False)
    temp_c: float = Field(ge=-40, le=85, allow_inf_nan=False)
    rings_per_day: float = Field(ge=0, le=1000, allow_inf_nan=False)
    wifi_rssi_dbm: float = Field(ge=-120, le=-10, allow_inf_nan=False)
    age_months: float = Field(ge=0, le=240, allow_inf_nan=False)


class Batch(BaseModel):
    rows: list[Reading] = Field(min_length=1, max_length=10000)


app = FastAPI(title="Doorbell battery API")
ms = lambda t0: round((time.perf_counter() - t0) * 1000, 3)


@app.get("/model_info")
def model_info():
    linears = [m for m in net.net if isinstance(m, torch.nn.Linear)]
    layers = []
    for i, m in enumerate(linears, 1):
        p = sum(q.numel() for q in m.parameters())
        layers.append({"name": f"linear{i}", "in": m.in_features, "out": m.out_features,
                       "params": p, "bytes_f32": p * 4, "macs": m.in_features * m.out_features})
    tp = sum(l["params"] for l in layers)
    return {"layers": layers, "total_params": tp, "total_bytes_f32": tp * 4,
            "total_bytes_int8": tp, "total_macs": sum(l["macs"] for l in layers),
            "fits_300kb_sram_f32": tp * 4 <= 300 * 1024, "fits_300kb_sram_int8": tp <= 300 * 1024}


@app.get("/health")
def health():
    return {"status": "ok", "team": cfg["team"], "params": N_PARAMS, "val_mae": cfg["val_mae"]}


async def run(request: Request, schema):
    t_all = time.perf_counter(); t = {}
    t0 = time.perf_counter()
    try:
        data = schema.model_validate_json(await request.body())
    except ValidationError as e:
        return JSONResponse(status_code=422, content={"error": "bad reading", "detail": json.loads(e.json(include_input=False, include_url=False))})
    t["parse"] = ms(t0)
    rows = data.rows if isinstance(data, Batch) else [data]
    t0 = time.perf_counter()
    x = (np.array([[getattr(r, f) for f in FEATURES] for r in rows], np.float32) - MU) / SD
    t["normalise"] = ms(t0)
    t0 = time.perf_counter()
    with torch.no_grad():
        y = net(torch.from_numpy(x)).numpy()
    t["inference"] = ms(t0)
    t0 = time.perf_counter()
    days = [round(float(v), 2) for v in np.maximum(y, 0)]
    body = {"days_left": days if isinstance(data, Batch) else days[0]}
    t["build"] = ms(t0)
    t["server_total"] = ms(t_all)
    body["timings_ms"] = t
    return body


@app.post("/predict")
async def predict(request: Request):
    return await run(request, Reading)


@app.post("/predict_batch")
async def predict_batch(request: Request):
    return await run(request, Batch)

# EcoLogistics — Carbon Tracker Service (Microservicio de Huella de Carbono)

## Objetivo
Construir un microservicio API que calcule emisiones estimadas de CO₂ para operaciones logísticas con variables:

- **Tipo de vehículo:** `ELECTRIC`, `DIESEL`, `HYBRID`
- **Peso de carga:** toneladas
- **Distancia:** km
- **Factor de eficiencia (energía/combustible):** ajusta el consumo real (**más alto = más eficiente**)

---

## Fase 1 — Diseño y Definición de Prompts

### Prompt de contexto inicial
Eres un Desarrollador Senior y Arquitecto de Software. Vas a actuar como mi pair programmer.
Stack: Python 3.11, FastAPI, Pydantic v2, PyTest.
Objetivo: construir un microservicio "Carbon Tracker Service" modular, seguro y testeable.
Estándares:
- Clean Code, SOLID, separación de responsabilidades (API vs negocio).
- Validación estricta de entrada con Pydantic.
- Manejo robusto de errores con códigos HTTP claros.
- Código con typing, docstrings y estructura de carpetas.
- Tests unitarios con cobertura alta (casos normales y edge cases).
- Seguridad: evitar overflows, validar rangos, no confiar en input.
Reglas de respuesta:
1) Antes de escribir código, entrega un diseño (estructura, contratos, fórmula).
2) Luego entrega el código por archivos (rutas, servicios, modelos, tests).
3) Evita “mega-archivos”; prioriza módulos pequeños.

### Prompt de Chain-of-Thought
Propón la lógica del cálculo y los supuestos del modelo de emisiones.
Razona internamente paso a paso, pero entrégame solo:
- fórmula final,
- supuestos,
- validaciones,
- ejemplos de entrada/salida,
- y estructura de módulos.
No escribas código todavía.

---

## Fase 2 — Codificación

### Prompt para generar la función principal
Genera la función principal calculate_emissions(request) con:
- inputs: vehicle_type, load_tons, distance_km, efficiency_factor
- output: emissions_kg, emissions_tons, ton_km, factor_used
Incluye validaciones (no negativos, rangos razonables) y errores claros.
No construyas aún la API; solo el dominio (servicio).
### Refinamiento de la función
Refina la función para:
- usar tipos Enum para vehicle_type,
- validar límites máximos (p.ej. distancia <= 200000 km, carga <= 1000 tons),
- manejar efficiency_factor mínimo (>= 0.1),
- lanzar excepciones de dominio (DomainValidationError) en vez de ValueError.
### Prompt para scaffolding
Propón una estructura de proyecto FastAPI que separe:
- routers (API)
- schemas (Pydantic)
- services (negocio)
- domain/errors
- config
Incluye ejemplos de archivos y cómo se conectan.

---

## Fase 3 — Código completo

### Estructura de carpetas

```ecologistics-carbon-tracker/
  app/
    __init__.py
    main.py
    api/
      __init__.py
      routes.py
    schemas/
      __init__.py
      carbon.py
    services/
      __init__.py
      carbon_calculator.py
      emission_factors.py
    domain/
      __init__.py
      errors.py
      types.py
  tests/
    test_carbon_calculator.py
    test_api.py
  pyproject.toml  (opcional)
  README.md
```

### Código Fuente

```
app/domain/types.py
from enum import Enum

class VehicleType(str, Enum):
    ELECTRIC = "ELECTRIC"
    DIESEL = "DIESEL"
    HYBRID = "HYBRID"
```
```
app/domain/errors.py
class DomainValidationError(Exception):
    """Raised when domain/business rules are violated.""
```
```
app/services/emission_factors.py
from app.domain.types import VehicleType

# Configurable defaults (placeholders). Replace with your company/standard factors.
DEFAULT_FACTORS_KGCO2_PER_TON_KM: dict[VehicleType, float] = {
    VehicleType.DIESEL: 0.12,
    VehicleType.HYBRID: 0.08,
    VehicleType.ELECTRIC: 0.02,
}
```
```
app/services/carbon_calculator.py
from dataclasses import dataclass

from app.domain.errors import DomainValidationError
from app.domain.types import VehicleType
from app.services.emission_factors import DEFAULT_FACTORS_KGCO2_PER_TON_KM


@dataclass(frozen=True)
class CarbonResult:
    vehicle_type: VehicleType
    load_tons: float
    distance_km: float
    efficiency_factor: float
    ton_km: float
    factor_used_kgco2_per_ton_km: float
    emissions_kgco2: float
    emissions_tco2: float


def calculate_emissions(
    *,
    vehicle_type: VehicleType,
    load_tons: float,
    distance_km: float,
    efficiency_factor: float,
    factor_override_kgco2_per_ton_km: float | None = None,
) -> CarbonResult:
    """
    Estimate CO2 emissions based on ton-km and an emission factor per vehicle type.

    emissions_kg = (load_tons * distance_km) * factor(vehicle) / efficiency_factor
    """
    # Domain limits (security + sanity). Tune based on business.
    if load_tons < 0:
        raise DomainValidationError("load_tons must be >= 0")
    if distance_km < 0:
        raise DomainValidationError("distance_km must be >= 0")
    if efficiency_factor < 0.1:
        raise DomainValidationError("efficiency_factor must be >= 0.1")

    if load_tons > 1000:
        raise DomainValidationError("load_tons exceeds max allowed (1000)")
    if distance_km > 200_000:
        raise DomainValidationError("distance_km exceeds max allowed (200000)")

    ton_km = load_tons * distance_km

    if factor_override_kgco2_per_ton_km is not None:
        if factor_override_kgco2_per_ton_km <= 0:
            raise DomainValidationError("factor_override_kgco2_per_ton_km must be > 0")
        factor_used = factor_override_kgco2_per_ton_km
    else:
        factor_used = DEFAULT_FACTORS_KGCO2_PER_TON_KM[vehicle_type]

    emissions_kg = (ton_km * factor_used) / efficiency_factor
    emissions_t = emissions_kg / 1000.0

    return CarbonResult(
        vehicle_type=vehicle_type,
        load_tons=load_tons,
        distance_km=distance_km,
        efficiency_factor=efficiency_factor,
        ton_km=ton_km,
        factor_used_kgco2_per_ton_km=factor_used,
        emissions_kgco2=emissions_kg,
        emissions_tco2=emissions_t,
    )
  ```
```
app/schemas/carbon.py
from pydantic import BaseModel, Field

from app.domain.types import VehicleType


class CarbonRequest(BaseModel):
    vehicle_type: VehicleType
    load_tons: float = Field(ge=0, le=1000)
    distance_km: float = Field(ge=0, le=200_000)
    efficiency_factor: float = Field(ge=0.1, le=10.0)

    # Optional: allow overriding factor for experiments / future config.
    factor_override_kgco2_per_ton_km: float | None = Field(default=None, gt=0)


class CarbonResponse(BaseModel):
    vehicle_type: VehicleType
    load_tons: float
    distance_km: float
    efficiency_factor: float
    ton_km: float
    factor_used_kgco2_per_ton_km: float
    emissions_kgco2: float
    emissions_tco2: float
 ```
 ```
app/api/routes.py
from fastapi import APIRouter, HTTPException

from app.domain.errors import DomainValidationError
from app.schemas.carbon import CarbonRequest, CarbonResponse
from app.services.carbon_calculator import calculate_emissions

router = APIRouter(prefix="/v1", tags=["carbon"])


@router.post("/carbon/calculate", response_model=CarbonResponse)
def calculate_carbon(req: CarbonRequest) -> CarbonResponse:
    try:
        result = calculate_emissions(
            vehicle_type=req.vehicle_type,
            load_tons=req.load_tons,
            distance_km=req.distance_km,
            efficiency_factor=req.efficiency_factor,
            factor_override_kgco2_per_ton_km=req.factor_override_kgco2_per_ton_km,
        )
    except DomainValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return CarbonResponse(
        vehicle_type=result.vehicle_type,
        load_tons=result.load_tons,
        distance_km=result.distance_km,
        efficiency_factor=result.efficiency_factor,
        ton_km=result.ton_km,
        factor_used_kgco2_per_ton_km=result.factor_used_kgco2_per_ton_km,
        emissions_kgco2=result.emissions_kgco2,
        emissions_tco2=result.emissions_tco2,
    )
```
```
app/main.py
from fastapi import FastAPI

from app.api.routes import router as carbon_router

app = FastAPI(
    title="EcoLogistics Carbon Tracker Service",
    version="1.0.0",
)

app.include_router(carbon_router)


@app.get("/health")
def health():
    return {"status": "ok"}
```

---


## Fase 4 — Pruebas

### Prompt para generar tests unitarios
Genera una suite de tests con PyTest que cubra:
- casos normales por tipo de vehículo
- distancia cero
- carga negativa (error)
- efficiency_factor < 0.1 (error)
- tipo de vehículo no soportado (validación)
Incluye tests del servicio (puro) y tests del endpoint FastAPI.

```
tests/test_carbon_calculator.py
import pytest

from app.domain.errors import DomainValidationError
from app.domain.types import VehicleType
from app.services.carbon_calculator import calculate_emissions


def test_calculate_emissions_diesel_basic():
    r = calculate_emissions(
        vehicle_type=VehicleType.DIESEL,
        load_tons=10,
        distance_km=100,
        efficiency_factor=1.0,
    )
    # ton_km = 1000; diesel factor=0.12 => 120 kg
    assert r.ton_km == 1000
    assert r.emissions_kgco2 == pytest.approx(120.0, rel=1e-9)


def test_distance_zero_is_zero_emissions():
    r = calculate_emissions(
        vehicle_type=VehicleType.HYBRID,
        load_tons=10,
        distance_km=0,
        efficiency_factor=1.0,
    )
    assert r.ton_km == 0
    assert r.emissions_kgco2 == 0


def test_negative_load_raises():
    with pytest.raises(DomainValidationError):
        calculate_emissions(
            vehicle_type=VehicleType.DIESEL,
            load_tons=-1,
            distance_km=10,
            efficiency_factor=1.0,
        )


def test_efficiency_too_low_raises():
    with pytest.raises(DomainValidationError):
        calculate_emissions(
            vehicle_type=VehicleType.ELECTRIC,
            load_tons=1,
            distance_km=10,
            efficiency_factor=0.05,
        )


def test_efficiency_reduces_emissions():
    r1 = calculate_emissions(
        vehicle_type=VehicleType.DIESEL,
        load_tons=10,
        distance_km=100,
        efficiency_factor=1.0,
    )
    r2 = calculate_emissions(
        vehicle_type=VehicleType.DIESEL,
        load_tons=10,
        distance_km=100,
        efficiency_factor=2.0,
    )
    assert r2.emissions_kgco2 == pytest.approx(r1.emissions_kgco2 / 2.0, rel=1e-9)


def test_factor_override():
    r = calculate_emissions(
        vehicle_type=VehicleType.DIESEL,
        load_tons=10,
        distance_km=100,
        efficiency_factor=1.0,
        factor_override_kgco2_per_ton_km=0.5,
    )
    # ton_km=1000; 0.5 => 500 kg
    assert r.emissions_kgco2 == pytest.approx(500.0, rel=1e-9)
```
```
tests/test_api.py
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_calculate_ok():
    payload = {
        "vehicle_type": "DIESEL",
        "load_tons": 10,
        "distance_km": 100,
        "efficiency_factor": 1.0
    }
    r = client.post("/v1/carbon/calculate", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["ton_km"] == 1000
    assert abs(data["emissions_kgco2"] - 120.0) < 1e-9


def test_api_rejects_negative_load():
    payload = {
        "vehicle_type": "DIESEL",
        "load_tons": -1,
        "distance_km": 100,
        "efficiency_factor": 1.0
    }
    r = client.post("/v1/carbon/calculate", json=payload)
    # Pydantic blocks it before domain, typically 422
    assert r.status_code == 422


def test_api_rejects_unknown_vehicle_type():
    payload = {
        "vehicle_type": "GASOLINE",
        "load_tons": 1,
        "distance_km": 10,
        "efficiency_factor": 1.0
    }
    r = client.post("/v1/carbon/calculate", json=payload)
    assert r.status_code == 422
```

---

## Reflexión crítica

Usar un LLM como pair programmer acelera el diseño inicial, mejora la productividad y ayuda a generar rápidamente estructura modular y pruebas. El riesgo principal es confiar ciegamente en supuestos (por ejemplo, factores de emisión “placeholder”) o aceptar validaciones incompletas. En este caso, el LLM es muy útil para scaffolding y cobertura de edge cases, pero el equipo debe verificar la fórmula y factores con una fuente oficial, y hacer code review humano para seguridad (límites, abuso, observabilidad) antes de llevarlo a producción.



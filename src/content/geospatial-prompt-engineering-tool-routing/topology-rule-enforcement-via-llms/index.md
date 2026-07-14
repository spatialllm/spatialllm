# Topology Rule Enforcement via LLMs

Large language models excel at translating natural language into structured spatial queries, but they lack inherent geometric reasoning. When LLMs generate coordinates, polygons, or spatial relationships, they frequently violate fundamental topological constraints such as self-intersection, sliver gaps, duplicate vertices, or invalid ring orientations. **Topology Rule Enforcement via LLMs** addresses this gap by embedding deterministic validation layers between generative outputs and downstream spatial systems. Within the broader [Geospatial Prompt Engineering & Tool Routing](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/) paradigm, enforcing topology is not a post-processing afterthought but a mandatory routing checkpoint that guarantees spatial integrity before data enters analytical or operational pipelines.

<figure class="diagram">
<svg viewBox="0 0 860 300" role="img" aria-labelledby="tre-t tre-d" xmlns="http://www.w3.org/2000/svg">
  <title id="tre-t">Topology rule-enforcement pipeline</title>
  <desc id="tre-d">Geometry intake, rule definition, a validation pass, error detection, and repair with enforcement run in sequence to produce a topologically valid geometry set for pipelines, QA, and publishing.</desc>
  <defs>
    <marker id="tre-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" fill="#5b6471"/>
    </marker>
  </defs>
  <g fill="#e3f0f4" stroke="#1f6b8a" stroke-width="2">
    <rect x="15" y="55" width="150" height="80" rx="8"/>
    <rect x="183" y="55" width="150" height="80" rx="8"/>
    <rect x="351" y="55" width="150" height="80" rx="8"/>
    <rect x="519" y="55" width="150" height="80" rx="8"/>
    <rect x="687" y="55" width="150" height="80" rx="8"/>
  </g>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#tre-arrow)">
    <line x1="166" y1="95" x2="181" y2="95"/>
    <line x1="334" y1="95" x2="349" y2="95"/>
    <line x1="502" y1="95" x2="517" y2="95"/>
    <line x1="670" y1="95" x2="685" y2="95"/>
  </g>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#tre-arrow)">
    <line x1="90" y1="137" x2="90" y2="202"/>
    <line x1="258" y1="137" x2="258" y2="202"/>
    <line x1="426" y1="137" x2="426" y2="202"/>
    <line x1="594" y1="137" x2="594" y2="202"/>
    <line x1="762" y1="137" x2="762" y2="202"/>
  </g>
  <rect x="15" y="205" width="822" height="58" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/>
  <g fill="#1f2937" font-size="13" text-anchor="middle">
    <text x="90" y="90"><tspan x="90" dy="0">Geometry</tspan><tspan x="90" dy="16">intake</tspan></text>
    <text x="258" y="90"><tspan x="258" dy="0">Rule</tspan><tspan x="258" dy="16">definition</tspan></text>
    <text x="426" y="90"><tspan x="426" dy="0">Validation</tspan><tspan x="426" dy="16">pass</tspan></text>
    <text x="594" y="90"><tspan x="594" dy="0">Error</tspan><tspan x="594" dy="16">detection</tspan></text>
    <text x="762" y="90"><tspan x="762" dy="0">Repair</tspan><tspan x="762" dy="16">&amp; enforce</tspan></text>
  </g>
  <text x="426" y="240" fill="#1f2937" font-size="15" font-weight="600" text-anchor="middle">Topologically valid geometry set</text>
  <text x="426" y="287" fill="#5b6471" font-size="12" text-anchor="middle">Downstream: pipelines · QA · publishing</text>
</svg>
</figure>

## 1. Constraint-Driven Prompt Design & Schema Enforcement

The foundation of reliable spatial generation lies in explicitly constraining the LLM's output schema. Freeform geometry requests consistently produce topologically invalid payloads. Instead, system prompts must declare invariant rules using [OGC Simple Features](https://www.ogc.org/standards/sfs) terminology. Specify mandatory properties: closed linear rings, explicit coordinate reference systems (CRS), minimum vertex spacing, and non-overlapping interiors for adjacent features. This structured approach aligns directly with [Prompt-to-Spatial-SQL Generation](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/), where JSON or GeoJSON templates are paired with strict validation instructions.

By embedding topological preconditions into the prompt's system directive, you establish a machine-readable contract that the LLM must satisfy. The following Pydantic schema enforces GeoJSON compliance, CRS declaration, and coordinate bounds before any geometric validation occurs.

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal, List, Tuple
import json

class GeoJSONFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    crs: dict = Field(default_factory=lambda: {"type": "name", "properties": {"name": "EPSG:4326"}})
    geometry: dict
    properties: dict = Field(default_factory=dict)

    @field_validator("geometry")
    @classmethod
    def validate_geometry_structure(cls, v: dict) -> dict:
        valid_types = ("Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon")
        if v.get("type") not in valid_types:
            raise ValueError(f"Unsupported geometry type: {v.get('type')}")
        if "coordinates" not in v:
            raise ValueError("Missing 'coordinates' array in geometry")
        return v

    @field_validator("geometry")
    @classmethod
    def validate_coordinate_bounds(cls, v: dict) -> dict:
        """Validates WGS84 bounds for top-level coordinate pairs."""
        def check_bounds(c):
            if isinstance(c, (int, float)):
                return  # Scalars aren't coordinates by themselves
            if (isinstance(c, (list, tuple)) and len(c) >= 2 and
                    all(isinstance(x, (int, float)) for x in c[:2])):
                x, y = c[0], c[1]
                if not (-180 <= x <= 180 and -90 <= y <= 90):
                    raise ValueError(f"Coordinate ({x}, {y}) exceeds WGS84 bounds")
            elif isinstance(c, (list, tuple)):
                for sub in c:
                    check_bounds(sub)
        check_bounds(v.get("coordinates", []))
        return v

# Example LLM payload validation
llm_payload = {
    "type": "Feature",
    "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
    "geometry": {
        "type": "Polygon",
        "coordinates": [[[-74.0, 40.7], [-73.9, 40.7], [-73.9, 40.8], [-74.0, 40.8], [-74.0, 40.7]]]
    },
    "properties": {"name": "Manhattan Block"}
}

validated = GeoJSONFeature(**llm_payload)
print("Schema validation passed:", validated.geometry["type"])
```

## 2. Deterministic Engine Routing & Dispatch Architecture

Once the LLM returns a geometry payload, the pipeline must route it to the appropriate spatial engine for validation. Production systems typically employ a dual-layer strategy: lightweight in-memory validation via `shapely` for rapid iteration, and heavy-duty relational validation via PostGIS for enterprise-scale datasets. Implementing [GeoPandas & PostGIS Tool Routing](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/geopandas-postgis-tool-routing/) requires a dispatcher that evaluates payload complexity, coordinate count, and latency SLAs.

The routing layer should parse the LLM output, validate JSON/GeoJSON structure, and branch execution paths based on feature count and topology complexity. Small batches (<100 features) route to Python for immediate feedback, while multi-polygon networks or topology-heavy administrative boundaries trigger asynchronous PostGIS transactions. The dispatcher must also capture execution metadata, including validation status codes and geometry complexity metrics, to inform downstream retry logic.

```python
import shapely
from shapely.geometry import shape
from typing import Dict, Any, Tuple

def route_validation_engine(feature: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Routes geometry payload to appropriate validation engine based on complexity."""
    geom = shape(feature["geometry"])

    # Count coordinates across geometry types
    if hasattr(geom, "coords"):
        coord_count = len(list(geom.coords))
    elif hasattr(geom, "geoms"):
        coord_count = sum(
            len(list(part.exterior.coords)) if hasattr(part, "exterior") else len(list(part.coords))
            for part in geom.geoms
        )
    elif hasattr(geom, "exterior"):
        coord_count = len(list(geom.exterior.coords))
    else:
        coord_count = 1

    MAX_MEMORY_COORDS = 5000
    IS_POLYGON = geom.geom_type in ("Polygon", "MultiPolygon")

    metadata = {
        "geom_type": geom.geom_type,
        "coord_count": coord_count,
        "is_valid": geom.is_valid,
        "crs": feature.get("crs", {}).get("properties", {}).get("name", "unknown")
    }

    if coord_count >= MAX_MEMORY_COORDS and IS_POLYGON:
        return "postgis_async", metadata
    else:
        return "shapely_sync", metadata

# Dispatch example
route, meta = route_validation_engine(validated.model_dump())
print(f"Routed to: {route} | Complexity: {meta['coord_count']} coords")
```

## 3. Explicit Validation, CRS Normalization & Error Mapping

Validation must be deterministic and explicitly map failures to actionable error codes. LLMs frequently generate self-intersecting rings, unclosed polygons, or geometries with incorrect winding order. The validation layer must normalize CRS, enforce ring closure, and apply topological repair where safe. For detailed algorithmic strategies on handling these edge cases, see [Enforcing Topological Rules in LLM-Generated Geometries](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/enforcing-topological-rules-in-llm-generated-geometries/).

The following pipeline demonstrates explicit validation, CRS enforcement, and structured error mapping. It uses `shapely.make_valid` for safe repair and captures precise failure reasons for human-in-the-loop or automated retry workflows.

```python
from shapely.validation import make_valid
from shapely.geometry import shape, Polygon
from shapely.ops import transform
import pyproj
import shapely

class TopologyError(Exception):
    def __init__(self, code: str, message: str, geometry_hint: str):
        self.code = code
        self.message = message
        self.geometry_hint = geometry_hint
        super().__init__(self.message)

def enforce_topology(feature: Dict[str, Any]) -> Dict[str, Any]:
    """Validates, repairs, and normalizes LLM-generated geometry."""
    repaired = False
    try:
        geom = shape(feature["geometry"])

        # 1. CRS Enforcement: Ensure WGS84 or project to target
        target_crs = "EPSG:4326"
        declared_crs = feature.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
        if declared_crs != target_crs:
            project = pyproj.Transformer.from_crs(declared_crs, target_crs, always_xy=True).transform
            geom = transform(project, geom)
            feature["crs"] = {"type": "name", "properties": {"name": target_crs}}

        # 2. Topology Validation & Repair
        if not geom.is_valid:
            fixed = make_valid(geom)
            if not fixed.is_valid:
                raise TopologyError(
                    "ERR_TOPO_UNRECOVERABLE",
                    "Geometry contains unrecoverable topological defects.",
                    str(geom.wkt[:200])
                )
            geom = fixed
            repaired = True

        # 3. Ring Orientation & Closure Enforcement (OGC SFS)
        if geom.geom_type == "Polygon":
            if not geom.exterior.is_ring:
                raise TopologyError("ERR_RING_UNCLOSED", "Exterior ring is not closed.", str(geom.wkt[:200]))

        # 4. Sliver/Minimum Area Check
        if geom.area < 1e-10:
            raise TopologyError("ERR_SLIVER_GEOMETRY", "Geometry area below minimum threshold.", str(geom.wkt[:200]))

        # Update payload
        feature["geometry"] = shapely.to_geojson(geom)
        feature["_validation"] = {"status": "PASS", "engine": "shapely", "repaired": repaired}
        return feature

    except TopologyError as e:
        feature["_validation"] = {"status": "FAIL", "code": e.code, "message": e.message, "hint": e.geometry_hint}
        return feature

# Execute validation
result = enforce_topology(validated.model_dump())
print("Validation Result:", result["_validation"])
```

## 4. Pipeline Integration & Production Considerations

Topology rule enforcement must be treated as a synchronous gate in LLM-assisted geoprocessing workflows. When validation fails, the structured error payload should trigger one of three routing paths:
1. **Automated Retry:** Inject the error code and geometry hint back into the LLM context window with a corrective system directive (e.g., `"ERR_RING_UNCLOSED: Ensure first and last coordinates match exactly."`).
2. **Fallback Engine:** Route to PostGIS for `ST_IsValid` and `ST_MakeValid` execution, leveraging database-level topology rules for complex multi-geometry networks.
3. **Human Review Queue:** Flag payloads with `ERR_TOPO_UNRECOVERABLE` or `ERR_SLIVER_GEOMETRY` for spatial data scientist review, preserving the original LLM output alongside the validation trace.

Monitoring should track validation pass/fail rates, average repair latency, and CRS mismatch frequency. These metrics directly inform prompt refinement cycles and help calibrate temperature/sampling parameters for spatial generation tasks. By treating topology enforcement as a first-class routing checkpoint rather than a cleanup script, platform teams guarantee that LLM-generated spatial data remains interoperable, queryable, and compliant with enterprise GIS standards from ingestion to visualization.

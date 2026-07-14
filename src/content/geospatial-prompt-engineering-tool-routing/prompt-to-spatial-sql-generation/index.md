# Prompt-to-Spatial-SQL Generation

Prompt-to-Spatial-SQL Generation represents a deterministic workflow stage within modern geospatial AI architectures, bridging natural language intent with execution-safe spatial database operations. For AI/ML engineers, spatial data scientists, and platform teams, the primary challenge is not syntactic SQL generation, but producing spatially aware PostGIS statements that respect schema constraints, coordinate reference systems, and query performance boundaries. This capability operates as a foundational component of the [Geospatial Prompt Engineering & Tool Routing](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/) paradigm, where large language models function strictly as semantic translators rather than direct database executors. Production-grade implementations require explicit validation gates, structured error mapping, and deterministic fallback mechanisms to prevent unbounded spatial scans, topology violations, and silent data corruption.

<figure class="diagram">
<svg viewBox="0 0 860 300" role="img" aria-labelledby="psq-t psq-d" xmlns="http://www.w3.org/2000/svg">
  <title id="psq-t">Prompt-to-spatial-SQL pipeline</title>
  <desc id="psq-d">Natural language, schema grounding, SQL synthesis, spatial-function binding, and query validation run in sequence to emit a validated spatial SQL query for PostGIS routing, execution, and caching.</desc>
  <defs>
    <marker id="psq-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
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
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#psq-arrow)">
    <line x1="166" y1="95" x2="181" y2="95"/>
    <line x1="334" y1="95" x2="349" y2="95"/>
    <line x1="502" y1="95" x2="517" y2="95"/>
    <line x1="670" y1="95" x2="685" y2="95"/>
  </g>
  <g stroke="#5b6471" stroke-width="2" marker-end="url(#psq-arrow)">
    <line x1="90" y1="137" x2="90" y2="202"/>
    <line x1="258" y1="137" x2="258" y2="202"/>
    <line x1="426" y1="137" x2="426" y2="202"/>
    <line x1="594" y1="137" x2="594" y2="202"/>
    <line x1="762" y1="137" x2="762" y2="202"/>
  </g>
  <rect x="15" y="205" width="822" height="58" rx="8" fill="#fdf3e0" stroke="#c46a3d" stroke-width="2"/>
  <g fill="#1f2937" font-size="13" text-anchor="middle">
    <text x="90" y="90"><tspan x="90" dy="0">Natural</tspan><tspan x="90" dy="16">language</tspan></text>
    <text x="258" y="90"><tspan x="258" dy="0">Schema</tspan><tspan x="258" dy="16">grounding</tspan></text>
    <text x="426" y="90"><tspan x="426" dy="0">SQL</tspan><tspan x="426" dy="16">synthesis</tspan></text>
    <text x="594" y="90"><tspan x="594" dy="0">Spatial</tspan><tspan x="594" dy="16">functions</tspan></text>
    <text x="762" y="90"><tspan x="762" dy="0">Query</tspan><tspan x="762" dy="16">validation</tspan></text>
  </g>
  <text x="426" y="240" fill="#1f2937" font-size="15" font-weight="600" text-anchor="middle">Validated spatial SQL query</text>
  <text x="426" y="287" fill="#5b6471" font-size="12" text-anchor="middle">Downstream: PostGIS routing · execution · caching</text>
</svg>
</figure>

## Step 1: Schema Context Injection & Constrained Prompt Construction

The foundation of reliable Prompt-to-Spatial-SQL Generation begins with precise metadata injection. LLMs lack inherent awareness of spatial indexes, CRS definitions, or table partitioning strategies. You must extract and serialize the following into the system prompt:

1. **Table DDL & Geometry Metadata**: Column names, data types, `geometry`/`geography` types, SRIDs, and index definitions (`GIST` on spatial columns).
2. **Spatial Function Allowlist**: Explicitly permit only production-tested PostGIS functions (`ST_Intersects`, `ST_DWithin`, `ST_Transform`, `ST_Union`, `ST_Buffer`). Block expensive or unsafe operations (`ST_DumpPoints` on large tables, unbounded `ST_Distance`).
3. **CRS Alignment Rules**: Enforce `ST_Transform` requirements when joining across mismatched SRIDs.
4. **Few-Shot Spatial Patterns**: Provide examples demonstrating bounding box pre-filtering (`&&` operator) before precise geometry predicates, and proper use of `EXPLAIN` hints.

Constrain LLM output using JSON schema validation or regex extraction to isolate the raw SQL string. Strip markdown formatting, enforce lowercase SQL keywords, and require explicit `LIMIT` clauses for exploratory queries.

```python
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class SpatialSchemaContext(BaseModel):
    table_name: str
    geometry_column: str
    srid: int
    allowed_functions: List[str] = Field(default_factory=lambda: [
        "ST_Intersects", "ST_DWithin", "ST_Transform", "ST_Union",
        "ST_Buffer", "ST_Contains", "ST_Centroid"
    ])
    index_hints: List[str] = Field(default_factory=lambda: ["GIST"])

def build_constrained_prompt(schema: SpatialSchemaContext, user_query: str) -> str:
    """Constructs a deterministic system prompt with strict spatial guardrails."""
    system_prompt = f"""
You are a spatial SQL compiler. Your task is to translate natural language into valid PostGIS SQL.

SCHEMA CONTEXT:
- Table: {schema.table_name}
- Geometry Column: {schema.geometry_column} (SRID: {schema.srid})
- Allowed Spatial Functions: {', '.join(schema.allowed_functions)}
- Index Strategy: Bounding box pre-filtering (&&) is MANDATORY before any ST_* predicate.

CONSTRAINTS:
1. ALWAYS use lowercase SQL keywords.
2. ALWAYS include a LIMIT clause (default 1000) unless explicitly overridden.
3. If joining geometries with mismatched SRIDs, wrap with ST_Transform().
4. NEVER use unbounded ST_Distance or ST_DumpPoints on large tables.
5. Output ONLY valid JSON matching this schema: {{"sql": "string", "explanation": "string"}}

USER QUERY: {user_query}
    """
    return system_prompt.strip()
```

## Step 2: Multi-Tier Validation Pipeline

Generated SQL must pass through a deterministic validation stack before reaching the database. Implement a three-tier validation sequence:

**Tier 1: Syntax & AST Parsing**
Use `sqlglot` to parse the generated statement into an abstract syntax tree. Verify that all referenced tables and columns exist in the injected schema. Catch missing commas, malformed CTEs, and invalid PostGIS function signatures before execution.

**Tier 2: Spatial Semantics Verification**
Validate geometry type compatibility and CRS alignment. For example, if a query attempts `ST_Intersects(geom_a, geom_b)` where `geom_a` is `geometry(Point, 4326)` and `geom_b` is `geometry(Polygon, 3857)`, inject an automatic `ST_Transform` wrapper or reject the statement.

**Tier 3: Performance & Index Boundaries**
Ensure bounding box operators (`&&`) precede expensive spatial joins. Reject queries lacking `LIMIT` clauses for exploratory patterns. Validate that `ST_Buffer` or `ST_DWithin` include explicit distance units and do not exceed safe computational thresholds.

```python
import sqlglot
from sqlglot import expressions as exp
from typing import Tuple

class SpatialValidator:
    def __init__(self, schema_context: SpatialSchemaContext):
        self.schema = schema_context

    def validate(self, raw_sql: str) -> Tuple[bool, str]:
        """Runs 3-tier validation. Returns (is_valid, error_or_clean_sql)."""
        try:
            # Tier 1: AST Parsing & Syntax Check
            ast = sqlglot.parse_one(raw_sql, dialect="postgres")
            self._check_table_columns(ast)

            # Tier 2: Spatial Semantics & CRS Alignment
            self._check_crs_alignment(ast)

            # Tier 3: Performance & Index Boundaries
            self._check_bounding_box_pre_filter(raw_sql)
            self._check_limit_clause(ast)

            return True, raw_sql

        except sqlglot.errors.ParseError as e:
            return False, f"SYNTAX_ERROR: {str(e)}"
        except ValueError as e:
            return False, f"SPATIAL_VIOLATION: {str(e)}"

    def _check_table_columns(self, ast: exp.Expression):
        for table in ast.find_all(exp.Table):
            if table.name and table.name != self.schema.table_name:
                raise ValueError(f"Unauthorized table reference: {table.name}")

    def _check_crs_alignment(self, ast: exp.Expression):
        """In production, query pg_catalog to detect SRID mismatches."""
        pass  # Extend with live schema introspection

    def _check_bounding_box_pre_filter(self, raw_sql: str):
        """Validates && operator precedes ST_Intersects/ST_DWithin."""
        sql_upper = raw_sql.upper()
        needs_bbox = any(f in sql_upper for f in ("ST_INTERSECTS", "ST_DWITHIN"))
        if needs_bbox and "&&" not in raw_sql:
            raise ValueError(
                "Missing bounding-box pre-filter (&&). "
                "Queries using ST_Intersects or ST_DWithin require a && pre-filter to use the GiST index."
            )

    def _check_limit_clause(self, ast: exp.Expression):
        if not ast.find(exp.Limit):
            raise ValueError("Missing LIMIT clause for exploratory query.")
```

## Step 3: Execution Guardrails & Deterministic Fallback Routing

Once validated, the SQL statement enters a controlled execution environment. Direct database connections from LLM agents must be abstracted behind a query router that enforces timeout boundaries, row-count caps, and automatic retry logic. When PostGIS execution fails due to memory constraints or topology errors, the pipeline should route to alternative compute layers.

For medium-scale spatial operations, routing to [GeoPandas & PostGIS Tool Routing](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/geopandas-postgis-tool-routing/) enables in-memory vector processing with explicit geometry validation. For topology-heavy workflows, such as snapping, gap closing, or network routing, the pipeline delegates to specialized [Topology Rule Enforcement via LLMs](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/topology-rule-enforcement-via-llms/) modules that apply OGC Simple Features standards before committing results.

```python
import asyncpg
import asyncio
from typing import Optional, Dict, Any

class SpatialQueryExecutor:
    def __init__(self, pool: asyncpg.Pool, timeout_sec: float = 30.0, max_rows: int = 5000):
        self.pool = pool
        self.timeout_sec = timeout_sec
        self.max_rows = max_rows

    async def execute_safe(self, validated_sql: str) -> Dict[str, Any]:
        """Executes validated SQL with strict resource boundaries."""
        async with self.pool.acquire() as conn:
            try:
                await conn.execute(f"SET statement_timeout = '{int(self.timeout_sec * 1000)}ms'")
                rows = await conn.fetch(validated_sql)

                if len(rows) > self.max_rows:
                    return {
                        "status": "error",
                        "message": f"Result set exceeds {self.max_rows} row limit.",
                        "fallback_route": "apply_limit_and_retry"
                    }

                return {
                    "status": "success",
                    "rows_fetched": len(rows),
                    "data": [dict(r) for r in rows]
                }
            except asyncpg.exceptions.QueryCanceledError:
                return {
                    "status": "error",
                    "message": "Query exceeded timeout boundary.",
                    "fallback_route": "geopandas_memory_engine"
                }
            except asyncpg.exceptions.UndefinedTableError as e:
                return {
                    "status": "error",
                    "message": f"Table or column not found: {e}",
                    "fallback_route": "schema_refresh_and_retry"
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": str(e),
                    "fallback_route": "geopandas_memory_engine"
                }
```

## Step 4: Architectural Integration & Pipeline Orchestration

Prompt-to-Spatial-SQL Generation does not operate in isolation. It functions as the semantic translation layer within a broader LLM-assisted geoprocessing architecture. To maintain determinism across async vs sync processing boundaries, the pipeline must serialize prompt states, validation results, and execution metadata into a unified trace log. This enables reproducible debugging and continuous prompt optimization.

When integrating with enterprise geospatial platforms, align the validation schema with the official [PostGIS documentation](https://postgis.net/docs/) and reference the [OGC Simple Features specification](https://www.ogc.org/standard/sfa/) for geometry type compliance. For developers seeking deeper implementation patterns, the companion guide on [Generating Valid PostGIS Queries from Natural Language](https://www.spatialllm.org/geospatial-prompt-engineering-tool-routing/prompt-to-spatial-sql-generation/generating-valid-postgis-queries-from-natural-language/) provides extended few-shot templates, CRS transformation matrices, and AST manipulation recipes.

Key integration principles:
- **Stateless Prompt Routing**: Each generation request carries its own schema context. Never cache spatial prompts across sessions.
- **Deterministic Error Mapping**: Parse PostgreSQL error codes (`42883` for undefined function, `22001` for string too long, `XX000` for internal error) and map them to structured LLM feedback loops for self-correction.
- **Index-Aware Generation**: Prefer `ST_DWithin` over `ST_Distance` for radius searches to leverage GiST index bounds.
- **Topology Preservation**: When generating `ST_Union` or `ST_Collect`, enforce `ST_MakeValid` post-processing to prevent invalid geometry commits.

By treating Prompt-to-Spatial-SQL Generation as a constrained compiler rather than a generative text task, platform teams can safely scale natural language interfaces to spatial databases while maintaining strict performance, accuracy, and compliance boundaries.

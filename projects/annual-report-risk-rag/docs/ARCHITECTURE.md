# Architecture notes

## Request flow (RAG mode)

1. `ingest.load_document` reads the file and `chunk_document` splits it into overlapping word
   windows inside each Markdown section. Each chunk carries `chunk_id`, `section`, `position`.
2. `HybridRetriever` builds a BM25 index and (optionally) a vector index over
   `"{section}. {text}"`. Prepending the section title is a cheap way to inject table/heading
   context into both legs.
3. For each `MetricSpec` the pipeline retrieves `top_k` chunks for `spec.query`, calls the
   extractor, then the `GroundingValidator`.
4. The result is an `ExtractionReport` whose rows are `ValidatedMetric` objects with a
   `GroundingStatus` explaining acceptance or rejection.

## Request flow (agent mode)

The agent shares the retriever but the model decides what to search. Tools:

* `search_report(query, top_k)` returns excerpts with ids and records which chunks the model saw.
* `submit_metric(...)` accepts one `ExtractedMetric`; duplicates and schema violations are
  returned to the model as `is_error` tool results so it can correct itself.

The loop ends when every metric is submitted, the model stops calling tools, or `max_turns`
is hit. Submissions are validated against the union of chunks the model saw.

## Grounding validator

```
found == false                       -> NOT_FOUND
chunk_id not retrieved or quote not
  a whitespace-normalised substring  -> QUOTE_NOT_IN_CHUNK
no lossless rendering of value in
  the quote (12.3, 12.30, 1,480 ...)  -> VALUE_NOT_IN_QUOTE
unit-normalised value outside the
  spec's typical_range               -> OUT_OF_RANGE
otherwise                            -> GROUNDED (accepted)
```

Only lossless renderings are accepted when checking that the value is in the quote, so a
model that "rounds" 12.34 to 12.3 or invents a digit is rejected rather than silently accepted.

## Testing strategy

* Pure functions (tokeniser, BM25, hashing embedder, number parsing) have direct unit tests.
* The Anthropic adapter is tested with a fake client object that records kwargs, so the exact
  request shape (`output_format`, `output_config.effort`, tool params, transcript translation)
  is pinned without network access.
* The LLM extractor and the agent are tested with `StubStructuredLLM` / `ScriptedToolLLM`,
  including adversarial cases: fabricated quotes, wrong metric ids, duplicate submissions,
  invalid tool arguments and runaway loops.
* CLI commands are tested through Typer's `CliRunner`.

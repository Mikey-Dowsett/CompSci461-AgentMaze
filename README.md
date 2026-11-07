# Agent Maze Search Profiler

Launch the GUI from this directory with:

```sh
python3 main.py
```

## Evaluation modes

- **Single run** profiles the selected algorithm on the current maze, then shows its search trace and metrics.
- **Batch compare** runs the fixed reproducibility campaign: 10×10, 15×15, 20×20, and 30×30 mazes at 8%, 16%, and 24% obstacle density. Each of the 12 size/density cases uses a deterministic seed derived from the entered base seed. Every method/configuration is run five times against the same generated maze.

The batch contains 18 applicable algorithm configurations: BFS, DFS, and ID-DFS in both movement modes; Greedy Best-First and A* in both movement modes with Manhattan, Euclidean, and Chebyshev heuristics. Heuristic variants are omitted for algorithms that do not use heuristics. The default campaign performs 1,080 runs. It runs in a background thread, reports progress, and can be stopped between search runs.

The same campaign can be configured from JSON and launched without the GUI:

```json
{
  "base_seed": 70804,
  "sizes": [10, 15, 20, 30],
  "densities": [0.08, 0.16, 0.24],
  "repetitions": 5,
  "output_dir": "batch_results"
}
```

Save that as `campaign.json`, then run `python3 batch_runner.py --config campaign.json`. The runner rejects fewer than five repetitions.

Reports are written to `batch_results/` as a timestamped annotated JSON report and a CSV of raw runs. The report records the protocol, maze grids and seeds, raw measurements, aggregates, and metric definitions. Each result window compares one matched graph at a time and can chart runtime, expansions, path length, RSS, or algorithmic footprint.

## Metric conventions

- Runtime is `perf_counter_ns` around the search call; maze generation and subprocess startup are excluded.
- Path length, solution depth, and path cost count moves; diagonal moves cost one.
- Generated counts unique discovered states for BFS/DFS/priority searches. ID-DFS counts states generated again in later depth limits.
- Expanded counts each processed expansion, including repeated ID-DFS expansions.
- Observed branching is the mean and maximum number of legal neighbors of expanded nodes; already discovered neighbors are included.
- Algorithmic footprint is the peak number of frontier entries plus expanded states (for ID-DFS, expanded states in the current depth iteration).
- Python peak is the `tracemalloc` peak during the search call. Process peak RSS is the isolated worker process high-water RSS, including interpreter/import and maze-setup baseline; compare it as process memory, not as incremental search-only allocation.
- Runtime and memory summaries show the mean and best/worst repetition range.

BFS and DFS now honor the movement selector. ID-DFS does not use a heuristic; its heuristic control is disabled. Diagonal A* may use heuristics that are not admissible under unit-cost diagonal moves, so its path is not guaranteed to be optimal for every heuristic.

## Tests

Run the standard-library test suite with:

```sh
python3 -m unittest discover -s tests
```

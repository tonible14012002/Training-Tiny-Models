# Phase Path Queries

This document explains how phase relationships work using `phase_path` and the available query methods.

## Phase Path Structure

Phases form a chain where each phase points to its entire ancestry through `phase_path`:

- **Base phase**: `phase_path = ""`
- **1st child**: `phase_path = "<base_phase_id>"`
- **2nd child**: `phase_path = "<base_phase_id>/<1st_child_id>"`
- **nth child**: `phase_path = "<base_phase_id>/.../<n-1_phase_id>"`

### Example Hierarchy

```
Phase 0 (id=a, phase_path="")
  └─> Phase 1 (id=b, phase_path="a")
       └─> Phase 2 (id=c, phase_path="a/b")
            └─> Phase 3 (id=d, phase_path="a/b/c")
```

## Model Methods (PipelinePhase)

### `get_previous_phase_id() -> Optional[str]`

Returns the ID of the immediate previous phase (last element in path).

```python
Phase 3 (phase_path="a/b/c").get_previous_phase_id()  # Returns "c"
Phase 2 (phase_path="a/b").get_previous_phase_id()    # Returns "b"
Phase 1 (phase_path="a").get_previous_phase_id()      # Returns "a"
Phase 0 (phase_path="").get_previous_phase_id()       # Returns None
```

### `get_parent_phase_id() -> Optional[str]`

Returns the ID of the root/base phase (first element in path).

```python
Phase 3 (phase_path="a/b/c").get_parent_phase_id()  # Returns "a"
Phase 2 (phase_path="a/b").get_parent_phase_id()    # Returns "a"
Phase 1 (phase_path="a").get_parent_phase_id()      # Returns "a"
Phase 0 (phase_path="").get_parent_phase_id()       # Returns None
```

### `get_ancestor_phase_ids() -> List[str]`

Returns all ancestor phase IDs from root to immediate previous.

```python
Phase 3 (phase_path="a/b/c").get_ancestor_phase_ids()  # Returns ["a", "b", "c"]
Phase 2 (phase_path="a/b").get_ancestor_phase_ids()    # Returns ["a", "b"]
Phase 1 (phase_path="a").get_ancestor_phase_ids()      # Returns ["a"]
Phase 0 (phase_path="").get_ancestor_phase_ids()       # Returns []
```

### `get_depth() -> int`

Returns the depth of this phase in the hierarchy (0 for root).

```python
Phase 3 (phase_path="a/b/c").get_depth()  # Returns 3
Phase 2 (phase_path="a/b").get_depth()    # Returns 2
Phase 1 (phase_path="a").get_depth()      # Returns 1
Phase 0 (phase_path="").get_depth()       # Returns 0
```

### `build_child_path() -> str`

Constructs the path that a new child phase should have.

```python
Phase 0 (phase_path="").build_child_path()       # Returns "a" (assuming id="a")
Phase 1 (phase_path="a").build_child_path()      # Returns "a/b" (assuming id="b")
Phase 2 (phase_path="a/b").build_child_path()    # Returns "a/b/c" (assuming id="c")
```

## Repository Methods (PhaseRepository)

### `create(previous_phase_id=None) -> PipelinePhase`

Creates a new phase as a child of the previous phase.

```python
# Create base phase
phase_0 = await repo.create(
    pipeline_id=pipeline_id,
    phase_number=0,
    previous_phase_id=None  # No previous - creates base phase with phase_path=""
)

# Create child phase
phase_1 = await repo.create(
    pipeline_id=pipeline_id,
    phase_number=1,
    previous_phase_id=phase_0.id  # Creates phase with phase_path="<phase_0_id>"
)
```

### `get_child_phases(phase_id) -> List[PipelinePhase]`

Gets ALL nested children using `phase_path LIKE "<phase_id>%"` query.

```python
# Given Phase 0 (id=a), returns [Phase 1, Phase 2, Phase 3]
children = await repo.get_child_phases("a")
# Queries: phase_path="a" OR phase_path LIKE "a/%"
```

### `get_direct_children(phase_id) -> List[PipelinePhase]`

Gets only DIRECT children (not nested descendants).

```python
# Given Phase 0 (id=a), returns only [Phase 1]
direct_children = await repo.get_direct_children("a")
# Queries: phase_path="a" (exact match)
```

### `get_previous_phase(phase_id) -> Optional[PipelinePhase]`

Gets the immediate previous phase by extracting and querying the last phase_id in path.

```python
# Given Phase 3 (id=d, phase_path="a/b/c"), returns Phase 2
previous = await repo.get_previous_phase("d")
# Extracts "c" from path, then queries: id="c"
```

### `get_parent_phase(phase_id) -> Optional[PipelinePhase]`

Gets the root/base phase by extracting and querying the first phase_id in path.

```python
# Given Phase 3 (id=d, phase_path="a/b/c"), returns Phase 0
parent = await repo.get_parent_phase("d")
# Extracts "a" from path, then queries: id="a" AND phase_path=""
```

### `get_all_ancestors(phase_id) -> List[PipelinePhase]`

Gets all ancestor phases from root to immediate previous.

```python
# Given Phase 3 (id=d, phase_path="a/b/c"), returns [Phase 0, Phase 1, Phase 2]
ancestors = await repo.get_all_ancestors("d")
# Extracts ["a", "b", "c"], then queries: id IN ("a", "b", "c")
```

### `get_phases_by_sequence(pipeline_id, root_phase_id) -> List[PipelinePhase]`

Gets all phases in a sequence (same root phase).

```python
# Given root Phase 0 (id=a), returns [Phase 0, Phase 1, Phase 2, Phase 3]
sequence = await repo.get_phases_by_sequence(pipeline_id, "a")
# Queries: (id="a") OR (phase_path="a") OR (phase_path LIKE "a/%")
```

### `get_root_phases(pipeline_id) -> List[PipelinePhase]`

Gets all root phases (phase_path="") for a pipeline.

```python
# Returns all base phases
roots = await repo.get_root_phases(pipeline_id)
# Queries: phase_path=""
```

## Usage Examples

### Creating a Sequence of Phases

```python
# Create base phase
phase_0 = await repo.create(
    pipeline_id=pipeline_id,
    phase_number=0,
    previous_phase_id=None
)
# Result: phase_path=""

# Create 1st child
phase_1 = await repo.create(
    pipeline_id=pipeline_id,
    phase_number=1,
    previous_phase_id=phase_0.id
)
# Result: phase_path="<phase_0_id>"

# Create 2nd child
phase_2 = await repo.create(
    pipeline_id=pipeline_id,
    phase_number=2,
    previous_phase_id=phase_1.id
)
# Result: phase_path="<phase_0_id>/<phase_1_id>"
```

### Finding the Latest Phase in a Sequence

```python
# Get all children of root phase
children = await repo.get_child_phases(root_phase_id)

if children:
    # Latest is the last one (deepest/highest depth)
    latest_phase = children[-1]
else:
    # No children, root is latest
    latest_phase = root_phase
```

### Navigating the Hierarchy

```python
# Get immediate previous phase
previous = await repo.get_previous_phase(current_phase.id)

# Get root/base phase
root = await repo.get_parent_phase(current_phase.id)

# Get all ancestors
ancestors = await repo.get_all_ancestors(current_phase.id)
# Returns: [root_phase, ..., immediate_previous_phase]

# Get all descendants
descendants = await repo.get_child_phases(current_phase.id)

# Get only direct children
direct_children = await repo.get_direct_children(current_phase.id)
```

## Key Insights

1. **No parent_phase_id parameter**: Relationships are determined solely by `phase_path`
2. **Efficient queries**: Use SQL LIKE patterns and exact matches instead of multiple lookups
3. **Clear semantics**:
   - `parent_phase` = root/base phase (phase_path="")
   - `previous_phase` = immediate predecessor
   - `child_phases` = all descendants
   - `direct_children` = only immediate children
4. **Path structure**: Each phase's path contains ALL ancestor IDs, making traversal efficient

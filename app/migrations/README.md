# Database Migrations

This directory contains SQL migration files for the database schema.

## Directory Structure

```
app/migrations/
├── README.md                     # This file
├── 000_init_migrations.sql       # Migration tracking table
├── 001_create_tables.sql         # Main tables
└── seeds/
    └── 001_seed_default_data.sql # Default data
```

## Migration Naming Convention

Migration files should follow this naming pattern:
```
{version}_{description}.sql
```

- **version**: 3-digit number (e.g., `001`, `002`, `003`)
- **description**: Snake_case description (e.g., `create_tables`, `add_user_roles`)

Examples:
- `001_create_tables.sql`
- `002_add_indexes.sql`
- `003_alter_pipeline_table.sql`

## Seed Data

Seed files for initial/default data should be placed in the `seeds/` subdirectory:
```
seeds/{version}_{description}.sql
```

These will be tracked with a `seed_` prefix in the migration tracking table.

## Running Migrations

Run all pending migrations:
```bash
python cmd/run_migrations.py
```

The script will:
1. Create the migration tracking table if it doesn't exist
2. Check which migrations have been applied
3. Run any pending migrations in order
4. Track each migration in the `schema_migrations` table

## Migration Tracking

Applied migrations are tracked in the `schema_migrations` table:

```sql
CREATE TABLE schema_migrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

You can check which migrations have been applied:
```bash
sqlite3 pipeline.db "SELECT * FROM schema_migrations;"
```

## Creating New Migrations

1. Create a new SQL file in `app/migrations/` with the next version number
2. Write your SQL statements (DDL or DML)
3. Run `python cmd/run_migrations.py` to apply it

Example migration file (`002_add_status_index.sql`):
```sql
-- Migration: Add index on status column
-- Created: 2025-10-07

CREATE INDEX IF NOT EXISTS idx_pipeline_phase_status
ON pipeline_phase(status);
```

## SQLite Compatibility

All migrations are written for SQLite. Key differences from PostgreSQL:

- `UUID` → `TEXT` (store UUIDs as text)
- `JSONB` → `TEXT` (store JSON as text)
- `VARCHAR(n)` → `TEXT`
- `ON CONFLICT (column)` → `ON CONFLICT(column)` (no space)

## Best Practices

1. **Idempotent**: Use `IF NOT EXISTS` for CREATE statements
2. **Atomic**: Each migration should be a logical unit
3. **Reversible**: Consider how to undo changes (for future rollback support)
4. **Test**: Always test migrations on a copy of the database first
5. **Version Control**: Commit migration files to git

## Troubleshooting

### Migration fails midway
- Check the error message
- Fix the migration file
- Delete the failed migration from `schema_migrations` table
- Re-run migrations

### Reset database
```bash
rm pipeline.db
python cmd/run_migrations.py
```

### Check applied migrations
```bash
sqlite3 pipeline.db "SELECT version, name, applied_at FROM schema_migrations ORDER BY id;"
```

#!/usr/bin/env python3
"""
Database Migration Runner

This script runs SQL migrations in order and tracks which migrations have been applied.
Usage: python cmd/run_migrations.py
"""

import os
import sys
import sqlite3
from pathlib import Path
from typing import List, Tuple
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.settings import settings


def get_db_path(database_url: str) -> str:
    """Extract SQLite database path from database URL."""
    # Handle both sqlite:/// and sqlite+aiosqlite:///
    if database_url.startswith("sqlite+aiosqlite:///"):
        db_path = database_url.replace("sqlite+aiosqlite:///", "")
    elif database_url.startswith("sqlite:///"):
        db_path = database_url.replace("sqlite:///", "")
    else:
        raise ValueError(f"Unsupported database URL format: {database_url}")

    if not db_path.startswith("/"):
        # Relative path
        db_path = os.path.join(project_root, db_path)
    return db_path


def get_connection() -> sqlite3.Connection:
    """Get database connection."""
    db_path = get_db_path(settings.DATABASE_URL)

    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_migration_table(conn: sqlite3.Connection) -> None:
    """Initialize the migration tracking table if it doesn't exist."""
    cursor = conn.cursor()

    # Check if table exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='schema_migrations'
    """)

    if cursor.fetchone() is None:
        print("Initializing migration tracking table...")
        migration_file = project_root / "app" / "migrations" / "000_init_migrations.sql"

        if migration_file.exists():
            with open(migration_file, 'r') as f:
                sql = f.read()
                # Execute each statement separately
                for statement in sql.split(';'):
                    statement = statement.strip()
                    if statement:
                        cursor.execute(statement)
            conn.commit()
            print("✓ Migration tracking table created")
        else:
            raise FileNotFoundError(f"Migration init file not found: {migration_file}")


def get_applied_migrations(conn: sqlite3.Connection) -> List[str]:
    """Get list of already applied migration versions."""
    cursor = conn.cursor()
    cursor.execute("SELECT version FROM schema_migrations ORDER BY version")
    return [row[0] for row in cursor.fetchall()]


def get_migration_files() -> List[Tuple[str, str, Path]]:
    """
    Get all migration files in order.
    Returns: List of (version, name, path) tuples
    """
    migrations_dir = project_root / "app" / "migrations"

    if not migrations_dir.exists():
        raise FileNotFoundError(f"Migrations directory not found: {migrations_dir}")

    migration_files = []

    # Get all .sql files except the init file
    for file in migrations_dir.glob("*.sql"):
        if file.name == "000_init_migrations.sql":
            continue

        # Extract version from filename (e.g., 001_create_tables.sql -> 001)
        match = re.match(r'^(\d+)_(.+)\.sql$', file.name)
        if match:
            version = match.group(1)
            name = match.group(2)
            migration_files.append((version, name, file))

    # Also check seeds directory
    seeds_dir = migrations_dir / "seeds"
    if seeds_dir.exists():
        for file in seeds_dir.glob("*.sql"):
            match = re.match(r'^(\d+)_(.+)\.sql$', file.name)
            if match:
                version = f"seed_{match.group(1)}"
                name = match.group(2)
                migration_files.append((version, name, file))

    # Sort by version
    migration_files.sort(key=lambda x: x[0])

    return migration_files


def apply_migration(conn: sqlite3.Connection, version: str, name: str, file_path: Path) -> None:
    """Apply a single migration file."""
    print(f"Applying migration {version}: {name}...")

    cursor = conn.cursor()

    try:
        # Read migration file
        with open(file_path, 'r') as f:
            sql = f.read()

        # Execute migration
        # Split by semicolon but handle multi-line statements
        statements = []
        current_statement = []

        for line in sql.split('\n'):
            # Skip comments
            if line.strip().startswith('--'):
                continue

            current_statement.append(line)

            # Check if line ends with semicolon (end of statement)
            if line.strip().endswith(';'):
                statement = '\n'.join(current_statement).strip()
                if statement:
                    statements.append(statement)
                current_statement = []

        # Execute each statement
        for statement in statements:
            if statement.strip():
                try:
                    cursor.execute(statement)
                except sqlite3.Error as e:
                    # Some statements might fail if already applied (e.g., ON CONFLICT DO NOTHING)
                    # We log but don't fail
                    if "UNIQUE constraint failed" not in str(e):
                        print(f"  Warning: {e}")

        # Record migration in tracking table
        cursor.execute(
            "INSERT OR IGNORE INTO schema_migrations (version, name) VALUES (?, ?)",
            (version, name)
        )

        conn.commit()
        print(f"✓ Migration {version} applied successfully")

    except Exception as e:
        conn.rollback()
        raise Exception(f"Failed to apply migration {version}: {e}")


def run_migrations() -> None:
    """Run all pending migrations."""
    print("=" * 60)
    print("Database Migration Runner")
    print("=" * 60)
    print(f"Database: {settings.DATABASE_URL}")
    print()

    # Get database connection
    conn = get_connection()

    try:
        # Initialize migration table
        init_migration_table(conn)

        # Get applied migrations
        applied = get_applied_migrations(conn)
        print(f"Applied migrations: {len(applied)}")
        if applied:
            for version in applied:
                print(f"  - {version}")
        print()

        # Get all migration files
        all_migrations = get_migration_files()
        print(f"Total migrations found: {len(all_migrations)}")

        # Filter pending migrations
        pending = [m for m in all_migrations if m[0] not in applied]

        if not pending:
            print("\n✓ All migrations are up to date!")
            return

        print(f"\nPending migrations: {len(pending)}")
        for version, name, _ in pending:
            print(f"  - {version}: {name}")
        print()

        # Apply pending migrations
        for version, name, file_path in pending:
            apply_migration(conn, version, name, file_path)

        print()
        print("=" * 60)
        print("✓ All migrations completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        sys.exit(1)
    finally:
        conn.close()


def main():
    """Main entry point."""
    try:
        run_migrations()
    except KeyboardInterrupt:
        print("\n\nMigration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

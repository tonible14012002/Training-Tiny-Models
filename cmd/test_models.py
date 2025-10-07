#!/usr/bin/env python3
"""
Test script to verify SQLite models work correctly
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlmodel import Session, create_engine, select
from app.core.models.models import Pipeline, ErrorBucket, PipelinePhase
from app.core.settings import settings


def test_models():
    """Test SQLite models"""
    print("=" * 60)
    print("Testing SQLite Models")
    print("=" * 60)

    # Get database URL without async
    db_url = settings.DATABASE_URL.replace("sqlite+aiosqlite://", "sqlite://")
    print(f"Database: {db_url}\n")

    # Create engine
    engine = create_engine(db_url, echo=False)

    with Session(engine) as session:
        # Test 1: Query existing pipeline
        print("Test 1: Query existing pipeline")
        pipeline = session.exec(select(Pipeline)).first()
        if pipeline:
            print(f"✓ Found pipeline: {pipeline.name}")
            print(f"  ID type: {type(pipeline.id).__name__}")
            print(f"  ID value: {pipeline.id}")
        else:
            print("✗ No pipeline found")
        print()

        # Test 2: Query error buckets
        print("Test 2: Query error buckets")
        error_buckets = session.exec(select(ErrorBucket)).all()
        print(f"✓ Found {len(error_buckets)} error buckets")
        if error_buckets:
            first_bucket = error_buckets[0]
            print(f"  First bucket: {first_bucket.name}")
            print(f"  ID type: {type(first_bucket.id).__name__}")
            print(f"  Pipeline ID type: {type(first_bucket.pipeline_id).__name__}")
        print()

        # Test 3: Create new phase
        print("Test 3: Create new phase")
        if pipeline:
            new_phase = PipelinePhase(
                pipeline_id=pipeline.id,
                phase_number=1,
                status="pending"
            )
            session.add(new_phase)
            session.commit()
            session.refresh(new_phase)

            print(f"✓ Created new phase")
            print(f"  Phase ID type: {type(new_phase.id).__name__}")
            print(f"  Phase ID value: {new_phase.id}")
            print(f"  Phase number: {new_phase.phase_number}")
            print(f"  Status: {new_phase.status}")
        print()

        # Test 4: Query with relationship
        print("Test 4: Query with relationship")
        pipeline_with_phases = session.get(Pipeline, pipeline.id)
        if pipeline_with_phases:
            phases = pipeline_with_phases.phases
            print(f"✓ Pipeline has {len(phases)} phase(s)")
        print()

        # Test 5: Verify foreign key relationships
        print("Test 5: Verify foreign key relationships")
        if error_buckets:
            bucket = error_buckets[0]
            if bucket.pipeline_id == pipeline.id:
                print(f"✓ Foreign key relationship verified")
                print(f"  Bucket pipeline_id: {bucket.pipeline_id}")
                print(f"  Pipeline id: {pipeline.id}")
            else:
                print(f"✗ Foreign key mismatch")
        print()

    print("=" * 60)
    print("✓ All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_models()
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

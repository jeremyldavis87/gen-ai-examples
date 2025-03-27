# Migration Versions

This directory contains individual database migration scripts for the gen-ai-examples project, using SQLAlchemy and Alembic with PostgreSQL and pgvector, following AWS best practices.

## Table of Contents

1. [Overview](#overview)
2. [Migration Files](#migration-files)
3. [Migration Workflow](#migration-workflow)
4. [Best Practices](#best-practices)
5. [AWS RDS Considerations](#aws-rds-considerations)

## Overview

The versions directory contains individual Alembic migration scripts that represent specific changes to the database schema. Each migration has a unique identifier and includes both upgrade and downgrade paths to ensure that changes can be applied and reverted reliably.

## Migration Files

Each migration file follows this naming convention:

```
{revision_id}_{description}.py
```

For example:

```
1a2b3c4d5e6f_create_documents_table.py
```

Each migration file contains:

- A unique revision ID
- A reference to the previous revision (down_revision)
- An upgrade() function that applies the changes
- A downgrade() function that reverts the changes
- Optional branch labels and dependencies

## Migration Workflow

### Migration Sequence

Migrations are applied in sequence based on their dependencies. The sequence is determined by the `down_revision` reference in each migration file, which creates a directed acyclic graph (DAG) of migrations.

### Initial Migration

The initial migration typically creates the base tables and extensions required by the application, such as:

- Installing the pgvector extension
- Creating the documents table with vector column
- Setting up initial indexes

### Subsequent Migrations

Subsequent migrations might include:

- Adding new tables or columns
- Modifying existing tables or columns
- Creating or dropping indexes
- Adding constraints or foreign keys
- Data migrations

## Best Practices

### Writing Migrations

1. **Keep Migrations Focused**: Each migration should make a specific, focused change to the database schema.

2. **Include Both Paths**: Always provide both upgrade and downgrade paths to ensure reversibility.

3. **Use Raw SQL When Necessary**: For pgvector-specific operations that aren't supported by SQLAlchemy, use raw SQL with `op.execute()`.

4. **Test Migrations**: Test both upgrade and downgrade paths in a development environment before applying to production.

5. **Comment Complex Logic**: Add comments to explain complex migration logic or non-obvious changes.

### Example Migration

```python
"""Create documents table with vector support

Revision ID: 1a2b3c4d5e6f
Revises: None
Create Date: 2023-01-15 10:30:45.678901

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic
revision = '1a2b3c4d5e6f'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create pgvector extension if it doesn't exist
    op.execute('CREATE EXTENSION IF NOT EXISTS vector;')
    
    # Create documents table
    op.create_table(
        'documents',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('embedding', postgresql.ARRAY(sa.Float()), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Convert embedding column to vector type
    op.execute('ALTER TABLE documents ALTER COLUMN embedding TYPE vector(1536);')
    
    # Create index on the embedding column
    op.execute(
        "CREATE INDEX embeddings_idx ON documents "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
    )


def downgrade():
    # Drop the documents table
    op.drop_table('documents')
    
    # We don't drop the pgvector extension as it might be used by other tables
```

## AWS RDS Considerations

### Aurora PostgreSQL Compatibility

When writing migrations for AWS Aurora PostgreSQL Serverless v2, consider these points:

1. **Extension Installation**: The pgvector extension must be installed in the `postgres` database before running migrations.

2. **Parameter Groups**: Ensure that the DB parameter group allows the pgvector extension.

3. **Version Compatibility**: Verify that the pgvector version is compatible with your PostgreSQL version.

4. **Performance Considerations**: Indexes on vector columns can be resource-intensive to create. For large tables, consider creating indexes concurrently to avoid locking the table.

5. **Serverless Scaling**: Be aware that resource-intensive migrations might cause Aurora Serverless to scale up, which could affect costs.

### Example Aurora-Specific Migration

```python
"""Optimize vector indexes for Aurora

Revision ID: 2b3c4d5e6f7g
Revises: 1a2b3c4d5e6f
Create Date: 2023-01-20 14:45:30.123456

"""

from alembic import op

# revision identifiers, used by Alembic
revision = '2b3c4d5e6f7g'
down_revision = '1a2b3c4d5e6f'
branch_labels = None
depends_on = None


def upgrade():
    # Drop existing index
    op.execute("DROP INDEX IF EXISTS embeddings_idx;")
    
    # Create optimized index for Aurora
    # Using more lists for better search performance on larger datasets
    op.execute(
        "CREATE INDEX embeddings_idx ON documents "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 200);"
    )
    
    # Set storage parameters for the table
    op.execute(
        "ALTER TABLE documents SET (autovacuum_vacuum_scale_factor = 0.05);"
    )


def downgrade():
    # Revert to original index
    op.execute("DROP INDEX IF EXISTS embeddings_idx;")
    op.execute(
        "CREATE INDEX embeddings_idx ON documents "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
    )
    
    # Reset storage parameters
    op.execute(
        "ALTER TABLE documents RESET (autovacuum_vacuum_scale_factor);"
    )
```

For more information on managing migrations, refer to the parent [migrations/README.md](../README.md) file.

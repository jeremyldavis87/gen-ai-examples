"""Create documents table with pgvector support

Revision ID: 001
Revises: 
Create Date: 2025-03-26

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Create documents table
    op.create_table(
        'documents',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('source', sa.String(255), nullable=True),
        sa.Column('chunk_id', sa.String(255), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Add vector column (can't use SQLAlchemy directly for this)
    op.execute('ALTER TABLE documents ADD COLUMN embedding vector(1536)')
    
    # Create index for vector similarity search
    op.execute(
        'CREATE INDEX documents_embedding_idx ON documents USING ivfflat (embedding vector_cosine_ops)'
    )


def downgrade():
    # Drop table
    op.drop_table('documents')
    
    # We don't drop the pgvector extension as it might be used by other tables

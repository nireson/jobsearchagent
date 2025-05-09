"""Rename query to search_query

Revision ID: c56459ef0ee2
Revises: 
Create Date: 2025-04-25 10:48:57.020425

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c56459ef0ee2'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('search_result', schema=None) as batch_op:
        batch_op.add_column(sa.Column('search_query', sa.Text(), nullable=False))
        batch_op.drop_column('query')

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('search_result', schema=None) as batch_op:
        batch_op.add_column(sa.Column('query', sa.TEXT(), nullable=False))
        batch_op.drop_column('search_query')

    # ### end Alembic commands ###

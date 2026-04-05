# Database connection pool
# provides single connection pool to PostgreSQL.
# All modules that need database access import get_db() to get connection rather than creating their own connection pool.

from contextlib import contextmanager
from typing import Generator

from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor

from config import DATABASE_URL


_pool=pg_pool.ThreadedConnectionPool | None = None 

def init_pool(minconn:int=2,maxconn:int=10):
    ''' Initialize the connection pool. Called once during application startup '''
    global _pool
    _pool=pg_pool.ThreadedConnectionPool(minconn,maxconn,DATABASE_URL)

def get_pool()->pg_pool.ThreadedConnectionPool:
    """ Return the pool. Initialize it if needed."""
    global _pool 
    if _pool is None:
        init_pool()
    return _pool

@contextmanager
def get_db()->Generator:
    """
    Context manager that provides a database connection from the pool.
    Usage:
        with get_db as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM anamoly_table )
                rows=cur.fetchall()
        Connection is automatically returned to the pool when the block exits.
        If an exception occurs the transction automatically rolls back.         
    """
    conn=get_pool().getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        get_pool().putconn(conn)    
        
def get_dict_cursor():
    """
    Convenience function to get a cursor that returns rows as dictionaries instead of tuples.
    usage:
    with get_db() as cur:
        cur.executr("SELECT * form  monitored_regions")
        regions=cur.fetchall() # Each row is a dict like {'region_id': 'eth_tigray', ...} 
    """
    with get_db() as conn:
        with conn.cursor(RealDictCursor) as cur:
            yield cur

def initialize_schema():
    """
    Runs schema.sql against the database.
    Safe to run multiple time cause all statements use "CREATE if not exists"
    """
    with open("sehema.sql",'r') as f:
        sql=f.read()
    
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
    print("Database schema initialized...")

def seed_regions() -> None:
    """
    Inserts monitored regions from config.py into the database.
    Uses ON CONFLICT DO UPDATE so it's safe to run repeatedly.
    """
    from config import MONITORED_REGIONS
 
    with get_db() as conn:
        with conn.cursor() as cur:
            for region in MONITORED_REGIONS:
                bbox = region.bbox  # [min_lng, min_lat, max_lng, max_lat]
                cur.execute(
                    """
                    INSERT INTO monitored_regions
                        (region_id, name, country_code, admin1, boundary, buyer_ids)
                    VALUES (
                        %s, %s, %s, %s,
                        ST_MakeEnvelope(%s, %s, %s, %s, 4326),
                        %s
                    )
                    ON CONFLICT (region_id) DO UPDATE SET
                        name         = EXCLUDED.name,
                        country_code = EXCLUDED.country_code,
                        admin1       = EXCLUDED.admin1,
                        boundary     = EXCLUDED.boundary,
                        buyer_ids    = EXCLUDED.buyer_ids
                    """,
                    (
                        region.region_id,
                        region.name,
                        region.country_code,
                        region.admin1,
                        bbox[0], bbox[1], bbox[2], bbox[3],
                        region.buyer_ids,
                    )
                )
    print(f"✓ Seeded {len(MONITORED_REGIONS)} monitored regions.")
 
 
if __name__ == "__main__":
    # Run directly to bootstrap a fresh database:
    #   python db.py
    print("Initializing Witness database...")
    initialize_schema()
    seed_regions()
    print("Done. Database is ready.")
                    
                 
            
        
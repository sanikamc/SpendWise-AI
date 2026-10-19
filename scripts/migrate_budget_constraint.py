"""
scripts/migrate_budget_constraint.py
====================================
Safe, idempotent SQLite migration to update the Budget table uniqueness
constraint from (year, month) to (user_id, year, month).

Key features:
1. Backs up or verifies pre-migration database snapshot.
2. Performs 12-step SQLite table rebuild inside an atomic transaction.
3. Automatically rolls back changes on any failure.
4. Preserves all primary keys, user_ids, amounts, dates, and timestamps.
5. Idempotent: safe to run multiple times without data loss or corruption.
"""

import os
import sys
import sqlite3
from typing import Optional, List, Tuple, Dict, Any


def is_already_migrated(conn: sqlite3.Connection) -> bool:
    """Check if the budgets table already has the (user_id, year, month) unique constraint."""
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='budgets'")
    row = cursor.fetchone()
    if not row or not row[0]:
        return False
    sql = row[0]
    return "uq_user_budget_year_month" in sql or "UNIQUE (user_id, year, month)" in sql or "UNIQUE(user_id, year, month)" in sql


def get_budget_rows(conn: sqlite3.Connection) -> List[Tuple]:
    """Retrieve all rows from budgets table in deterministic order."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, user_id, year, month, amount_paise, created_at
        FROM budgets
        ORDER BY id ASC
    """)
    return cursor.fetchall()


def migrate_budget_schema(db_path: str) -> bool:
    """
    Executes the SQLite table rebuild to update the unique constraint.
    Returns True on success or if already migrated.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        # 1. Check idempotency
        if is_already_migrated(conn):
            print(f"[IDEMPOTENT] '{db_path}': 'budgets' table already enforces uq_user_budget_year_month.")
            return True

        # 2. Pre-checks and data snapshot
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        integrity = cursor.fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"Database pre-migration integrity check failed: {integrity}")

        cursor.execute("PRAGMA foreign_key_check;")
        fk_pre = cursor.fetchall()
        if fk_pre:
            raise RuntimeError(f"Database pre-migration foreign key check failed: {fk_pre}")

        snapshot_rows = get_budget_rows(conn)
        print(f"[PRE-CHECK] Captured {len(snapshot_rows)} existing budget rows for preservation.")

        # 3. Check for duplicates under the new constraint before modifying
        user_year_month_set = set()
        for r in snapshot_rows:
            key = (r[1], r[2], r[3])  # (user_id, year, month)
            if r[1] is not None and key in user_year_month_set:
                raise ValueError(f"Existing duplicate detected for user {r[1]} in {r[2]}-{r[3]}")
            if r[1] is not None:
                user_year_month_set.add(key)

        # 4. Atomic Table Rebuild
        conn.execute("PRAGMA foreign_keys = OFF;")
        try:
            conn.execute("BEGIN TRANSACTION;")

            # Clean any stale temporary table from previous failed attempts
            conn.execute("DROP TABLE IF EXISTS budgets_new;")

            # Create new table with updated multi-tenant unique constraint
            conn.execute("""
                CREATE TABLE budgets_new (
                    id INTEGER NOT NULL, 
                    user_id INTEGER, 
                    year INTEGER NOT NULL, 
                    month INTEGER NOT NULL, 
                    amount_paise INTEGER NOT NULL, 
                    created_at DATETIME, 
                    PRIMARY KEY (id), 
                    CONSTRAINT uq_user_budget_year_month UNIQUE (user_id, year, month), 
                    FOREIGN KEY(user_id) REFERENCES users (id)
                );
            """)

            # Copy all existing data exactly
            conn.execute("""
                INSERT INTO budgets_new (id, user_id, year, month, amount_paise, created_at)
                SELECT id, user_id, year, month, amount_paise, created_at FROM budgets;
            """)

            # Drop old table and rename new table
            conn.execute("DROP TABLE budgets;")
            conn.execute("ALTER TABLE budgets_new RENAME TO budgets;")

            # Recreate indices
            conn.execute("CREATE INDEX IF NOT EXISTS ix_budgets_user_id ON budgets (user_id);")

            # Verify foreign keys before committing
            fk_post = conn.execute("PRAGMA foreign_key_check;").fetchall()
            if fk_post:
                raise RuntimeError(f"Foreign key integrity check failed after rebuild: {fk_post}")

            conn.execute("COMMIT;")
            print("[TRANSACTION] Committed budgets table rebuild successfully.")

        except Exception as exc:
            conn.execute("ROLLBACK;")
            print(f"[ROLLBACK] Error occurred, transaction rolled back: {exc}")
            raise exc
        finally:
            conn.execute("PRAGMA foreign_keys = ON;")

        # 5. Post-migration data verification
        post_rows = get_budget_rows(conn)
        if len(post_rows) != len(snapshot_rows):
            raise AssertionError(
                f"Row count mismatch! Pre: {len(snapshot_rows)}, Post: {len(post_rows)}"
            )

        for pre_r, post_r in zip(snapshot_rows, post_rows):
            if pre_r != post_r:
                raise AssertionError(
                    f"Data mismatch for budget row!\nPre:  {pre_r}\nPost: {post_r}"
                )

        cursor.execute("PRAGMA integrity_check;")
        post_integrity = cursor.fetchone()[0]
        if post_integrity != "ok":
            raise RuntimeError(f"Database post-migration integrity check failed: {post_integrity}")

        print(f"[SUCCESS] Migration completed and verified for '{db_path}'.")
        print(f"[VERIFIED] {len(post_rows)} budget rows preserved intact.")
        return True

    finally:
        conn.close()


if __name__ == "__main__":
    target_db = sys.argv[1] if len(sys.argv) > 1 else os.path.join("instance", "expenses.db")
    print(f"Target database: {target_db}")
    try:
        success = migrate_budget_schema(target_db)
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Migration failed: {e}", file=sys.stderr)
        sys.exit(1)

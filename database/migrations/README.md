# Database migrations

Empty on purpose: the application is not deployed yet, so schema changes are
made directly in `src/db.py::_SCHEMA` (every statement is idempotent and the
recorded `schema_version` stays at 1). Existing databases are development
databases and can be recreated.

Once the application is deployed, this folder holds the migrations:

- one script per schema change, ordered by a numeric prefix —
  `001_add_x.sql` (plain SQL) or `001_add_x.py` (a `migrate(conn)` function
  for data transformations);
- `src.db.ensure_database` applies, in order, every script whose number is
  above the database's recorded `schema_version`, then records the new
  version;
- migrations must be additive/non-destructive wherever possible and are
  committed to git (this folder is exempted from the `database/` ignore
  rule).

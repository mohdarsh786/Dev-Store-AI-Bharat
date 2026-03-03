# Database Migrations

This directory contains PostgreSQL schema migration scripts for the DevStore platform.

## Migration Files

The migrations are numbered sequentially and should be run in order:

1. **001_create_resources_table.sql** - Main resources table (APIs, Models, Datasets)
2. **002_create_categories_tables.sql** - Categories and resource_categories tables
3. **003_create_users_table.sql** - User profiles and preferences
4. **004_create_search_history_table.sql** - Search history tracking
5. **005_create_resource_usage_table.sql** - Resource usage tracking
6. **006_create_resource_rankings_table.sql** - Computed ranking scores
7. **007_create_health_checks_table.sql** - Health check results

## Running Migrations

### Option 1: Using psql command line

```bash
# Run all migrations in order
for file in backend/migrations/*.sql; do
    psql -h <hostname> -U <username> -d <database> -f "$file"
done
```

### Option 2: Using the Python migration runner

```bash
# Run all pending migrations
python backend/run_migrations.py

# Run a specific migration
python backend/run_migrations.py --file 001_create_resources_table.sql
```

### Option 3: Manual execution

Connect to your PostgreSQL database and execute each file:

```bash
psql -h <hostname> -U <username> -d <database>
\i backend/migrations/001_create_resources_table.sql
\i backend/migrations/002_create_categories_tables.sql
# ... continue for all files
```

## Database Schema Overview

### Core Tables

- **resources**: Stores all APIs, Models, and Datasets with metadata
- **categories**: Hierarchical categories for organizing resources
- **resource_categories**: Many-to-many relationship between resources and categories

### User Tables

- **users**: User profiles with authentication and preferences
- **search_history**: User search queries and interactions
- **resource_usage**: Track user actions (view, download, test, bookmark)

### Analytics Tables

- **resource_rankings**: Daily computed ranking scores
- **health_checks**: Historical health check results

## Indexes

All tables include optimized indexes for:
- Primary key lookups
- Foreign key relationships
- Common query patterns (filtering, sorting)
- Time-based queries

## Constraints

- **CHECK constraints**: Enforce valid enum values (type, pricing_type, health_status, action)
- **FOREIGN KEY constraints**: Maintain referential integrity with CASCADE deletes
- **UNIQUE constraints**: Prevent duplicates (cognito_id, email, category slug)
- **NOT NULL constraints**: Ensure required fields are populated

## Requirements

- PostgreSQL 12 or higher (for gen_random_uuid() function)
- UUID extension (usually enabled by default)

## Rollback

To rollback migrations, drop tables in reverse order:

```sql
DROP TABLE IF EXISTS health_checks CASCADE;
DROP TABLE IF EXISTS resource_rankings CASCADE;
DROP TABLE IF EXISTS resource_usage CASCADE;
DROP TABLE IF EXISTS search_history CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS resource_categories CASCADE;
DROP TABLE IF EXISTS categories CASCADE;
DROP TABLE IF EXISTS resources CASCADE;
```

## Notes

- All migrations use `IF NOT EXISTS` to be idempotent
- JSONB fields provide flexibility for type-specific data
- Timestamps use `NOW()` for automatic creation tracking
- All tables include comments for documentation

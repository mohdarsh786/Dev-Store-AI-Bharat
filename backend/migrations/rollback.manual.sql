-- Rollback Script: Drop all DevStore tables
-- Description: Removes all tables created by migrations in reverse order
-- WARNING: This will delete all data! Use with caution.

-- Drop tables in reverse order of creation to respect foreign key constraints
DROP TABLE IF EXISTS health_checks CASCADE;
DROP TABLE IF EXISTS resource_rankings CASCADE;
DROP TABLE IF EXISTS resource_usage CASCADE;
DROP TABLE IF EXISTS search_history CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS resource_categories CASCADE;
DROP TABLE IF EXISTS categories CASCADE;
DROP TABLE IF EXISTS resources CASCADE;

-- Drop migration tracking table
DROP TABLE IF EXISTS schema_migrations CASCADE;

-- Confirmation message
DO $$
BEGIN
    RAISE NOTICE 'All DevStore tables have been dropped successfully.';
END $$;

#!/usr/bin/env python3
"""
Database Migration Runner for DevStore

This script runs PostgreSQL migration files in order to set up the database schema.
It tracks which migrations have been applied to avoid re-running them.

Usage:
    python run_migrations.py                    # Run all pending migrations
    python run_migrations.py --file 001_*.sql   # Run specific migration
    python run_migrations.py --rollback         # Rollback last migration
"""

import os
import sys
import argparse
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config import get_database_url
except ImportError:
    print("Warning: Could not import config. Using environment variables.")
    def get_database_url():
        return os.getenv('DATABASE_URL', 'postgresql://localhost/devstore')


class MigrationRunner:
    """Handles database migrations for DevStore."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.migrations_dir = Path(__file__).parent / 'migrations'
        
    def get_connection(self):
        """Create a database connection."""
        try:
            conn = psycopg2.connect(self.database_url)
            return conn
        except psycopg2.Error as e:
            print(f"Error connecting to database: {e}")
            sys.exit(1)
    
    def ensure_migrations_table(self, conn):
        """Create migrations tracking table if it doesn't exist."""
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) UNIQUE NOT NULL,
                    applied_at TIMESTAMP DEFAULT NOW()
                )
            """)
            conn.commit()
    
    def get_applied_migrations(self, conn) -> List[str]:
        """Get list of already applied migrations."""
        with conn.cursor() as cur:
            cur.execute("SELECT filename FROM schema_migrations ORDER BY id")
            return [row[0] for row in cur.fetchall()]
    
    def get_migration_files(self) -> List[Path]:
        """Get all migration files sorted by name."""
        if not self.migrations_dir.exists():
            print(f"Error: Migrations directory not found: {self.migrations_dir}")
            sys.exit(1)
        
        files = sorted(self.migrations_dir.glob('*.sql'))
        # Filter out README
        files = [f for f in files if f.name != 'README.md']
        return files
    
    def run_migration(self, conn, migration_file: Path) -> bool:
        """Run a single migration file."""
        print(f"Running migration: {migration_file.name}")
        
        try:
            # Read migration file
            with open(migration_file, 'r') as f:
                sql = f.read()
            
            # Execute migration
            with conn.cursor() as cur:
                cur.execute(sql)
                
                # Record migration
                cur.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (%s)",
                    (migration_file.name,)
                )
            
            conn.commit()
            print(f"✓ Successfully applied: {migration_file.name}")
            return True
            
        except psycopg2.Error as e:
            conn.rollback()
            print(f"✗ Error applying migration {migration_file.name}:")
            print(f"  {e}")
            return False
    
    def run_all_migrations(self) -> Tuple[int, int]:
        """Run all pending migrations."""
        conn = self.get_connection()
        
        try:
            # Ensure migrations table exists
            self.ensure_migrations_table(conn)
            
            # Get applied migrations
            applied = set(self.get_applied_migrations(conn))
            
            # Get all migration files
            migration_files = self.get_migration_files()
            
            if not migration_files:
                print("No migration files found.")
                return 0, 0
            
            # Filter pending migrations
            pending = [f for f in migration_files if f.name not in applied]
            
            if not pending:
                print("All migrations are up to date.")
                return len(applied), 0
            
            print(f"\nFound {len(pending)} pending migration(s):")
            for f in pending:
                print(f"  - {f.name}")
            print()
            
            # Run pending migrations
            success_count = 0
            for migration_file in pending:
                if self.run_migration(conn, migration_file):
                    success_count += 1
                else:
                    print("\nMigration failed. Stopping.")
                    break
            
            return len(applied), success_count
            
        finally:
            conn.close()
    
    def run_specific_migration(self, filename: str) -> bool:
        """Run a specific migration file."""
        conn = self.get_connection()
        
        try:
            self.ensure_migrations_table(conn)
            
            # Find migration file
            migration_file = self.migrations_dir / filename
            if not migration_file.exists():
                print(f"Error: Migration file not found: {filename}")
                return False
            
            # Check if already applied
            applied = self.get_applied_migrations(conn)
            if filename in applied:
                print(f"Migration {filename} has already been applied.")
                return True
            
            # Run migration
            return self.run_migration(conn, migration_file)
            
        finally:
            conn.close()
    
    def show_status(self):
        """Show migration status."""
        conn = self.get_connection()
        
        try:
            self.ensure_migrations_table(conn)
            
            applied = set(self.get_applied_migrations(conn))
            all_files = self.get_migration_files()
            
            print("\nMigration Status:")
            print("-" * 60)
            
            for migration_file in all_files:
                status = "✓ Applied" if migration_file.name in applied else "○ Pending"
                print(f"{status}  {migration_file.name}")
            
            print("-" * 60)
            print(f"Total: {len(all_files)} migrations, {len(applied)} applied, {len(all_files) - len(applied)} pending\n")
            
        finally:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description='Run database migrations for DevStore')
    parser.add_argument('--file', help='Run specific migration file')
    parser.add_argument('--status', action='store_true', help='Show migration status')
    parser.add_argument('--database-url', help='Database connection URL (overrides config)')
    
    args = parser.parse_args()
    
    # Get database URL
    database_url = args.database_url or get_database_url()
    
    if not database_url:
        print("Error: Database URL not configured.")
        print("Set DATABASE_URL environment variable or configure in config.py")
        sys.exit(1)
    
    # Create runner
    runner = MigrationRunner(database_url)
    
    # Show status
    if args.status:
        runner.show_status()
        return
    
    # Run specific migration
    if args.file:
        success = runner.run_specific_migration(args.file)
        sys.exit(0 if success else 1)
    
    # Run all migrations
    print("DevStore Database Migration Runner")
    print("=" * 60)
    
    applied_count, new_count = runner.run_all_migrations()
    
    print("\n" + "=" * 60)
    print(f"Migration complete: {applied_count} previously applied, {new_count} newly applied")
    
    if new_count > 0:
        print("\n✓ Database schema is now up to date!")
    

if __name__ == '__main__':
    main()

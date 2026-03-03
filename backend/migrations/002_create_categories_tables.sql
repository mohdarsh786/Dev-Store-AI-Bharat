-- Migration: Create categories and resource_categories tables
-- Description: Tables for organizing resources into categories and subcategories
-- Requirements: Section 3.1 PostgreSQL Schema

-- Categories table for hierarchical organization
CREATE TABLE IF NOT EXISTS categories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    slug VARCHAR(100) NOT NULL UNIQUE,
    parent_id UUID REFERENCES categories(id) ON DELETE CASCADE,
    resource_type VARCHAR(20) CHECK (resource_type IN ('api', 'model', 'dataset', 'all')),
    display_order INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for categories
CREATE INDEX IF NOT EXISTS idx_categories_parent ON categories(parent_id);
CREATE INDEX IF NOT EXISTS idx_categories_type ON categories(resource_type);
CREATE INDEX IF NOT EXISTS idx_categories_slug ON categories(slug);

-- Many-to-many relationship between resources and categories
CREATE TABLE IF NOT EXISTS resource_categories (
    resource_id UUID REFERENCES resources(id) ON DELETE CASCADE,
    category_id UUID REFERENCES categories(id) ON DELETE CASCADE,
    PRIMARY KEY (resource_id, category_id)
);

-- Create indexes for resource_categories
CREATE INDEX IF NOT EXISTS idx_rc_resource ON resource_categories(resource_id);
CREATE INDEX IF NOT EXISTS idx_rc_category ON resource_categories(category_id);

-- Add comments for documentation
COMMENT ON TABLE categories IS 'Hierarchical categories for organizing resources';
COMMENT ON COLUMN categories.slug IS 'URL-friendly identifier for the category';
COMMENT ON COLUMN categories.parent_id IS 'Parent category ID for hierarchical structure';
COMMENT ON COLUMN categories.resource_type IS 'Type of resources this category applies to';
COMMENT ON TABLE resource_categories IS 'Many-to-many relationship between resources and categories';

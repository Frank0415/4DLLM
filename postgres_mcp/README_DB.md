# Database Setup for 4D-STEM Analysis

This document explains how to set up the PostgreSQL database for the 4D-STEM analysis tools.

## Prerequisites

1. PostgreSQL database server running
2. Python dependencies installed (psycopg2, etc.)

## Database Configuration

1. Copy the example configuration file:
   ```bash
   cp config/db_config_example.json config/database.json
   ```

2. Edit `config/database.json` with your database credentials:
   ```json
   {
     "host": "localhost",
     "port": 5432,
     "user": "your_username",
     "password": "your_password",
     "dbname": "4dllm"
   }
   ```

## Initialize Database Tables

Run the database initialization script to create the required tables:

```bash
python postgres_mcp/init_db.py
```

This will create three tables:
- `scans`: Stores metadata about each scan
- `raw_mat_files`: Links to .mat files for each scan
- `diffraction_patterns`: Individual diffraction patterns within .mat files

## Verify Database Setup

To verify that the database is properly set up:

```bash
python postgres_mcp/verify_db.py
```

## Troubleshooting

If you encounter errors:
1. Make sure PostgreSQL is running
2. Verify database credentials in `config/database.json`
3. Check that the database user has CREATE TABLE permissions
4. Ensure the database exists (you may need to create it first with `createdb 4dllm`)
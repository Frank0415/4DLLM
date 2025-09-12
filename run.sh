#!/bin/bash

# Script to check and use running Docker containers' ports for PostgreSQL and pgAdmin
# If containers are running, output their current mapped ports
# If not, start them via docker-compose

COMPOSE_FILE="docker-compose.yml"  # Adjust if your file is named differently

# Function to get mapped port for a container
get_mapped_port() {
    local container_name=$1
    local internal_port=$2
    docker port "$container_name" | grep "$internal_port" | cut -d: -f2 || echo "N/A"
}

# Check if PostgreSQL container is running
if docker ps --filter "name=postgres" --filter "status=running" | grep -q postgres; then
    echo "PostgreSQL container is running."
    pg_port=$(get_mapped_port postgres 5432)
    echo "PostgreSQL mapped port: $pg_port"
else
    echo "PostgreSQL container not running. Starting..."
    docker compose -f "$COMPOSE_FILE" up -d postgres_db
    sleep 5  # Wait for startup
    pg_port=$(get_mapped_port postgres 5432)
    echo "PostgreSQL now running on mapped port: $pg_port"
fi

# Check if pgAdmin container is running
if docker ps --filter "name=pgadmin" --filter "status=running" | grep -q pgadmin; then
    echo "pgAdmin container is running."
    pgadmin_port=$(get_mapped_port pgadmin 80)
    echo "pgAdmin mapped port: $pgadmin_port"
else
    echo "pgAdmin container not running. Starting..."
    docker compose -f "$COMPOSE_FILE" up -d pgadmin
    sleep 5  # Wait for startup
    pgadmin_port=$(get_mapped_port pgadmin 80)
    echo "pgAdmin now running on mapped port: $pgadmin_port"
fi

echo "Access pgAdmin at: http://localhost:$pgadmin_port"
echo "Connect to PostgreSQL at: localhost:$pg_port (user: postgres, db: 4dllm)"
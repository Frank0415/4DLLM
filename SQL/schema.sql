-- #############################################################################
-- #  4D-STEM Research Database Schema
-- #  Database: 4dllm
-- #############################################################################

-- Drop tables in reverse dependency order
DROP TABLE IF EXISTS pattern_comparisons cascade;
DROP TABLE IF EXISTS simulated_patterns cascade;
DROP TABLE IF EXISTS cif_files cascade;
DROP TABLE IF EXISTS llm_analysis_batches cascade;
DROP TABLE IF EXISTS llm_analysis_tags cascade;
DROP TABLE IF EXISTS llm_analysis_results cascade;
DROP TABLE IF EXISTS llm_representative_patterns cascade;
DROP TABLE IF EXISTS llm_analyses cascade;
DROP TABLE IF EXISTS pattern_cluster_assignments cascade;
DROP TABLE IF EXISTS identified_clusters cascade;
DROP TABLE IF EXISTS clustering_runs cascade;
DROP TABLE IF EXISTS diffraction_patterns cascade;
DROP TABLE IF EXISTS raw_mat_files cascade;
DROP TABLE IF EXISTS scans cascade;

-- ===============================================================================
-- Layer 1: Raw Data Layer
-- ===============================================================================

-- Table 1: scans (Scan experiments)
-- Purpose: Record each complete scan experiment, the top-level organization unit.
CREATE TABLE scans (
    id SERIAL PRIMARY KEY,
    scan_name VARCHAR(255) UNIQUE NOT NULL,
    folder_path VARCHAR(1024) UNIQUE NOT NULL,
    classification_map_path VARCHAR(1024),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
COMMENT ON TABLE scans IS 'Record each complete scan experiment, corresponding to a top-level folder.';
COMMENT ON COLUMN scans.classification_map_path IS 'Global color map path generated after K-Means classification.';

-- Table 2: raw_mat_files (Raw .mat data files)
-- Purpose: Record raw .mat files for each scan experiment.
CREATE TABLE raw_mat_files (
    id SERIAL PRIMARY KEY,
    scan_id INTEGER NOT NULL,
    row_index INTEGER NOT NULL,
    file_path VARCHAR(1024) NOT NULL,
    UNIQUE (scan_id, row_index)
);
COMMENT ON TABLE raw_mat_files IS 'Record raw .mat files for each scan experiment.';

-- Table 3: diffraction_patterns (Basic diffraction points)
-- Purpose: Record the most basic data unit: a (row, col) coordinate point.
CREATE TABLE diffraction_patterns (
    id SERIAL PRIMARY KEY,
    source_mat_id INTEGER NOT NULL,
    col_index INTEGER NOT NULL,
    cluster_label INTEGER,
    clustering_run_id INTEGER,
    UNIQUE (source_mat_id, col_index)
);
COMMENT ON TABLE diffraction_patterns IS 'Most basic data unit, representing a (row, col) coordinate point.';

-- ===============================================================================
-- Layer 2: Clustering Analysis Layer
-- ===============================================================================

-- Table 4: clustering_runs (Clustering experiment logs)
-- Purpose: Record each independent K-Means clustering experiment for reproducibility.
CREATE TABLE clustering_runs (
    id SERIAL PRIMARY KEY,
    scan_id INTEGER NOT NULL,
    run_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    k_value INTEGER NOT NULL,
    algorithm_details TEXT,
    results_path VARCHAR(1024)
);
COMMENT ON TABLE clustering_runs IS 'Record each independent K-Means clustering experiment.';

-- Table 5: identified_clusters (Identified clusters)
-- Purpose: Represent a specific category identified in a clustering experiment.
CREATE TABLE identified_clusters (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL,
    cluster_index INTEGER NOT NULL,
    pattern_count INTEGER DEFAULT 0,
    centroid_features JSONB,
    UNIQUE (run_id, cluster_index)
);
COMMENT ON TABLE identified_clusters IS 'Represent a specific category identified in a clustering experiment.';

-- Table 6: pattern_cluster_assignments (Point-cluster assignments)
-- Purpose: Core connection table recording which cluster each diffraction point is assigned to.
CREATE TABLE pattern_cluster_assignments (
    pattern_id INTEGER NOT NULL,
    cluster_id INTEGER NOT NULL,
    assignment_confidence FLOAT,
    PRIMARY KEY (pattern_id, cluster_id)
);
COMMENT ON TABLE pattern_cluster_assignments IS 'Core connection table recording which cluster each diffraction point is assigned to.';

-- ===============================================================================
-- Layer 3: LLM Analysis Layer
-- ===============================================================================

-- Table 7: llm_analyses (LLM analysis results)
-- Purpose: Store deep LLM analysis results for identified clusters.
CREATE TABLE llm_analyses (
    id SERIAL PRIMARY KEY,
    cluster_id INTEGER UNIQUE NOT NULL,
    representative_patterns_count INTEGER,
    llm_assigned_class VARCHAR(255),
    llm_detailed_features JSONB,
    analysis_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    report_path VARCHAR(1024)
);
COMMENT ON TABLE llm_analyses IS 'Store deep LLM analysis results for identified clusters.';

-- Table 8: llm_representative_patterns (LLM representative patterns)
-- Purpose: Record which diffraction points were selected as representatives for LLM analysis.
CREATE TABLE llm_representative_patterns (
    analysis_id INTEGER NOT NULL,
    pattern_id INTEGER NOT NULL,
    selection_reason TEXT,
    PRIMARY KEY (analysis_id, pattern_id)
);
COMMENT ON TABLE llm_representative_patterns IS 'Record which diffraction points were selected as representatives for LLM analysis.';

-- ===============================================================================
-- Layer 4: LLM Analysis Results Layer
-- ===============================================================================

-- Table 9: llm_analysis_results (LLM analysis final results)
-- Purpose: Store final results for each diffraction point after LLM analysis.
CREATE TABLE llm_analysis_results (
    id SERIAL PRIMARY KEY,
    pattern_id INTEGER UNIQUE NOT NULL,
    scan_id INTEGER NOT NULL,
    clustering_run_id INTEGER,
    cluster_id INTEGER,
    llm_analysis_id INTEGER,
    row_index INTEGER NOT NULL,
    col_index INTEGER NOT NULL,
    cluster_index INTEGER,
    llm_assigned_class VARCHAR(255),
    llm_detailed_features JSONB,
    analysis_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
COMMENT ON TABLE llm_analysis_results IS 'Store final results for each diffraction point after LLM analysis.';

-- Table 10: llm_analysis_tags (LLM analysis tags)
-- Purpose: Store structured tags from LLM analysis for querying and statistics.
CREATE TABLE llm_analysis_tags (
    id SERIAL PRIMARY KEY,
    result_id INTEGER NOT NULL,
    tag_category VARCHAR(100) NOT NULL,
    tag_value VARCHAR(100) NOT NULL,
    confidence_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
COMMENT ON TABLE llm_analysis_tags IS 'Store structured tags from LLM analysis for querying and statistics.';

-- Table 11: llm_analysis_batches (LLM analysis batches)
-- Purpose: Record batch information for LLM analysis for batch processing and monitoring.
CREATE TABLE llm_analysis_batches (
    id SERIAL PRIMARY KEY,
    batch_name VARCHAR(255) NOT NULL,
    total_patterns INTEGER NOT NULL,
    processed_patterns INTEGER DEFAULT 0,
    failed_patterns INTEGER DEFAULT 0,
    batch_status VARCHAR(50) DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
COMMENT ON TABLE llm_analysis_batches IS 'Record batch information for LLM analysis for batch processing and monitoring.';

-- ===============================================================================
-- Layer 5: CIF Simulation and Comparison Layer
-- ===============================================================================

-- Table 12: cif_files (CIF file information)
-- Purpose: Store CIF files and their crystallographic information.
CREATE TABLE cif_files (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(1024) NOT NULL,
    crystal_system VARCHAR(50),
    space_group VARCHAR(50),
    lattice_parameters JSONB,
    atoms JSONB,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
COMMENT ON TABLE cif_files IS 'Store CIF files and their crystallographic information.';

-- Table 13: simulated_patterns (Simulated diffraction patterns)
-- Purpose: Store simulated diffraction patterns generated from CIF files.
CREATE TABLE simulated_patterns (
    id SERIAL PRIMARY KEY,
    cif_id INTEGER NOT NULL,
    pattern_data BYTEA,
    metadata JSONB,
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
COMMENT ON TABLE simulated_patterns IS 'Store simulated diffraction patterns generated from CIF files.';

-- Table 14: pattern_comparisons (Pattern comparison results)
-- Purpose: Store comparison results between experimental and simulated patterns.
CREATE TABLE pattern_comparisons (
    id SERIAL PRIMARY KEY,
    experimental_pattern_id INTEGER,
    simulated_pattern_id INTEGER,
    similarity_score FLOAT,
    comparison_method VARCHAR(100),
    result_details JSONB,
    compared_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
COMMENT ON TABLE pattern_comparisons IS 'Store comparison results between experimental and simulated patterns.';

-- ===============================================================================
-- Create indexes for query performance
-- ===============================================================================
CREATE INDEX IF NOT EXISTS idx_scans_scan_name ON scans(scan_name);
CREATE INDEX IF NOT EXISTS idx_raw_mat_files_scan_id ON raw_mat_files(scan_id);
CREATE INDEX IF NOT EXISTS idx_raw_mat_files_row_index ON raw_mat_files(row_index);
CREATE INDEX IF NOT EXISTS idx_diffraction_patterns_source_mat_id ON diffraction_patterns(source_mat_id);
CREATE INDEX IF NOT EXISTS idx_diffraction_patterns_col_index ON diffraction_patterns(col_index);
CREATE INDEX IF NOT EXISTS idx_diffraction_patterns_cluster_label ON diffraction_patterns(cluster_label);
CREATE INDEX IF NOT EXISTS idx_diffraction_patterns_clustering_run_id ON diffraction_patterns(clustering_run_id);
CREATE INDEX IF NOT EXISTS idx_assignments_pattern_id ON pattern_cluster_assignments(pattern_id);
CREATE INDEX IF NOT EXISTS idx_assignments_cluster_id ON pattern_cluster_assignments(cluster_id);
CREATE INDEX IF NOT EXISTS idx_identified_clusters_run_id ON identified_clusters(run_id);
CREATE INDEX IF NOT EXISTS idx_llm_analyses_cluster_id ON llm_analyses(cluster_id);
CREATE INDEX IF NOT EXISTS idx_llm_analysis_results_coordinates ON llm_analysis_results(row_index, col_index);
CREATE INDEX IF NOT EXISTS idx_llm_analysis_results_class ON llm_analysis_results(llm_assigned_class);
CREATE INDEX IF NOT EXISTS idx_llm_analysis_results_scan_id ON llm_analysis_results(scan_id);
CREATE INDEX IF NOT EXISTS idx_llm_analysis_results_cluster_id ON llm_analysis_results(cluster_id);
CREATE INDEX IF NOT EXISTS idx_llm_analysis_results_clustering_run_id ON llm_analysis_results(clustering_run_id);
CREATE INDEX IF NOT EXISTS idx_llm_analysis_tags_result_id ON llm_analysis_tags(result_id);
CREATE INDEX IF NOT EXISTS idx_llm_analysis_tags_category ON llm_analysis_tags(tag_category);
CREATE INDEX IF NOT EXISTS idx_llm_analysis_tags_value ON llm_analysis_tags(tag_value);
CREATE INDEX IF NOT EXISTS idx_llm_analysis_batches_status ON llm_analysis_batches(batch_status);
CREATE INDEX IF NOT EXISTS idx_cif_files_filename ON cif_files(filename);
CREATE INDEX IF NOT EXISTS idx_simulated_patterns_cif_id ON simulated_patterns(cif_id);
CREATE INDEX IF NOT EXISTS idx_pattern_comparisons_experimental ON pattern_comparisons(experimental_pattern_id);
CREATE INDEX IF NOT EXISTS idx_pattern_comparisons_simulated ON pattern_comparisons(simulated_pattern_id);
CREATE INDEX IF NOT EXISTS idx_pattern_comparisons_similarity ON pattern_comparisons(similarity_score);

-- ===============================================================================
-- Add foreign key constraints
-- ===============================================================================
ALTER TABLE raw_mat_files 
ADD CONSTRAINT fk_raw_mat_files_scans 
FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE;

ALTER TABLE diffraction_patterns 
ADD CONSTRAINT fk_diffraction_patterns_raw_mat_files 
FOREIGN KEY (source_mat_id) REFERENCES raw_mat_files(id) ON DELETE CASCADE;

ALTER TABLE diffraction_patterns 
ADD CONSTRAINT fk_diffraction_patterns_clustering_runs 
FOREIGN KEY (clustering_run_id) REFERENCES clustering_runs(id) ON DELETE CASCADE;

ALTER TABLE clustering_runs 
ADD CONSTRAINT fk_clustering_runs_scans 
FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE;

ALTER TABLE identified_clusters 
ADD CONSTRAINT fk_identified_clusters_clustering_runs 
FOREIGN KEY (run_id) REFERENCES clustering_runs(id) ON DELETE CASCADE;

ALTER TABLE pattern_cluster_assignments 
ADD CONSTRAINT fk_pattern_cluster_assignments_diffraction_patterns 
FOREIGN KEY (pattern_id) REFERENCES diffraction_patterns(id) ON DELETE CASCADE;

ALTER TABLE pattern_cluster_assignments 
ADD CONSTRAINT fk_pattern_cluster_assignments_identified_clusters 
FOREIGN KEY (cluster_id) REFERENCES identified_clusters(id) ON DELETE CASCADE;

ALTER TABLE llm_analyses 
ADD CONSTRAINT fk_llm_analyses_identified_clusters 
FOREIGN KEY (cluster_id) REFERENCES identified_clusters(id) ON DELETE CASCADE;

ALTER TABLE llm_representative_patterns 
ADD CONSTRAINT fk_llm_representative_patterns_llm_analyses 
FOREIGN KEY (analysis_id) REFERENCES llm_analyses(id) ON DELETE CASCADE;

ALTER TABLE llm_representative_patterns 
ADD CONSTRAINT fk_llm_representative_patterns_diffraction_patterns 
FOREIGN KEY (pattern_id) REFERENCES diffraction_patterns(id) ON DELETE CASCADE;

ALTER TABLE llm_analysis_results 
ADD CONSTRAINT fk_llm_analysis_results_diffraction_patterns 
FOREIGN KEY (pattern_id) REFERENCES diffraction_patterns(id) ON DELETE CASCADE;

ALTER TABLE llm_analysis_results 
ADD CONSTRAINT fk_llm_analysis_results_scans 
FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE;

ALTER TABLE llm_analysis_results 
ADD CONSTRAINT fk_llm_analysis_results_clustering_runs 
FOREIGN KEY (clustering_run_id) REFERENCES clustering_runs(id) ON DELETE CASCADE;

ALTER TABLE llm_analysis_results 
ADD CONSTRAINT fk_llm_analysis_results_identified_clusters 
FOREIGN KEY (cluster_id) REFERENCES identified_clusters(id) ON DELETE CASCADE;

ALTER TABLE llm_analysis_results 
ADD CONSTRAINT fk_llm_analysis_results_llm_analyses 
FOREIGN KEY (llm_analysis_id) REFERENCES llm_analyses(id) ON DELETE CASCADE;

ALTER TABLE llm_analysis_tags 
ADD CONSTRAINT fk_llm_analysis_tags_llm_analysis_results 
FOREIGN KEY (result_id) REFERENCES llm_analysis_results(id) ON DELETE CASCADE;

ALTER TABLE simulated_patterns 
ADD CONSTRAINT fk_simulated_patterns_cif_files 
FOREIGN KEY (cif_id) REFERENCES cif_files(id) ON DELETE CASCADE;

ALTER TABLE pattern_comparisons 
ADD CONSTRAINT fk_pattern_comparisons_diffraction_patterns 
FOREIGN KEY (experimental_pattern_id) REFERENCES diffraction_patterns(id) ON DELETE CASCADE;

ALTER TABLE pattern_comparisons 
ADD CONSTRAINT fk_pattern_comparisons_simulated_patterns 
FOREIGN KEY (simulated_pattern_id) REFERENCES simulated_patterns(id) ON DELETE CASCADE;

-- ===============================================================================
-- Create views for common queries
-- ===============================================================================

-- View 1: Cluster statistics
CREATE OR REPLACE VIEW cluster_statistics AS
SELECT 
    cr.scan_id,
    cr.id as clustering_run_id,
    ic.cluster_index,
    COUNT(dp.id) as pattern_count,
    s.scan_name
FROM diffraction_patterns dp
JOIN raw_mat_files rmf ON dp.source_mat_id = rmf.id
JOIN scans s ON rmf.scan_id = s.id
JOIN clustering_runs cr ON dp.clustering_run_id = cr.id
JOIN identified_clusters ic ON dp.cluster_label = ic.cluster_index AND cr.id = ic.run_id
WHERE dp.cluster_label IS NOT NULL
GROUP BY cr.scan_id, cr.id, ic.cluster_index, s.scan_name
ORDER BY cr.scan_id, ic.cluster_index;

-- View 2: Spatial cluster distribution
CREATE OR REPLACE VIEW spatial_cluster_distribution AS
SELECT 
    s.scan_name,
    cr.id as clustering_run_id,
    rmf.row_index as x_coordinate,
    dp.col_index as y_coordinate,
    dp.cluster_label
FROM diffraction_patterns dp
JOIN raw_mat_files rmf ON dp.source_mat_id = rmf.id
JOIN scans s ON rmf.scan_id = s.id
JOIN clustering_runs cr ON dp.clustering_run_id = cr.id
WHERE dp.cluster_label IS NOT NULL
ORDER BY s.scan_name, rmf.row_index, dp.col_index;

-- View 3: LLM analysis overview
CREATE OR REPLACE VIEW llm_analysis_overview AS
SELECT 
    s.scan_name,
    cr.id as clustering_run_id,
    ic.cluster_index,
    la.llm_assigned_class,
    la.analysis_timestamp,
    s.id as scan_id
FROM llm_analyses la
JOIN identified_clusters ic ON la.cluster_id = ic.id
JOIN clustering_runs cr ON ic.run_id = cr.id
JOIN scans s ON cr.scan_id = s.id
ORDER BY s.scan_name, cr.id, ic.cluster_index;

-- View 4: Tag statistics
CREATE OR REPLACE VIEW tag_statistics AS
SELECT 
    lat.tag_category,
    lat.tag_value,
    COUNT(*) as count,
    AVG(lat.confidence_score) as avg_confidence
FROM llm_analysis_tags lat
GROUP BY lat.tag_category, lat.tag_value
ORDER BY lat.tag_category, count DESC;

-- View 5: Batch processing statistics
CREATE OR REPLACE VIEW batch_processing_stats AS
SELECT 
    lab.batch_name,
    lab.total_patterns,
    lab.processed_patterns,
    lab.failed_patterns,
    lab.batch_status,
    ROUND(100.0 * lab.processed_patterns / NULLIF(lab.total_patterns, 0), 2) as completion_percentage,
    lab.started_at,
    lab.completed_at
FROM llm_analysis_batches lab
ORDER BY lab.created_at DESC;

-- View 6: CIF file statistics
CREATE OR REPLACE VIEW cif_statistics AS
SELECT 
    cf.id,
    cf.filename,
    cf.crystal_system,
    cf.space_group,
    COUNT(sp.id) as simulated_patterns_count,
    cf.uploaded_at
FROM cif_files cf
LEFT JOIN simulated_patterns sp ON cf.id = sp.cif_id
GROUP BY cf.id, cf.filename, cf.crystal_system, cf.space_group, cf.uploaded_at
ORDER BY cf.uploaded_at DESC;

-- View 7: Pattern comparison overview
CREATE OR REPLACE VIEW comparison_overview AS
SELECT 
    pc.id as comparison_id,
    dp.id as experimental_pattern_id,
    sp.id as simulated_pattern_id,
    cf.filename as cif_filename,
    pc.similarity_score,
    pc.comparison_method,
    pc.compared_at
FROM pattern_comparisons pc
JOIN diffraction_patterns dp ON pc.experimental_pattern_id = dp.id
JOIN simulated_patterns sp ON pc.simulated_pattern_id = sp.id
JOIN cif_files cf ON sp.cif_id = cf.id
ORDER BY pc.similarity_score DESC, pc.compared_at DESC;

-- Success message
SELECT '4D-STEM research database schema created successfully!' AS status;
"""
Pattern simulator for generating diffraction patterns from CIF data.
"""

import json
import logging
import numpy as np
import pickle
from typing import Dict, Any, Optional, Union

from ..sql import SqlDriver, SafeSqlDriver

logger = logging.getLogger(__name__)

try:
    import abtem
    import ase
    import dask
    from scipy.ndimage import affine_transform, gaussian_filter

    SIMULATION_AVAILABLE = True
except ImportError:
    SIMULATION_AVAILABLE = False
    logger.warning(
        "Simulation dependencies (abtem, ase) not available. Install with: pip install abtem ase"
    )


class PatternSimulator:
    """Simulates diffraction patterns from crystallographic data."""

    def __init__(self, sql_driver: Union[SqlDriver, SafeSqlDriver]):
        """
        Initialize pattern simulator.

        Args:
            sql_driver: Database driver instance
        """
        self.sql_driver = sql_driver

        # Default simulation configuration
        self.config = {
            "sample_width": 100,
            "sample_width_sampling": 0.1,
            "sample_thickness": 50,
            "sample_thickness_sampling": 50,
            "target_size": 256,
            "semi_angle_cutoff_range": [0.8, 1.3],
            "max_angle_range": [41.00, 41.00],
            "sigma_heat_range": [0.0, 0.1],
            "sigma_blur_range": [0.5, 1.5],
            "translation_range": [0, 0],
            "beam_energy": 200e3,  # 200 keV
        }

        if SIMULATION_AVAILABLE:
            self._setup_abtem()

    def _setup_abtem(self):
        """Configure abtem for simulation."""
        try:
            abtem.config.set({"visualize.cmap": "viridis"})
            abtem.config.set({"visualize.continuous_update": True})
            abtem.config.set({"visualize.autoscale": True})
            abtem.config.set({"visualize.reciprocal_space_units": "mrad"})
            abtem.config.set({"device": "cpu"})  # Use CPU by default for stability
            abtem.config.set({"fft": "fftw"})
            abtem.config.set({"dask.chunk-size-gpu": "2048 MB"})
            abtem.config.set({"dask.lazy": False})
            dask.config.set({"num_workers": 1})
            logger.info("abtem configured successfully")
        except Exception as e:
            logger.warning(f"Failed to configure abtem: {e}")

    async def generate_patterns(self, cif_id: int, **params) -> Dict[str, Any]:
        """
        Generate simulated diffraction patterns from CIF data.

        Args:
            cif_id: CIF file ID
            **params: Simulation parameters

        Returns:
            Dictionary with simulation results
        """
        if not SIMULATION_AVAILABLE:
            return {
                "cif_id": cif_id,
                "patterns_generated": 0,
                "status": "simulation_dependencies_not_available",
                "error": "Please install abtem and ase: pip install abtem ase",
            }

        logger.info(f"Generating patterns for CIF ID: {cif_id}")

        try:
            # Get simulation parameters
            count = params.get("count", 10)

            # Load CIF information from database
            cif_info = await self._get_cif_info(cif_id)
            if not cif_info:
                raise ValueError(f"CIF file with ID {cif_id} not found")

            # Load crystal structure from CIF file
            crystal_structure = self._load_crystal_structure(cif_info["file_path"])

            # Generate multiple patterns with random orientations
            pattern_ids = []
            results = []

            for i in range(count):
                logger.info(f"Generating pattern {i + 1}/{count}")

                # Simulate single pattern
                pattern_data, metadata = await self._simulate_single_pattern(
                    crystal_structure, pattern_index=i
                )

                # Store pattern in database
                pattern_id = await self._store_pattern(cif_id, pattern_data, metadata)
                pattern_ids.append(pattern_id)
                results.append({"pattern_id": pattern_id, "metadata": metadata})

            return {
                "cif_id": cif_id,
                "patterns_generated": len(pattern_ids),
                "pattern_ids": pattern_ids,
                "results": results,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Failed to generate patterns: {e}")
            return {
                "cif_id": cif_id,
                "patterns_generated": 0,
                "status": "error",
                "error": str(e),
            }

    async def _get_cif_info(self, cif_id: int) -> Optional[Dict[str, Any]]:
        """Get CIF information from database."""
        result = await self.sql_driver.execute_query(
            "SELECT id, filename, file_path FROM cif_files WHERE id = %s", [cif_id]
        )

        if not result or len(result) == 0:
            return None

        row = result[0].cells
        return {
            "id": row["id"],
            "filename": row["filename"],
            "file_path": row["file_path"],
        }

    def _load_crystal_structure(self, cif_file_path: str):
        """Load crystal structure from CIF file."""
        try:
            # Load primitive unit cell
            primitive_unit = ase.io.read(cif_file_path)

            # Orthogonalize the cell
            orthogonal_unit = abtem.orthogonalize_cell(primitive_unit)

            # Create supercell based on sample dimensions
            sample_cell_parameter_list = orthogonal_unit.cell.cellpar()
            sample_cell_length_min = np.min(sample_cell_parameter_list[0:3])
            sample_cell_count = int(
                np.ceil(
                    np.max(
                        [self.config["sample_width"], self.config["sample_thickness"]]
                    )
                    / sample_cell_length_min
                    * 2
                )
            )

            # Create supercell
            sample = orthogonal_unit * (
                sample_cell_count,
                sample_cell_count,
                sample_cell_count,
            )

            # Center the sample
            sample.center(
                about=(
                    self.config["sample_width"] / 2,
                    self.config["sample_width"] / 2,
                    self.config["sample_thickness"] / 2,
                )
            )

            return sample

        except Exception as e:
            logger.error(f"Failed to load crystal structure: {e}")
            raise

    async def _simulate_single_pattern(self, sample, pattern_index: int = 0) -> tuple:
        """Simulate a single diffraction pattern."""

        # Create a copy for this simulation
        sim_sample = sample.copy()

        # Random rotation
        phi = np.random.uniform(0, 360)
        theta = np.random.uniform(0, 180)
        psi = np.random.uniform(0, 360)

        sim_sample.euler_rotate(
            phi=phi,
            theta=theta,
            psi=psi,
            center=(
                self.config["sample_width"] / 2,
                self.config["sample_width"] / 2,
                self.config["sample_thickness"] / 2,
            ),
        )

        # Cut cell
        sim_sample.cell = (
            self.config["sample_width"],
            self.config["sample_width"],
            self.config["sample_thickness"],
        )
        sim_sample = abtem.atoms.atoms_in_cell(sim_sample, margin=0)

        # Simulation parameters
        semi_angle_cutoff = np.random.uniform(*self.config["semi_angle_cutoff_range"])
        max_angle = np.random.uniform(*self.config["max_angle_range"])
        sigma_heat = np.random.uniform(*self.config["sigma_heat_range"])
        sigma_blur = np.random.uniform(*self.config["sigma_blur_range"])

        # Simulate thermal vibrations
        frozen_phonons = abtem.FrozenPhonons(
            sim_sample,
            sigmas=sigma_heat,
            num_configs=8,
            seed=np.random.randint(1e6, 1e9),
        )

        # Create potential
        potential = abtem.Potential(
            frozen_phonons,
            sampling=self.config["sample_width_sampling"],
            slice_thickness=self.config["sample_thickness_sampling"],
        )

        # Create probe
        probe = abtem.Probe(
            energy=self.config["beam_energy"],
            semiangle_cutoff=semi_angle_cutoff,
            sampling=self.config["sample_width_sampling"],
        )

        # Create detector
        detector = abtem.PixelatedDetector(max_angle=max_angle)

        # Single point scan
        scan = abtem.GridScan(
            start=(self.config["sample_width"] / 2, self.config["sample_width"] / 2),
            end=(
                self.config["sample_width"] / 2 + 1,
                self.config["sample_width"] / 2 + 1,
            ),
            sampling=2,
            potential=potential,
        )

        # Run simulation
        measurement = probe.scan(scan=scan, potential=potential, detectors=detector)
        measurement.compute(progress_bar=False, scheduler="threads", num_workers=1)

        # Extract pattern
        pattern = measurement[0].array[0, :, :]

        # Post-process pattern
        processed_pattern, center, pixel_size = self._post_process_pattern(
            pattern, sigma_blur
        )

        # Create metadata
        metadata = {
            "pattern_index": pattern_index,
            "rotation": {"phi": phi, "theta": theta, "psi": psi},
            "semi_angle_cutoff": semi_angle_cutoff,
            "max_angle": max_angle,
            "sigma_heat": sigma_heat,
            "sigma_blur": sigma_blur,
            "center": center,
            "pixel_size": pixel_size,
            "beam_energy": self.config["beam_energy"],
            "target_size": self.config["target_size"],
        }

        return processed_pattern, metadata

    def _post_process_pattern(self, pattern, sigma_blur):
        """Post-process the simulated pattern."""

        target_size = self.config["target_size"]

        # Get reciprocal space coordinates
        origin_x = pattern.shape[0] // 2
        origin_y = pattern.shape[1] // 2

        # Apply transformation to center the pattern
        transformation_matrix = np.array(
            [[1, 0, origin_x], [0, 1, origin_y], [0, 0, 1]]
        ) @ np.array(
            [
                [1, 0, -(float(pattern.shape[0]) - 1) / 2],
                [0, 1, -(float(pattern.shape[1]) - 1) / 2],
                [0, 0, 1],
            ]
        )

        pattern = affine_transform(pattern, transformation_matrix, order=1)

        # Resize to target size
        final_size = target_size
        transformation_matrix = (
            np.array(
                [
                    [1, 0, (float(pattern.shape[0]) - 1) / 2],
                    [0, 1, (float(pattern.shape[1]) - 1) / 2],
                    [0, 0, 1],
                ]
            )
            @ np.array(
                [
                    [(pattern.shape[0] - 1) / (final_size - 1), 0, 0],
                    [0, (pattern.shape[1] - 1) / (final_size - 1), 0],
                    [0, 0, 1],
                ]
            )
            @ np.array(
                [
                    [1, 0, -(float(final_size) - 1) / 2],
                    [0, 1, -(float(final_size) - 1) / 2],
                    [0, 0, 1],
                ]
            )
        )

        pattern = affine_transform(pattern, transformation_matrix, order=1)

        # Apply Gaussian blur
        pattern = gaussian_filter(pattern, sigma_blur, truncate=10)

        # Normalize
        pattern = pattern / np.sum(pattern)

        # Calculate pixel size (approximate)
        pixel_size = [1.0, 1.0]  # mrad per pixel (approximate)
        center = [(float(target_size) - 1) / 2, (float(target_size) - 1) / 2]

        return pattern, center, pixel_size

    async def _store_pattern(
        self, cif_id: int, pattern_data: np.ndarray, metadata: Dict[str, Any]
    ) -> int:
        """Store simulated pattern in database."""

        # Convert pattern to bytes
        pattern_bytes = pickle.dumps(pattern_data.astype(np.float32))

        # Store in database
        result = await self.sql_driver.execute_query(
            """
            INSERT INTO simulated_patterns (cif_id, pattern_data, metadata)
            VALUES (%s, %s, %s)
            RETURNING id;
            """,
            [cif_id, pattern_bytes, json.dumps(metadata)],
        )

        if not result or len(result) == 0:
            raise RuntimeError("Failed to store simulated pattern")

        pattern_id = result[0].cells["id"]
        logger.info(f"Stored simulated pattern with ID: {pattern_id}")

        return pattern_id

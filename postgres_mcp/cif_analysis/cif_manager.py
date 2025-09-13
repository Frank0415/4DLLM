"""
Integrated CIF file manager for the 4D-STEM analysis system.
This module provides CIF file management using the unified database interface.
"""

import json
import logging
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from ..sql import SqlDriver, SafeSqlDriver

logger = logging.getLogger(__name__)


class CIFManager:
    """CIF file manager integrated with the postgres_mcp database system."""

    def __init__(
        self,
        sql_driver: Union[SqlDriver, SafeSqlDriver],
        storage_dir: Optional[str] = None,
    ):
        """
        Initialize CIF manager.

        Args:
            sql_driver: Database driver instance
            storage_dir: Directory to store CIF files (defaults to project/cif_files)
        """
        self.sql_driver = sql_driver

        # Set up storage directory
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            # Default to project root/cif_files
            project_root = Path(__file__).parent.parent.parent
            self.storage_dir = project_root / "cif_files"

        self.storage_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"CIF storage directory: {self.storage_dir}")

    async def download_cif(self, url: str, filename: Optional[str] = None) -> str:
        """
        Download CIF file from URL and store it.

        Args:
            url: CIF file URL
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Local file path of downloaded CIF
        """
        try:
            # Download the file
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Determine filename
            if not filename:
                filename = url.split("/")[-1]
                if not filename.endswith(".cif"):
                    filename += ".cif"

            # Ensure filename is safe
            filename = "".join(c for c in filename if c.isalnum() or c in "._-")
            if not filename.endswith(".cif"):
                filename += ".cif"

            file_path = self.storage_dir / filename

            # Write file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(response.text)

            logger.info(f"Downloaded CIF file: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to download CIF file from {url}: {e}")
            raise

    async def upload_cif(self, source_path: str) -> str:
        """
        Upload local CIF file to managed storage.

        Args:
            source_path: Path to local CIF file

        Returns:
            Path to copied file in managed storage
        """
        try:
            source = Path(source_path)
            if not source.exists():
                raise FileNotFoundError(f"Source file does not exist: {source_path}")

            # Copy to managed storage
            dest_path = self.storage_dir / source.name

            with (
                open(source, "r", encoding="utf-8") as src,
                open(dest_path, "w", encoding="utf-8") as dst,
            ):
                dst.write(src.read())

            logger.info(f"Uploaded CIF file: {dest_path}")
            return str(dest_path)

        except Exception as e:
            logger.error(f"Failed to upload CIF file {source_path}: {e}")
            raise

    def parse_cif(self, file_path: str) -> Dict[str, Any]:
        """
        Parse CIF file to extract crystallographic information.

        Args:
            file_path: Path to CIF file

        Returns:
            Dictionary containing extracted crystallographic data
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            cif_info = {
                "crystal_system": None,
                "space_group": None,
                "lattice_parameters": {},
                "atoms": [],
                "chemical_formula": None,
                "compound_name": None,
            }

            lines = content.split("\n")
            current_loop = None
            loop_data = []

            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Handle loop data
                if line.startswith("loop_"):
                    current_loop = []
                    loop_data = []
                    continue
                elif line.startswith("_") and current_loop is not None:
                    current_loop.append(line)
                    continue
                elif current_loop is not None and not line.startswith("_"):
                    # This is data for the current loop
                    if line and not line.startswith("loop_"):
                        loop_data.append(line.split())
                    else:
                        # End of loop, process atom data if applicable
                        if any("atom" in col.lower() for col in current_loop):
                            self._process_atom_loop(current_loop, loop_data, cif_info)
                        current_loop = None
                        loop_data = []

                # Parse single-value fields
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue

                key, value = parts[0], parts[1]

                # Clean up value (remove quotes and error values in parentheses)
                value = value.strip("'\"")
                if "(" in value:
                    value = value.split("(")[0]

                # Extract specific crystallographic data
                if key == "_symmetry_cell_setting":
                    cif_info["crystal_system"] = value
                elif key in [
                    "_symmetry_space_group_name_H-M",
                    "_space_group_name_H-M_alt",
                ]:
                    cif_info["space_group"] = value
                elif key == "_chemical_formula_sum":
                    cif_info["chemical_formula"] = value
                elif key == "_chemical_name_common":
                    cif_info["compound_name"] = value
                elif key in [
                    "_cell_length_a",
                    "_cell_length_b",
                    "_cell_length_c",
                    "_cell_angle_alpha",
                    "_cell_angle_beta",
                    "_cell_angle_gamma",
                ]:
                    param_name = key.split("_")[-1]
                    try:
                        cif_info["lattice_parameters"][param_name] = float(value)
                    except ValueError:
                        logger.warning(
                            f"Could not parse lattice parameter {key}: {value}"
                        )

            return cif_info

        except Exception as e:
            logger.error(f"Failed to parse CIF file {file_path}: {e}")
            raise

    def _process_atom_loop(
        self, headers: List[str], data: List[List[str]], cif_info: Dict[str, Any]
    ):
        """Process atom site loop data from CIF file."""
        try:
            # Find column indices for atom data
            label_idx = next(
                (i for i, h in enumerate(headers) if "atom_site_label" in h), None
            )
            symbol_idx = next(
                (i for i, h in enumerate(headers) if "atom_site_type_symbol" in h), None
            )
            x_idx = next(
                (i for i, h in enumerate(headers) if "atom_site_fract_x" in h), None
            )
            y_idx = next(
                (i for i, h in enumerate(headers) if "atom_site_fract_y" in h), None
            )
            z_idx = next(
                (i for i, h in enumerate(headers) if "atom_site_fract_z" in h), None
            )

            if not all(
                idx is not None for idx in [label_idx, symbol_idx, x_idx, y_idx, z_idx]
            ):
                return

            for row in data:
                if len(row) > max(label_idx, symbol_idx, x_idx, y_idx, z_idx):
                    try:
                        atom = {
                            "label": row[label_idx],
                            "symbol": row[symbol_idx],
                            "x": float(row[x_idx].split("(")[0]),
                            "y": float(row[y_idx].split("(")[0]),
                            "z": float(row[z_idx].split("(")[0]),
                        }
                        cif_info["atoms"].append(atom)
                    except (ValueError, IndexError):
                        continue

        except Exception as e:
            logger.warning(f"Failed to process atom loop data: {e}")

    async def store_cif_info(
        self, filename: str, file_path: str, cif_info: Dict[str, Any]
    ) -> int:
        """
        Store CIF information in the database.

        Args:
            filename: Original filename
            file_path: Path to stored file
            cif_info: Parsed CIF information

        Returns:
            Database ID of stored CIF record
        """
        try:
            result = await self.sql_driver.execute_query(
                """
                INSERT INTO cif_files 
                (filename, file_path, crystal_system, space_group, lattice_parameters, atoms)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                [
                    filename,
                    file_path,
                    cif_info.get("crystal_system"),
                    cif_info.get("space_group"),
                    json.dumps(cif_info.get("lattice_parameters", {})),
                    json.dumps(cif_info.get("atoms", [])),
                ],
            )

            if not result or len(result) == 0:
                raise RuntimeError("Failed to insert CIF record")

            cif_id = result[0].cells["id"]
            logger.info(f"Stored CIF information in database with ID: {cif_id}")
            return cif_id

        except Exception as e:
            logger.error(f"Failed to store CIF info in database: {e}")
            raise

    async def get_cif_info(self, cif_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve CIF information from database.

        Args:
            cif_id: CIF record ID

        Returns:
            CIF information dictionary or None if not found
        """
        try:
            result = await SafeSqlDriver.execute_param_query(
                self.sql_driver,
                """
                SELECT id, filename, file_path, crystal_system, space_group, 
                       lattice_parameters, atoms, uploaded_at
                FROM cif_files
                WHERE id = {};
                """,
                [cif_id],
            )

            if not result or len(result) == 0:
                return None

            row = result[0].cells
            return {
                "id": row["id"],
                "filename": row["filename"],
                "file_path": row["file_path"],
                "crystal_system": row["crystal_system"],
                "space_group": row["space_group"],
                "lattice_parameters": row["lattice_parameters"],
                "atoms": row["atoms"],
                "uploaded_at": row["uploaded_at"],
            }

        except Exception as e:
            logger.error(f"Failed to get CIF info for ID {cif_id}: {e}")
            raise

    async def list_cif_files(self) -> List[Dict[str, Any]]:
        """
        List all CIF files in the database.

        Returns:
            List of CIF file information dictionaries
        """
        try:
            result = await SafeSqlDriver.execute_param_query(
                self.sql_driver,
                """
                SELECT id, filename, crystal_system, space_group, uploaded_at
                FROM cif_files
                ORDER BY uploaded_at DESC;
                """,
                [],
            )

            if not result:
                return []

            return [
                {
                    "id": row.cells["id"],
                    "filename": row.cells["filename"],
                    "crystal_system": row.cells["crystal_system"],
                    "space_group": row.cells["space_group"],
                    "uploaded_at": row.cells["uploaded_at"],
                }
                for row in result
            ]

        except Exception as e:
            logger.error(f"Failed to list CIF files: {e}")
            raise

    async def delete_cif(self, cif_id: int) -> bool:
        """
        Delete CIF file and its database record.

        Args:
            cif_id: CIF record ID

        Returns:
            True if deletion was successful
        """
        try:
            # First get the file path
            cif_info = await self.get_cif_info(cif_id)
            if not cif_info:
                return False

            # Delete the file
            file_path = Path(cif_info["file_path"])
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted CIF file: {file_path}")

            # Delete from database
            await self.sql_driver.execute_query(
                "DELETE FROM cif_files WHERE id = %s;", [cif_id]
            )

            logger.info(f"Deleted CIF record with ID: {cif_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete CIF with ID {cif_id}: {e}")
            raise

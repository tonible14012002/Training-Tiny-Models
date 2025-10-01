from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class NumericalFileAccessMixin:
    """
    Mixin class for loading and accessing files by numerical indexing.

    This mixin provides functionality similar to the trainer service's checkpoint
    storage pattern, where files/directories are stored with numerical names
    and can be accessed by their number.

    Classes using this mixin should define:
    - base_directory: str or Path - The base directory containing numbered items
    """

    @property
    def base_directory(self) -> str:
        """
        Base directory containing numbered files/directories.
        Must be implemented by the class using this mixin.
        """
        raise NotImplementedError("Classes using NumericalFileAccessMixin must define base_directory")

    def _get_all_numerical_items(self) -> List[int]:
        """
        Get all numerical items in the base directory.

        Returns:
            List[int]: Sorted list of numerical item numbers
        """
        base_path = Path(self.base_directory)

        if not base_path.exists():
            return []

        numerical_items = []
        for item in base_path.iterdir():
            if item.name.isdigit():
                numerical_items.append(int(item.name))

        return sorted(numerical_items)

    def _get_latest_number(self) -> int:
        """
        Get the latest (highest) numerical item number.

        Returns:
            int: Latest item number (0 if no items exist)
        """
        items = self._get_all_numerical_items()
        return max(items, default=0)

    def _get_next_number(self) -> int:
        """
        Get the next available numerical item number.

        Returns:
            int: Next item number
        """
        return self._get_latest_number() + 1

    def _get_item_path(self, number: int) -> Path:
        """
        Get the path for a specific numerical item.

        Args:
            number: Item number

        Returns:
            Path: Path to the numbered item
        """
        return Path(self.base_directory) / str(number)

    def _item_exists(self, number: int) -> bool:
        """
        Check if a numerical item exists.

        Args:
            number: Item number to check

        Returns:
            bool: True if item exists, False otherwise
        """
        return self._get_item_path(number).exists()

    def _ensure_base_directory(self) -> None:
        """
        Ensure the base directory exists, creating it if necessary.
        """
        base_path = Path(self.base_directory)
        base_path.mkdir(parents=True, exist_ok=True)

    def get_latest_item_path(self) -> Optional[Path]:
        """
        Get the path to the latest numerical item.

        Returns:
            Optional[Path]: Path to latest item, None if no items exist
        """
        latest_number = self._get_latest_number()
        if latest_number == 0 and not self._item_exists(0):
            return None
        return self._get_item_path(latest_number)

    def get_item_path_by_number(self, number: int) -> Optional[Path]:
        """
        Get the path to a specific numerical item if it exists.

        Args:
            number: Item number

        Returns:
            Optional[Path]: Path to item if it exists, None otherwise
        """
        if self._item_exists(number):
            return self._get_item_path(number)
        return None

    def get_item_path_by_id(self, checkpoint_id: str) -> Optional[Path]:
        """
        Get the path to a specific checkpoint by ID (supports both integer and sub-versions).

        Args:
            checkpoint_id: Checkpoint identifier (e.g., "1", "2", "1.1", "2.3")

        Returns:
            Optional[Path]: Path to checkpoint if it exists, None otherwise
        """
        base_path = Path(self.base_directory)
        checkpoint_path = base_path / checkpoint_id

        if checkpoint_path.exists():
            return checkpoint_path
        return None

    def list_all_items(self) -> List[int]:
        """
        List all available numerical items.

        Returns:
            List[int]: Sorted list of available item numbers
        """
        return self._get_all_numerical_items()

    def get_item_count(self) -> int:
        """
        Get the total count of numerical items.

        Returns:
            int: Number of items in the directory
        """
        return len(self._get_all_numerical_items())

    def _get_latest_sub_version(self, base_number: int) -> Optional[str]:
        """
        Get the latest sub-version for a given base numerical item.

        For example, if base_number=10 and sub-versions 10.1, 10.2, 10.3 exist,
        this returns "10.3".

        Args:
            base_number: The base item number

        Returns:
            The latest sub-version name (e.g., "10.3") or None if no sub-versions exist
        """
        base_path = Path(self.base_directory)

        if not base_path.exists():
            return None

        # Find all sub-versions matching pattern {base_number}.{version}
        sub_versions = []

        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith(f"{base_number}."):
                try:
                    # Extract version number after the dot
                    parts = item.name.split('.')
                    if len(parts) == 2 and parts[0] == str(base_number):
                        version = int(parts[1])
                        sub_versions.append((version, item.name))
                except (ValueError, IndexError):
                    continue

        if not sub_versions:
            return None

        # Sort by version and return the latest
        sub_versions.sort(key=lambda x: x[0], reverse=True)
        return sub_versions[0][1]

    def _get_next_sub_version_paths(self, base_number: int) -> tuple[str, str]:
        """
        Determine the paths for continual training with sub-versioning.

        Returns the path to load from and the path to save to. If no sub-versions
        exist, loads from base and saves to base.1. Otherwise, loads from latest
        sub-version and saves to next sub-version.

        Args:
            base_number: The base item number

        Returns:
            Tuple of (path_to_load_from, path_to_save_to)
        """
        base_path = Path(self.base_directory)
        latest_sub = self._get_latest_sub_version(base_number)

        if latest_sub is None:
            # No sub-versions exist, start from base item and create .1
            load_from = str(base_path / str(base_number))
            save_to = str(base_path / f"{base_number}.1")
        else:
            # Load from latest sub-version and increment version
            parts = latest_sub.split('.')
            version = int(parts[1])
            next_version = version + 1
            load_from = str(base_path / latest_sub)
            save_to = str(base_path / f"{base_number}.{next_version}")

        return load_from, save_to
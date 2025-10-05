from pathlib import Path
from typing import Optional, List, Union
import logging

logger = logging.getLogger(__name__)


class NumericalFileAccessHelper:
    """
    Helper class for loading and accessing files by numerical indexing.
    storage pattern, where files/directories are stored with numerical names
    and can be accessed by their number id, sub id.
    """

    def __init__(self, base_directory: Union[str, Path]):
        """
        Initialize the helper with a base directory.

        Args:
            base_directory: The base directory containing numbered items
        """
        self.base_directory = str(base_directory)

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

    def get_next_sub_version_from_id(self, checkpoint_id: str) -> tuple[str, str]:
        """
        Get the next sub-version paths based on a given checkpoint ID.

        This allows continuing from any checkpoint (base or sub-version).
        - From "10" → loads "10", saves to "10.1"
        - From "10.1" → loads "10.1", saves to "10.2"
        - From "10.5" → loads "10.5", saves to "10.6"

        Args:
            checkpoint_id: The checkpoint to continue from (e.g., "10", "10.1")

        Returns:
            Tuple of (path_to_load_from, path_to_save_to)
        """
        base_path = Path(self.base_directory)
        load_from = str(base_path / checkpoint_id)

        # Parse checkpoint_id to determine next version
        parts = checkpoint_id.split('.')

        if len(parts) == 1:
            # Base checkpoint (e.g., "10") → create "10.1"
            save_to = str(base_path / f"{checkpoint_id}.1")
        else:
            # Sub-version (e.g., "10.1") → create next sub-version "10.2"
            base_num = parts[0]
            current_version = int(parts[1])
            next_version = current_version + 1
            save_to = str(base_path / f"{base_num}.{next_version}")

        return load_from, save_to
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
import logging

from app.core.services.trainer.trainer import TrainerService
from app.core.services.data_manager.data_manager import DataManager

logger = logging.getLogger(__name__)

class ModelAnalyzer:
    '''
    Analyze and provide insights on machine learning models.
    '''

    def __init__(self, trainer_service: TrainerService, data_manager: DataManager):
        self.trainer_service = trainer_service
        self.data_manager = data_manager
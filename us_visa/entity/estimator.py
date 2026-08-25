import sys
from pandas import DataFrame
from us_visa.exception import USvisaException
from us_visa.logger import logging



class TargetValueMapping:
    def __init__(self):
        try:
            self.mapping = {
                "Approved": 1,
                "Denied": 0
            }
        except Exception as e:
            raise USvisaException(e, sys) from e

    def _asdict(self) -> dict:
            try:
                return self.__dict__
            except Exception as e:
                raise USvisaException(e, sys) from e

    def reverse_mapping(self):
            mapping_response = self._asdict()
            return dict(zip(mapping_response.values(), mapping_response.keys()))
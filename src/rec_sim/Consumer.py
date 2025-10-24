"""
Created on June 7 08:00:00 2025

@author: isabella pizzuti
"""


class Consumer:
    def __init__(self, id, dem):
        """

        :param id: str --> identification code  e.g.: 'consumer1'
        :param dem: dict --> load curve  e.g  {'electricity': [0,0,...],'heat': [0,0,...]}
        """

        self.id = id
        self.dem = dem
        self.en_perf_evolution=dem






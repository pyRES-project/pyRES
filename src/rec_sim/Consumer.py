"""
Created on June 7 08:00:00 2025

@author: isabella pizzuti
"""

import numpy as np


class Consumer:
    def __init__(self, id, dem):
        """

        :param id: str --> identification code  e.g.: 'consumer1'
        :param dem: dict --> load curve  e.g  {'electricity': [0,0,...],'heat': [0,0,...]}
        """

        self.id = id
        self.dem = dem
        # [FIX #2] Copia difensiva: en_perf_evolution è una copia indipendente di dem,
        # non un alias. Questo evita che modifiche accidentali a en_perf_evolution
        # corrompano i dati originali di domanda.
        self.en_perf_evolution = {k: np.array(v, dtype=float) for k, v in dem.items()}

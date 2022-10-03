import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from policy import Policy


class ShockwavePolicy(Policy):
    def __init__(self):
        self._name = "shockwave"

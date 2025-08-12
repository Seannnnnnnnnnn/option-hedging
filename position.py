from option import Option


class OptionsPosition: 
    """Represents a position in an (Option) Positon"""

    def __init__(self, instrument: Option, notional: int):
        self.notional = notional
        self.instrument = instrument

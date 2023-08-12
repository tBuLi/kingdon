import json

import numpy as np


class MultiVectorEncoder(json.JSONEncoder):
    def default(self, item):
        from kingdon import MultiVector
        if isinstance(item, MultiVector):
            item = item.asfullmv()
            return {'mv': list(item.values())}

        if isinstance(item, np.integer):
            return int(item)

        # Let the base class default method raise the TypeError
        return super().default(item)

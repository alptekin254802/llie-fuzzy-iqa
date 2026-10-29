"""Five-level quality scoring shared by inference and stored-logit verification."""
import numpy as np
TOKEN_IDS=[1550,1661,6624,7852,3347]
WEIGHTS=np.array([1,.75,.5,.25,0],dtype=np.float64)
def quality(logits):
    z=np.asarray(logits,dtype=np.float64)
    if z.shape!=(5,) or not np.isfinite(z).all():
        raise FloatingPointError('Expected five finite quality logits')
    p=np.exp(z-z.max());p/=p.sum()
    return float(p@WEIGHTS)

"""Only the auxiliary weights change in this evidence-driven second stage."""
import importlib.util
from common import PARENT
spec=importlib.util.spec_from_file_location('audited_reconstruction_impl',PARENT/'model.py')
impl=importlib.util.module_from_spec(spec);spec.loader.exec_module(impl)
Reconstruction,make_adapter,mask_generator=impl.Reconstruction,impl.make_adapter,impl.mask_generator
CONFIGS={
    'masked3':dict(mask=True,weight=3.,ratio=.5),
    'masked10':dict(mask=True,weight=10.,ratio=.5),
    'unmasked3':dict(mask=False,weight=3.,ratio=.5),
    'unmasked10':dict(mask=False,weight=10.,ratio=.5),
}


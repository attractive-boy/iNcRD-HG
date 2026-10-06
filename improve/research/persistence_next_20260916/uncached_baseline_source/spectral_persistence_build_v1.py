"""Build an audited CPU Torch extension with the existing compiler, without Ninja."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

import torch


def load_extension(name, source, build, *, headers=(), optimization='-O2'):
    from setuptools import Distribution
    from torch.utils.cpp_extension import BuildExtension, CppExtension, library_paths
    if not name.isidentifier() or optimization not in ('-O0', '-O2'):
        raise ValueError('Invalid extension build configuration')
    source, build = Path(source).resolve(), Path(build).resolve()
    build.mkdir(parents=True, exist_ok=True)
    manifest = build / 'BUILD.json'
    recipe = dict(name=name, torch=torch.__version__, python=platform.python_version(),
        cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI), optimization=optimization,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [source] + [Path(h).resolve() for h in headers]})
    with (build / 'build.lock').open('w') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        old = json.loads(manifest.read_text()) if manifest.exists() else None
        binary = Path(old['binary']) if old and old.get('recipe') == recipe else None
        if binary is None or not binary.exists() or hashlib.sha256(binary.read_bytes()).hexdigest() != old['binary_sha256']:
            extension = CppExtension(name, [str(source)], extra_compile_args=[optimization, '-g0'],
                extra_link_args=['-Wl,-rpath,' + path for path in library_paths(device_type='cpu')])
            distribution = Distribution(dict(name=name, ext_modules=[extension],
                cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)}))
            command = distribution.get_command_obj('build_ext')
            command.ensure_finalized()
            command.build_lib = str(build)
            command.build_temp = str(build / 'objects')
            command.inplace = False
            command.force = True
            command.run()
            binary = Path(command.get_ext_fullpath(name)).resolve()
            record = dict(recipe=recipe, binary=str(binary),
                          binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
            manifest.write_text(json.dumps(record, indent=2) + '\n')
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    spec = importlib.util.spec_from_file_location(name, binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module

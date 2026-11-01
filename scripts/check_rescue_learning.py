"""Run the CPU learning/protocol suite from a clean checkout on Windows or Linux."""
import os
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]


def main():
    (ROOT/'tmp').mkdir(exist_ok=True)
    suffix='.exe' if os.name=='nt' else ''
    binaries=ROOT/'rust/target/release'
    os.environ.setdefault('PURUMOGU_RESCUE_SERVER',str(binaries/('rescue_server'+suffix)))
    os.environ.setdefault('PURUMOGU_GEOMETRY_PREPARER',str(binaries/('prepare_geometry'+suffix)))
    os.environ.setdefault('PURUMOGU_TSDF_BINARIES',str(binaries))
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_rescue*.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':sys.exit(main())

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(ext_modules=cythonize('oldcpotlib.pyx','oldcpotlib'),include_dirs=[numpy.get_include(),'/Users/Giuliano/gipsy/exe/apple_i64'])

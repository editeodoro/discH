from setuptools import setup
import shutil
import os
from Cython.Distutils import build_ext
from distutils.core import Extension
from Cython.Build import cythonize
import sysconfig
import numpy

def get_ext_filename_without_platform_suffix(filename):
    name, ext = os.path.splitext(filename)
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

    if ext_suffix == ext:
        return filename

    ext_suffix = ext_suffix.replace(ext, '')
    idx = name.find(ext_suffix)

    if idx == -1:
        return filename
    else:
        return name[:idx] + ext


class BuildExtWithoutPlatformSuffix(build_ext):
    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename(ext_name)
        return get_ext_filename_without_platform_suffix(filename)

cy=['discH/src/cpotlib.pyx']
cy_ext=Extension('discH/src/cpotlib',sources=cy)

gh=['discH/src/pot_halo/pot_c_ext/general_halo.pyx']
gh_ext=Extension('discH/src/pot_halo/pot_c_ext/general_halo',sources=gh)

ih=['discH/src/pot_halo/pot_c_ext/isothermal_halo.pyx']
ih_ext=Extension('discH/src/pot_halo/pot_c_ext/isothermal_halo',sources=ih)

ext_modules=cythonize([cy_ext,gh_ext,ih_ext])

setup(
		name='discH',
		version='0.0.3dev0',
		author='Giuliano Iorio',
		author_email='',
		url='',
        cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
		packages=['discH','discH/src','discH/src/pot_halo','discH/src/pot_halo/pot_c_ext','discH/src/pardo'],
        ext_modules=ext_modules,
        include_dirs=[numpy.get_include()]
)



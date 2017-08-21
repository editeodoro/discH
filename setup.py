from setuptools import setup
import shutil
import os
from Cython.Distutils import build_ext
from distutils.core import Extension
from Cython.Build import cythonize
import sysconfig
import numpy
import cython_gsl

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

infw=['discH/src/pot_halo/pot_c_ext/nfw_halo.pyx']
infw_ext=Extension('discH/src/pot_halo/pot_c_ext/nfw_halo',sources=infw)

iab=['discH/src/pot_halo/pot_c_ext/alfabeta_halo.pyx']
iab_ext=Extension('discH/src/pot_halo/pot_c_ext/alfabeta_halo',sources=iab,libraries=cython_gsl.get_libraries(),library_dirs=[cython_gsl.get_library_dir()],include_dirs=[cython_gsl.get_cython_include_dir()])

ph=['discH/src/pot_halo/pot_c_ext/plummer_halo.pyx']
ph_ext=Extension('discH/src/pot_halo/pot_c_ext/plummer_halo',sources=ph)


gd=['discH/src/pot_disc/pot_c_ext/integrand_functions.pyx']
gd_ext=Extension('discH/src/pot_disc/pot_c_ext/integrand_functions',libraries=cython_gsl.get_libraries(),library_dirs=[cython_gsl.get_library_dir()],include_dirs=[cython_gsl.get_cython_include_dir(), numpy.get_include()],sources=gd)

rd=['discH/src/pot_disc/pot_c_ext/rdens_law.pyx']
rd_ext=Extension('discH/src/pot_disc/pot_c_ext/rdens_law',sources=rd)

fd=['discH/src/pot_disc/pot_c_ext/rflare_law.pyx']
fd_ext=Extension('discH/src/pot_disc/pot_c_ext/rflare_law',sources=fd)

pd=['discH/src/pot_disc/pot_c_ext/potential_disc.pyx']
pd_ext=Extension('discH/src/pot_disc/pot_c_ext/potential_disc',sources=pd)

#ext_modules=cythonize([cy_ext,gh_ext,ih_ext,infw_ext,gd_ext,rd_ext,fd_ext])

#extra_compile_args = ['-std=c99']
#sturct_c_src=['discH/src/pot_disc/pot_c_ext/struct.c']
#struct_c_ext = Extension('discH/src/pot_disc/pot_c_ext/struct',
                     #sources=sturct_c_src,
                     #extra_compile_args=extra_compile_args
                     #)


ext_modules=cythonize([cy_ext,gh_ext,ih_ext,infw_ext,gd_ext,rd_ext,fd_ext,pd_ext,iab_ext,ph_ext])

setup(
		name='discH',
		version='2.0.1.dev0',
		author='Giuliano Iorio',
		author_email='',
		url='',
        cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
		packages=['discH','discH/src','discH/src/pot_halo','discH/src/pot_halo/pot_c_ext','discH/src/pardo','discH/src/pot_disc', 'discH/src/pot_disc/pot_c_ext', 'discH/src/galpotential', 'discH/src/discHeigth', 'discH/src/discHeigth/c_ext' , 'discH/src/fitlib' ],
        ext_modules=ext_modules,
        include_dirs=[numpy.get_include(),cython_gsl.get_include()],
        install_requires=['numpy>=1.9', 'scipy>=0.19', 'matplotlib', 'cython', 'CythonGSL']
)

shutil.rmtree('build')
shutil.rmtree('dist')
shutil.rmtree('discH.egg-info')


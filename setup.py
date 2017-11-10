from __future__ import print_function
import pip
import time

#Check Cython installation
print('Checking Ctyhon')
try:
    import Cython
    print('OK!')
except:
    print('Cython is not present, I will install it for you my lord')
    pip.main(['install','Cython'])

#Check CythonGSL installation
print('Checking CtyhonGSL')

try:
    import cython_gsl
    print('OK!')
except:
    print('Cython is not present, I will install it for you my lord')
    pip.main(['install','CythonGSL'])

from setuptools import setup
import shutil
import os
from Cython.Distutils import build_ext
from distutils.core import Extension
from Cython.Build import cythonize
import sysconfig
import numpy
import cython_gsl
import sys

if sys.version_info[0]==2:
    #time.sleep(5)
    cmdclass_option = {}
    print('You are using Python2, what a shame!')
    #raise ValueError('You are using Python2, what a shame! Download Python3 to use this module. \n If you are using anaconda you can install a python3 virtual env just typing:\n "conda create -n yourenvname python=3.6 anaconda". \n Then you can activate the env with the bash command  "source activate yourenvname"')

elif sys.version_info[0]==3:
    print('You are using Python3, you are a wise person!')

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
    cmdclass_option = {'build_ext': BuildExtWithoutPlatformSuffix}
else:
    raise ValueError('You are not using neither Python2 nor Python3, probably you are a time traveller from the Future or from the Past')



gth=['discH/src/pot_halo/pot_c_ext/general_triaxial_halo.pyx']
gth_ext=Extension('discH/src/pot_halo/pot_c_ext/general_triaxial_halo',sources=gth,include_dirs=[numpy.get_include(),cython_gsl.get_include()])

tdp=['discH/src/pot_halo/pot_c_ext/triaxial_doublepower_halo.pyx']
tdp_ext=Extension('discH/src/pot_halo/pot_c_ext/triaxial_doublepower_halo',sources=tdp,libraries=cython_gsl.get_libraries(),library_dirs=[cython_gsl.get_library_dir()],include_dirs=[cython_gsl.get_cython_include_dir(),numpy.get_include()])

tex=['discH/src/pot_halo/pot_c_ext/triaxial_exponential_halo.pyx']
tex_ext=Extension('discH/src/pot_halo/pot_c_ext/triaxial_exponential_halo',sources=tex,libraries=cython_gsl.get_libraries(),library_dirs=[cython_gsl.get_library_dir()],include_dirs=[cython_gsl.get_cython_include_dir(),numpy.get_include()])

gh=['discH/src/pot_halo/pot_c_ext/general_halo.pyx']
gh_ext=Extension('discH/src/pot_halo/pot_c_ext/general_halo',sources=gh,include_dirs=[numpy.get_include(),cython_gsl.get_include()])

ih=['discH/src/pot_halo/pot_c_ext/isothermal_halo.pyx']
ih_ext=Extension('discH/src/pot_halo/pot_c_ext/isothermal_halo',sources=ih,include_dirs=[numpy.get_include(),cython_gsl.get_include()])

infw=['discH/src/pot_halo/pot_c_ext/nfw_halo.pyx']
infw_ext=Extension('discH/src/pot_halo/pot_c_ext/nfw_halo',sources=infw,include_dirs=[numpy.get_include(),cython_gsl.get_include()])

iab=['discH/src/pot_halo/pot_c_ext/alfabeta_halo.pyx']
iab_ext=Extension('discH/src/pot_halo/pot_c_ext/alfabeta_halo',sources=iab,libraries=cython_gsl.get_libraries(),library_dirs=[cython_gsl.get_library_dir()],include_dirs=[cython_gsl.get_cython_include_dir(),numpy.get_include()])

ph=['discH/src/pot_halo/pot_c_ext/plummer_halo.pyx']
ph_ext=Extension('discH/src/pot_halo/pot_c_ext/plummer_halo',sources=ph,include_dirs=[numpy.get_include(),cython_gsl.get_include()])

eh=['discH/src/pot_halo/pot_c_ext/einasto_halo.pyx']
eh_ext=Extension('discH/src/pot_halo/pot_c_ext/einasto_halo',sources=eh,libraries=cython_gsl.get_libraries(),library_dirs=[cython_gsl.get_library_dir()],include_dirs=[cython_gsl.get_cython_include_dir(),numpy.get_include()])

pch=['discH/src/pot_halo/pot_c_ext/powercut_halo.pyx']
pch_ext=Extension('discH/src/pot_halo/pot_c_ext/powercut_halo',sources=pch,libraries=cython_gsl.get_libraries(),library_dirs=[cython_gsl.get_library_dir()],include_dirs=[cython_gsl.get_cython_include_dir(),numpy.get_include()])

gd=['discH/src/pot_disc/pot_c_ext/integrand_functions.pyx']
gd_ext=Extension('discH/src/pot_disc/pot_c_ext/integrand_functions',libraries=cython_gsl.get_libraries(),library_dirs=[cython_gsl.get_library_dir()],include_dirs=[cython_gsl.get_cython_include_dir(), numpy.get_include()],sources=gd)

rd=['discH/src/pot_disc/pot_c_ext/rdens_law.pyx']
rd_ext=Extension('discH/src/pot_disc/pot_c_ext/rdens_law',sources=rd,include_dirs=[numpy.get_include(),cython_gsl.get_include()])

fd=['discH/src/pot_disc/pot_c_ext/rflare_law.pyx']
fd_ext=Extension('discH/src/pot_disc/pot_c_ext/rflare_law',sources=fd,include_dirs=[numpy.get_include(),cython_gsl.get_include()])

zd=['discH/src/pot_disc/pot_c_ext/zdens_law.pyx']
zd_ext=Extension('discH/src/pot_disc/pot_c_ext/zdens_law',sources=zd,include_dirs=[numpy.get_include(),cython_gsl.get_include()])

vcirc=['discH/src/pot_disc/pot_c_ext/integrand_vcirc.pyx']
vcirc_ext=Extension('discH/src/pot_disc/pot_c_ext/integrand_vcirc', sources=vcirc,include_dirs=[numpy.get_include(),cython_gsl.get_include()])

#ext_modules=cythonize([cy_ext,gh_ext,ih_ext,infw_ext,gd_ext,rd_ext,fd_ext])

#extra_compile_args = ['-std=c99']
#sturct_c_src=['discH/src/pot_disc/pot_c_ext/struct.c']
#struct_c_ext = Extension('discH/src/pot_disc/pot_c_ext/struct',
                     #sources=sturct_c_src,
                     #extra_compile_args=extra_compile_args
                     #)

ext_modules=cythonize([gth_ext,tdp_ext,tex_ext,gh_ext,ih_ext,infw_ext,gd_ext,rd_ext,fd_ext,iab_ext,ph_ext,eh_ext,pch_ext,zd_ext,vcirc_ext,])



setup(
        name='discH',
        version='3.1.0.dev0',
        author='Giuliano Iorio',
        author_email='',
        url='',
        cmdclass=cmdclass_option,
        packages=['discH','discH/src','discH/src/pot_halo','discH/src/pot_halo/pot_c_ext','discH/src/pardo','discH/src/pot_disc', 'discH/src/pot_disc/pot_c_ext', 'discH/src/galpotential', 'discH/src/discHeight', 'discH/src/discHeight/c_ext' , 'discH/src/fitlib' ],
        ext_modules=ext_modules,
        include_dirs=[numpy.get_include(),cython_gsl.get_include()],
        install_requires=['numpy>=1.9', 'scipy>=0.19', 'matplotlib','emcee']
)

"""
try:
    shutil.rmtree('build')
    shutil.rmtree('dist')
    shutil.rmtree('discH.egg-info')
except:
    pass
"""

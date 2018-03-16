from __future__ import print_function
import pip, time

def checkModule(module):
    print('Checking %s '%module,end='')
    try:
        __import__(module)
        print ("OK!")
    except ImportError:
        print("Module '%s' is not present, I will install it for you my lord."%module)
        pip.main(['install',module])
        

modules = ['scipy','Cython','cython_gsl', 'emcee']
 
for m in modules: 
    checkModule(m)

#Check Scipy>1.0 installation
print('Checking Scipy>1.0')
import scipy
scv=scipy.__version__
scvl=scv.split('.')
if int(scvl[0])>0 or int(scvl[1])>19:
    print('OK! (Version %s)'%scv)
else:
    print('Version %s too old. I will install the latest version' % scv)
    pip.main(['install','scipy','--upgrade'])


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
    cmdclass_option = {}
    print('You are using Python2, what a shame!')

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

#cython gsl
cy_gsl_lib=cython_gsl.get_libraries()
cy_gsl_inc=cython_gsl.get_include()
cy_gsl_lib_dic=cython_gsl.get_library_dir()
#cython
cy_gsl_inc_cy=cython_gsl.get_cython_include_dir()
#numpy
np_inc=numpy.get_include()

cfiles = ['discH/src/pot_halo/pot_c_ext/general_triaxial_halo.pyx', 'discH/src/pot_halo/pot_c_ext/triaxial_doublepower_halo.pyx',\
          'discH/src/pot_halo/pot_c_ext/triaxial_exponential_halo.pyx', 'discH/src/pot_halo/pot_c_ext/general_halo.pyx',\
          'discH/src/pot_halo/pot_c_ext/isothermal_halo.pyx','discH/src/pot_halo/pot_c_ext/nfw_halo.pyx',\
          'discH/src/pot_halo/pot_c_ext/alfabeta_halo.pyx', 'discH/src/pot_halo/pot_c_ext/plummer_halo.pyx',\
          'discH/src/pot_halo/pot_c_ext/einasto_halo.pyx', 'discH/src/pot_halo/pot_c_ext/powercut_halo.pyx',\
          'discH/src/pot_halo/pot_c_ext/exponential_halo.pyx','discH/src/pot_halo/pot_c_ext/valy_halo.pyx',\
          'discH/src/pot_disc/pot_c_ext/integrand_functions.pyx','discH/src/pot_disc/pot_c_ext/rdens_law.pyx',\
          'discH/src/pot_disc/pot_c_ext/rflare_law.pyx','discH/src/pot_disc/pot_c_ext/zdens_law.pyx',\
          'discH/src/pot_disc/pot_c_ext/integrand_vcirc.pyx']

incs = [numpy.get_include(),cython_gsl.get_include(),cython_gsl.get_cython_include_dir()]
libs = cython_gsl.get_libraries()
ldir = [cython_gsl.get_library_dir()]

mods = []
for c in cfiles:
    c_ext = Extension(c[:-4],sources=[c],include_dirs=incs,library_dirs=ldir,libraries=libs)
    mods.append(c_ext)

ext_modules=cythonize(mods)

setup(
        name='discH',
        version='3.1.0.dev0',
        author='Giuliano Iorio',
        author_email='',
        url='',
        cmdclass=cmdclass_option,
        packages=['discH','discH/src','discH/src/pot_halo','discH/src/pot_halo/pot_c_ext','discH/src/pardo','discH/src/pot_disc', 'discH/src/pot_disc/pot_c_ext', 'discH/src/galpotential', 'discH/src/discHeight', 'discH/src/discHeight/c_ext' , 'discH/src/fitlib' ],
        ext_modules=ext_modules,
        include_dirs=[np_inc,cython_gsl.get_include()],
        install_requires=['numpy>=1.9', 'scipy>=0.19', 'matplotlib','emcee']
)


'''
try:
    shutil.rmtree('build')
    shutil.rmtree('dist')
    shutil.rmtree('discH.egg-info')
except:
    pass
'''

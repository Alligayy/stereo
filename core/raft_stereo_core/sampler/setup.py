from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os


code_dir = os.path.dirname(os.path.realpath(__file__))
# print(code_dir)
# exit()

extra_compile_flags = {}
gencodes = ['-arch=sm_50',
            '-gencode', 'arch=compute_50,code=sm_50',
            '-gencode', 'arch=compute_52,code=sm_52',
            '-gencode', 'arch=compute_60,code=sm_60',
            '-gencode', 'arch=compute_61,code=sm_61',
            '-gencode', 'arch=compute_70,code=sm_70',
            '-gencode', 'arch=compute_75,code=sm_75',
            '-gencode', 'arch=compute_75,code=compute_75',]

# extra_compile_flags['nvcc'] = gencodes

setup(
    name='corr_sampler',
    ext_modules=[
        CUDAExtension('corr_sampler', [
            os.path.join(code_dir, 'sampler.cpp'),
            os.path.join(code_dir, 'sampler_kernel.cu'),
        ],
      extra_compile_args=extra_compile_flags)
    ],
    cmdclass={
        'build_ext': BuildExtension
    })



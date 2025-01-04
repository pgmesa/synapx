import os
from setuptools import setup
from wheel.bdist_wheel import bdist_wheel

# Custom wheel class to set platform tags
class BDistWheelCustom(bdist_wheel):
    def finalize_options(self):
        bdist_wheel.finalize_options(self)
        self.root_is_pure = False

here = os.path.abspath(os.path.dirname(__file__))

setup(
    cmdclass={
        'bdist_wheel': BDistWheelCustom,
    },
)
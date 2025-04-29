from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'f1tenth_rl'

setup(
    name=package_name,
    version='0.2.0',
    packages=[
        package_name,
        package_name + '.agents',
        package_name + '.utils'
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'torch',
        'matplotlib',
        'transforms3d'  # Use transforms3d instead of tf_transformations
    ],
    zip_safe=True,
    maintainer='F1TENTH Team',
    maintainer_email='maintainer@f1tenth.org',
    description='ROS2 package for reinforcement learning with F1TENTH simulator',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rl_agent = f1tenth_rl.rl_agent_node:main',
        ],
    },
)

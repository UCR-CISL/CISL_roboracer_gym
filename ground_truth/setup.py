from setuptools import setup
import os
from glob import glob

package_name = 'ground_truth'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ground_truth_node = ground_truth.ground_truth_node:main',
            'wall_follower = ground_truth.wall_follower_node:main',
            'learner_node = ground_truth.learner_node:main',
            'train = ground_truth.train:main',
            'inference = ground_truth.inference:main',
            'dagger_logger = ground_truth.dagger_logger:main',

        ],
    },
)

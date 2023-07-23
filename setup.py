from setuptools import setup

setup(name='rl_algorithms',
      version='0.01',
      description='Package contains the reinforcement learning algorithms',
      url='https://github.com/ahadjawaid/rl-algorithms',
      author='Ahad Jawaid',
      packages=['rl_algorithms', 'rl_algorithms.mab', 'rl_algorithms.dp'],
      author_email='',
      license='MIT License',
      install_requires=['gym>=0.2.3'],
)
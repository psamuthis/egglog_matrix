import unittest
import sys
from pathlib import Path


loader = unittest.TestLoader()
runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)
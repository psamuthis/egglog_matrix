import unittest

loader = unittest.TestLoader()
suite = loader.discover('test')
runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)
import sys
import types


if sys.platform == 'win32':
  for name in ('fcntl', 'termios'):
    sys.modules.setdefault(name, types.ModuleType(name))

  params_pyx = types.ModuleType('openpilot.common.params_pyx')

  class Params:
    def __init__(self, *args, **kwargs):
      pass

    def get(self, key, block=False, return_default=False):
      return None

  class ParamKeyFlag:
    ALL = 0

  class ParamKeyType:
    STRING = 0

  class UnknownKeyName(Exception):
    pass

  params_pyx.Params = Params
  params_pyx.ParamKeyFlag = ParamKeyFlag
  params_pyx.ParamKeyType = ParamKeyType
  params_pyx.UnknownKeyName = UnknownKeyName
  sys.modules.setdefault('openpilot.common.params_pyx', params_pyx)

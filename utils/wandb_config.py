from collections import OrderedDict

# six >=1.13.0 not available in all environmnents yet
#from six.moves.collections_abc import Sequence
try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import inspect
import os
import sys
import six
import types
import yaml
import platform

import numpy as np
from datetime import date, datetime

FNAME = 'config.yaml'

def boolify(s):
    if s.lower() == 'none':
        return None
    if s.lower() == 'true':
        return True
    if s.lower() == 'false':
        return False
    raise ValueError("Not a boolean")

def is_kaggle():
    return os.getenv("KAGGLE_KERNEL_RUN_TYPE") != None or "kaggle_environments" in sys.modules

class Config(object):
    """Creates a W&B config object."""

    def __init__(self, config_path):
        # OrderedDict to make writing unit tests easier. (predictable order for
        # .key())
        object.__setattr__(self, '_items', OrderedDict())
        object.__setattr__(self, '_descriptions', {})

        self._load_file(config_path)

        # Do this after defaults because it triggers loading of pre-existing
        # config.yaml (if it exists)

        #self.persist()

    def _telemetry_update(self):
        """Add telemetry data to internal config structure."""
        updated = False

        # detect framework by checking what is loaded
        loaded = {}
        loaded['lightgbm'] = sys.modules.get('lightgbm')
        loaded['xgboost'] = sys.modules.get('xgboost')
        loaded['fastai'] = sys.modules.get('fastai')
        loaded['torch'] = sys.modules.get('torch')
        loaded['keras'] = sys.modules.get('keras')  # vanilla keras
        loaded['tensorflow'] = sys.modules.get('tensorflow')
        loaded['sklearn'] = sys.modules.get('sklearn')
        # TODO(jhr): tfkeras is always loaded with recent tensorflow
        #loaded['tfkeras'] = sys.modules.get('tensorflow.python.keras')

        priority = ('lightgbm', 'xgboost', 'fastai', 'torch', 'keras', 'tfkeras', 'tensorflow', 'sklearn')
        framework = next((f for f in priority if loaded.get(f)), None)
        if framework:
            self._set_wandb('framework', framework)
            updated = True

        return updated

    # @classmethod
    # def from_environment_or_defaults(cls):
    #     conf_paths = os.environ.get(env.CONFIG_PATHS, [])
    #     run_dir = os.environ.get(env.RUN_DIR)
    #     if conf_paths:
    #         conf_paths = conf_paths.split(',')
    #     return Config(config_paths=conf_paths, wandb_dir=wandb.wandb_dir(), run_dir=run_dir)

    def _load_defaults(self):
        defaults_path = os.path.join('config-defaults.yaml')
        self._load_file(defaults_path)

    def _load_file(self, path):
        subkey = None
        if '::' in path:
            conf_path, subkey = path.split('::', 1)
        else:
            conf_path = path
        try:
            conf_file = open(conf_path)
        except (OSError, IOError):
            raise ValueError('Couldn\'t read config file: %s' % conf_path)
        try:
            loaded = yaml.full_load(conf_file)
        except yaml.parser.ParserError:
            raise ValueError('Invalid YAML in config-defaults.yaml')
        if subkey:
            try:
                loaded = loaded[subkey]
            except KeyError:
                raise ValueError('Asked for {} but {} not present in {}'.format(
                    path, subkey, conf_path))
        for key, val in loaded.items():
            if key == 'wandb_version':
                continue
            if isinstance(val, dict):
                if 'value' not in val:
                    raise ValueError('In config {} value of {} is dict, but does not contain "value" key'.format(
                        path, key))
                self._items[key] = val['value']
                if 'desc' in val:
                    self._descriptions[key] = val['desc']
            else:
                self._items[key] = val

    def _load_values(self):
        """Load config.yaml from the run directory if available."""
        path = self._config_path()
        if path is not None and os.path.isfile(path):
            self._load_file(path)

    def _config_path(self):
        if self._run_dir and os.path.isdir(self._run_dir):
            return os.path.join(self._run_dir, FNAME)
        return None

    def keys(self):
        """All keys in the current configuration"""
        return [k for k in self._items.keys() if k != '_wandb']

    def desc(self, key):
        """The description of a given key"""
        return self._descriptions.get(key)

    def load_json(self, json):
        """Loads existing config from JSON"""
        for key in json:
            if key == "wandb_version":
                continue
            self._items[key] = json[key].get('value')
            self._descriptions[key] = json[key].get('desc')

    def set_run_dir(self, run_dir):
        """Set the run directory to which this Config should be persisted.

        Changes to this Config won't be written anywhere unless the run directory
        is set.
        """
        object.__setattr__(self, '_run_dir', run_dir)
        self._load_values()

    def persist(self):
        """Stores the current configuration for pushing to W&B"""
        # In dryrun mode, without wandb run, we don't
        # save config  on initial load, because the run directory
        # may not be created yet (because we don't know if we're
        # being used in a run context, or as an API).
        # TODO: Defer saving somehow, maybe via an events system
        path = self._config_path()
        if path is None:
            return
        with open(path, "w") as conf_file:
            conf_file.write(str(self))

    def get(self, *args):
        return self._items.get(*args)

    def __getitem__(self, key):
        return self._items[key]

    def __setitem__(self, key, val):
        key, val = self._sanitize(key, val)
        self._items[key] = val
        self.persist()

    __setattr__ = __setitem__

    def __getattr__(self, key):
        return self.__getitem__(key)

    def _sanitize(self, key, val, allow_val_change=False):
        # We always normalize keys by stripping '-'
        key = key.strip('-')
        val = self._sanitize_val(val)
        if not allow_val_change:
            if key in self._items and val != self._items[key]:
                raise ValueError('Attempted to change value of key "{}" from {} to {}\nIf you really want to do this, pass allow_val_change=True to config.update()'.format(
                    key, self._items[key], val))
        return key, val

    def _sanitize_val(self, val):
        """Turn all non-builtin values into something safe for YAML"""
        if isinstance(val, dict):
            converted = {}
            for key, value in six.iteritems(val):
                converted[key] = self._sanitize_val(value)
            return converted
        val, _ = json_friendly(val)
        if isinstance(val, Sequence) and not isinstance(val, six.string_types):
            converted = []
            for value in val:
                converted.append(self._sanitize_val(value))
            return converted
        else:
            if val.__class__.__module__ not in ('builtins', '__builtin__'):
                val = str(val)
            return val

    def _update(self, params, allow_val_change=False, as_defaults=False, exclude_keys=None, include_keys=None):
        exclude_keys = exclude_keys or []
        include_keys = include_keys or []
        params = params or {}
        if not isinstance(params, dict):
            # Handle some cases where params is not a dictionary
            # by trying to convert it into a dictionary
            meta = inspect.getmodule(params)
            if meta:
                is_tf_flags_module = isinstance(
                    params, types.ModuleType) and meta.__name__ == 'tensorflow.python.platform.flags'
                if is_tf_flags_module or meta.__name__ == 'absl.flags':
                    params = params.FLAGS
                    meta = inspect.getmodule(params)

            # newer tensorflow flags (post 1.4) uses absl.flags
            if meta and meta.__name__ == "absl.flags._flagvalues":
                params = {name: params[name].value for name in dir(params)}
            elif "__flags" in vars(params):
                # for older tensorflow flags (pre 1.4)
                if not '__parsed' in vars(params):
                    params._parse_flags()
                params = vars(params)['__flags']
            elif not hasattr(params, '__dict__'):
                raise TypeError(
                    "config must be a dict or have a __dict__ attribute.")
            else:
                # params is a Namespace object (argparse)
                # or something else
                params = vars(params)

        if not isinstance(params, dict):
            raise ValueError('Expected dict but received %s' % params)
        if exclude_keys and include_keys:
            raise ValueError('Expected at most only one of exclude_keys or include_keys')
        for key, val in params.items():
            if key in exclude_keys:
                continue
            if include_keys and key not in include_keys:
                continue
            key, val = self._sanitize(key, val, allow_val_change=allow_val_change or as_defaults)
            if as_defaults and key in self._items:
                continue
            self._items[key] = val
        self.persist()

    def update(self, params, allow_val_change=False, exclude_keys=None, include_keys=None):
        self._update(params,
                exclude_keys=exclude_keys,
                include_keys=include_keys,
                allow_val_change=allow_val_change)

    def setdefaults(self, params, exclude_keys=None, include_keys=None):
        self._update(params,
                exclude_keys=exclude_keys,
                include_keys=include_keys,
                as_defaults=True)
        return dict(self)

    def setdefault(self, key, default=None):
        key, val = self._sanitize(key, default, allow_val_change=True)
        if key in self._items:
            return self._items[key]
        self._items[key] = val
        self.persist()
        return val

    def as_dict(self):
        defaults = {}
        for key, val in self._items.items():
            defaults[key] = {'value': val,
                             'desc': self._descriptions.get(key)}
        return defaults

    def user_items(self):
        """Retrieve user configured config parameters as a key value tuple generator"""
        for key, val in self._items.items():
            if key != '_wandb':
                yield (key, val)

    def __str__(self):
        s = b"wandb_version: 1"
        as_dict = self.as_dict()
        if as_dict:  # adding an empty dictionary here causes a parse error
            s += b'\n\n' + yaml.dump(as_dict, Dumper=yaml.SafeDumper, default_flow_style=False,
                                     allow_unicode=True, encoding='utf-8')
        return s.decode("utf-8")


class ConfigStatic(object):
    def __init__(self, config):
        object.__setattr__(self, "__dict__", dict(config))

    def __setattr__(self, name, value):
        raise AttributeError("Error: wandb.run.config_static is a readonly object")

    def __setitem__(self, key, val):
        raise AttributeError("Error: wandb.run.config_static is a readonly object")

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __str__(self):
        return str(self.__dict__)







def get_full_typename(o):
    """We determine types based on type names so we don't have to import
    (and therefore depend on) PyTorch, TensorFlow, etc.
    """
    instance_name = o.__class__.__module__ + "." + o.__class__.__name__
    if instance_name in ["builtins.module", "__builtin__.module"]:
        return o.__name__
    else:
        return instance_name

def is_tf_tensor_typename(typename):
    return typename.startswith('tensorflow.') and ('Tensor' in typename or 'Variable' in typename)

def is_tf_eager_tensor_typename(typename):
    return typename.startswith('tensorflow.') and ('EagerTensor' in typename)


def is_pytorch_tensor(obj):
    import torch
    return isinstance(obj, torch.Tensor)


def is_pytorch_tensor_typename(typename):
    return typename.startswith('torch.') and ('Tensor' in typename or 'Variable' in typename)

def is_numpy_array(obj):
    return np and isinstance(obj, np.ndarray)

def json_friendly(obj):
    """Convert an object into something that's more becoming of JSON"""
    converted = True
    typename = get_full_typename(obj)

    if is_tf_eager_tensor_typename(typename):
        obj = obj.numpy()
    elif is_tf_tensor_typename(typename):
        obj = obj.eval()
    elif is_pytorch_tensor_typename(typename):
        try:
            if obj.requires_grad:
                obj = obj.detach()
        except AttributeError:
            pass  # before 0.4 is only present on variables

        try:
            obj = obj.data
        except RuntimeError:
            pass  # happens for Tensors before 0.4

        if obj.size():
            obj = obj.numpy()
        else:
            return obj.item(), True

    if is_numpy_array(obj):
        if obj.size == 1:
            obj = obj.flatten()[0]
        elif obj.size <= 32:
            obj = obj.tolist()
    elif np and isinstance(obj, np.generic):
        obj = obj.item()
    elif isinstance(obj, bytes):
        obj = obj.decode('utf-8')
    elif isinstance(obj, (datetime, date)):
        obj = obj.isoformat()
    else:
        converted = False
    return obj, converted
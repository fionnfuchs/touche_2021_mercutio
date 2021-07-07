import os

import yaml

ENV_MAPPING = {("chat_noir", "api_key"): "CHAT_NOIR_API_KEY"}


class Config:
    def __init__(self, path, env_mapping=None):
        """
            Combines reading configuration from a yaml file and environment variables. If a environment variable is set, it is
            used, otherwise the value will be read from a loaded yaml configuration file.
        :param path: Path to the yaml configuration file
        :param env_mapping: A dict that maps key tuples to their environment variable names (optional)
        """
        if env_mapping is None:
            self._env_mapping = ENV_MAPPING
        self.__path = path
        self._data = self._read_path(path)

    def _read_path(self, path):
        with open(path, "r") as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ConfigException(
                    "Error while reading configuration from %s" % path
                ) from e
        return data

    def get(
        self, *keys, expected_type=None, raise_exc=True, default=None, choices=None
    ):
        """
        Returns a property from the configuration, either from the OS' environment variables or from the yaml configuration
        :param keys:    One or multiple keys that access the property (one per indent level)
        :param expected_type:  if not None, checks if the retrieved key is of the given type. If not an exception is
        raised or the default value is returned (if raise_exc is set to False)
        :param raise_exc: If true, an exception is raised upon unexpected or unintended behaviour
        :param default:  A value that is returned if the property is not found under the given key or has the wrong type
        (if expected_type is set)
        :return:
        """

        def check_choices(val):
            if choices == None:
                return val
            if val in choices:
                return val
            raise ConfigException(
                "Expected configuration property '%s' to be one of: %s, but got %s."
                % (".".join(keys), choices, val)
            )

        env_var_name, env_value = self._get_from_env(*keys)
        if env_value is not None:
            if expected_type is not None:
                try:
                    return check_choices(expected_type(env_value))
                except (ValueError, TypeError) as e:
                    raise ConfigException(
                        "Expected env variable $%s to be convertible to type %s."
                        % (env_var_name, expected_type)
                    ) from e
            else:
                return check_choices(env_value)
        node = self._data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                if raise_exc:
                    raise ConfigException(
                        "Expected %s'%s' to be present in config, but was not found."
                        % (
                            "$%s to be set or " % env_var_name
                            if env_var_name is not None
                            else "",
                            ".".join(keys),
                        )
                    )
                else:
                    return check_choices(default)
            node = node[key]
        if expected_type is not None and type(node) is not expected_type:
            if raise_exc:
                raise ConfigException(
                    "Expected configuration property '%s' to be of type %s, but found %s."
                    % (".".join(keys), expected_type, type(node))
                )
            else:
                return check_choices(default)
        return check_choices(node)

    def _get_from_env(self, *keys):
        """
        checks if the keys can be read from env variables.
        :param keys:
        :return: A tuple with: 1. the env variable name, if a mapping exists and 2. the env variable value if it is set
        """
        as_tuple = tuple(keys)
        if as_tuple not in self._env_mapping:
            return None, None
        var_name = self._env_mapping[as_tuple]
        if var_name not in os.environ:
            return var_name, None
        return var_name, os.environ[var_name]

    def relative_path(self, path: str):
        config_dir = os.path.dirname(self.__path)
        return os.path.join(config_dir, path)


class ConfigException(Exception):
    pass

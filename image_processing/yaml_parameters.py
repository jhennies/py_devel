
from simple_logger import SimpleLogger
import yaml

__author__ = 'jhennies'


class YamlParams(SimpleLogger):

    _yaml = None
    _params = None

    def __init__(self, filename=None):

        if filename is not None:
            self.load_yaml(filename)
            self._yaml = filename
        else:
            pass

    def load_yaml(self, filename):

        with open(filename, 'r') as f:
            readict = yaml.load(f)

        self._params = readict

        return readict

    def get_params(self):
        return self._params

    def set_params(self, params):
        self._params = params

    def yaml2log(self, filepath=None):
        if filepath is None:
            filepath = self._yaml

        with open(filepath, 'r') as f:
            yaml_data = f.read()
        self.logging('>>> YAML >>>\n{}\n<<< YAML <<<', yaml_data)


if __name__ == '__main__':
    pass


from simple_logger import SimpleLogger
import yaml

__author__ = 'jhennies'


class YamlParams(SimpleLogger):

    _filename = None
    _params = None

    def __init__(self, filename=None, yaml=None):

        SimpleLogger.__init__(self)

        if filename is not None:
            self.load_yaml(filename)
            self._filename = filename
        elif yaml is not None:
            self.set_params(yaml.get_params())
            self.set_filename(yaml.get_filename())
            self.set_name(yaml.get_name())
            self.set_logger(yaml.get_logger())

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

    def get_filename(self):
        return self._filename

    def set_filename(self, filename):
        self._filename = filename

    def yaml2log(self, filepath=None):
        if filepath is None:
            filepath = self._filename

        with open(filepath, 'r') as f:
            yaml_data = f.read()
        self.logging('>>> YAML >>>\n{}\n<<< YAML <<<', yaml_data)


if __name__ == '__main__':
    pass

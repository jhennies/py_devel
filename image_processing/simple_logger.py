
import time
import sys
import traceback
import re


__author__ = 'jhennies'


class SimpleLogger():

    _logger = None
    _name = None
    _indent = 0
    _indentstr = '    '

    def __init__(self, name=None):
        if name is not None:
            self._name = name
        else:
            self._name = ''

    def set_logger(self, logger):
        self._logger = logger

    def get_logger(self):
        return self._logger

    def set_name(self, name):
        self._name = name

    def get_name(self):
        return self._name

    def set_indent(self, indent):
        self._indent = indent

    def get_indent(self):
        return self._indent

    def set_indentstr(self, indentstr):
        self._indentstr = indentstr

    def get_indentstr(self):
        return self._indentstr

    def startlogger(self, filename=None, type='a', name=None):

        if name is not None:
            self._name = name

        if filename is not None:
            self._logger = open(filename, type)

        self.logging("Logger started: {}\n", time.strftime('%X %z on %b %d, %Y'), name=True)

    def _insert_indents(self, text):
        indent = self._indentstr * self._indent
        text = re.sub('^', indent, text)
        text = re.sub('\n', '\n' + indent, text)
        return text

    def logging(self, text, *args, **kwargs):

        if 'name' in kwargs:
            name = kwargs.pop('name')
        else:
            name = False

        if type(text) is str:

            # Build the output
            text = text.format(*args)
            if self._name != '' and name:
                text = '{}: {}'.format(self._name, text)

            # Insert indentation at beginning of text and after every occurance of a newline
            text = self._insert_indents(text)

            # Ouput on console
            print text

            # Output to file
            if self._logger is not None:
                text += '\n'
                self._logger.write(text)

            # if self._name != '' and name:
            #     print '{}: {}'.format(self._name, text.format(*args))
            # else:
            #     print text.format(*args)
            # text += "\n"
            # if self._logger is not None:
            #     if self._name != '' and name:
            #         self._logger.write('{}: {}'.format(self._name, text.format(*args)))
            #     else:
            #         self._logger.write(text.format(*args))
        else:
            print text
            if self._logger is not None:
                self._logger.write(str(text))

    def stoplogger(self):

        self.logging("Logger stopped: {}", time.strftime('%X %z on %b %d, %Y'), name=True)

        if self._logger is not None:
            self._logger.close()

    def code2log(self, filename):
        self.logging('>>> CODE >>>')
        with open(filename) as f:
            script = f.read()
        self.logging('{}', script)
        self.logging('<<< CODE <<<')

    def errout(self, name, tb=traceback):
        self.logging('\n{}:\n---------------------------\n', name)
        self.logging('{}', tb.format_exc())
        self.logging('---------------------------')
        self.stoplogger()
        sys.exit()

    def errprint(self, name, tb):
        self.logging('\n{}:\n---------------------------\n', name)
        self.logging('{}', tb.format_exc())
        self.logging('---------------------------')


if __name__ == '__main__':
    pass

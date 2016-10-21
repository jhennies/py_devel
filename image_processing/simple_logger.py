
import time
import sys
import traceback


__author__ = 'jhennies'


class SimpleLogger():

    _logger = None

    def __init__(self):
        pass

    def startlogger(self, filename=None, type='a'):

        if filename is not None:
            self._logger = open(filename, type)

        self.logging("Logger started: {}\n".format(time.strftime('%X %z on %b %d, %Y')))

    def logging(self, format, *args):
        print format.format(*args)
        format += "\n"
        if self._logger is not None:
            self._logger.write(format.format(*args))

    def stoplogger(self):

        self.logging("Logger stopped: {}".format(time.strftime('%X %z on %b %d, %Y')))

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

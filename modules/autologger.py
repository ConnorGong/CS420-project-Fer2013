import sys

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.str=fileN
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.log.flush()

    def __del__(self):
        print(self.str+" file closed")
        self.log.close()

    def stop(self):
        sys.stdout=self.terminal

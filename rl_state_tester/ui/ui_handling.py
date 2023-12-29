import re
from threading import Thread
from typing import List

from flask import Flask
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

from rl_state_tester.global_harvesters.callbacks import Callback

app = Flask("State tester")
CORS(app)
app.secret_key = b'qzjojqkdnqz'


class UIHandler:
    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks
        self.app = app
        self.http_server = None
        self.running = True
        self.ui_thread = Thread(target=self._start)

        def __return_callback(callback: Callback):
            class_name = callback.__class__.__name__
            split_name = re.findall('[A-Z0-9][^A-Z0-9]*', class_name)
            class_name = ""
            path = ""

            for elt in split_name:
                class_name += elt + " "
                path += elt.lower() + "-"
            class_name = class_name[:-1]
            path = path[:-1]

            return {
                'name': class_name,
                'path': path
            }

        @self.app.get("/callbacks")
        def get_callbacks():
            callbacks = {'data': []}
            for c in self.callbacks:
                callbacks['data'].append(__return_callback(c))
            return callbacks


    def _start(self):
        self.http_server = WSGIServer(("localhost", 5000), app)
        print("Server started")
        self.http_server.serve_forever()

    def quit(self):
        self.running = False
        self.ui_thread.join()

    def serve(self):
        self.ui_thread.start()

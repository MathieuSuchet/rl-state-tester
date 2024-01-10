from threading import Thread
from typing import List

from flask import Flask, request
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

from rl_state_tester.global_harvesters.callbacks import Callback
from rl_state_tester.ui.encoders import JSONCallbackInfoEncoder
from rl_state_tester.utils.commands import Hittable

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
        self.callback_info_encoder = JSONCallbackInfoEncoder()
        self.formatted_cbs = [self.callback_info_encoder.default(callback) for callback in self.callbacks]

        @self.app.get("/callbacks")
        def get_callbacks():
            return {'data': self.formatted_cbs}

        @self.app.post("/callbacks")
        def activate_command():
            body = request.json

            def __find_command(c, name):
                command = None
                if not hasattr(c, name):
                    for k, v in c.__dict__.items():
                        if issubclass(v.__class__, Hittable):
                            command = v.get_command(name)
                            if command:
                                break
                        else:
                            if hasattr(v, name):
                                command = getattr(v, name)
                else:
                    command = getattr(c, name)
                return command

            found_callback = None

            for c in self.callbacks:
                if id(c) == body['id']:
                    if body['name'] == "get_cb_data":
                        return {'data': c.to_json()}
                    __find_command(c.commands, body['name']).target()
                    found_callback = c
                    break

            return {'data': found_callback.to_json()}

    def _start(self):
        self.http_server = WSGIServer(("localhost", 5000), app)
        print("Server started")
        self.http_server.serve_forever()

    def quit(self):
        self.running = False
        self.ui_thread.join()

    def serve(self):
        self.ui_thread.start()

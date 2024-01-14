import eventlet
import socketio

sio = socketio.Server()
app = socketio.WSGIApp(sio)

@sio.event
def connect(sid, environ):
    print('connect ', sid)

@sio.event
def my_message(sid, data):
    print('message ', data)

@sio.event
def disconnect(sid):
    print('disconnect ', sid)



class WebsocketService:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def run(self):
        eventlet.wsgi.server(eventlet.listen(('localhost', 5000)), app)

if __name__ == "__main__":
    WebsocketService("localhost", 5000).run()
const io = require("socket.io")

let srv = new io.Server();

srv.on(
    "connect", (socket) => {
        socket.send("Connected!")
    }
)
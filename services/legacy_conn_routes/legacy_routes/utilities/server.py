import socket

def get_open_tcp_port():
    sck = socket.socket()
    sck.bind(('', 0))

    port = sck.getsockname()[1]
    sck.close()
    return port
from tornado import websocket, web, ioloop
import json
import pg8000
import numpy as np
import scipy as sp

cl = []

class IndexHandler(web.RequestHandler):
    def get(self):
        self.render("index.html")

class SocketHandler(websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        if self not in cl:
            cl.append(self)

    def on_close(self):
        if self in cl:
            cl.remove(self)

class ApiHandler(web.RequestHandler):

    @web.asynchronous
    def get(self, *args):
        self.finish()
        id = self.get_argument("id")
        value = self.get_argument("value")
        data = {"id": id, "value" : value}
        data = json.dumps(data)
        for c in cl:
            c.write_message(data)
        print "cube_data shape %s %s" % cube_data.shape

    @web.asynchronous
    def post(self):
        pass

app = web.Application([
    (r'/', IndexHandler),
    (r'/ws', SocketHandler),
    (r'/api', ApiHandler),
    (r'/(favicon.ico)', web.StaticFileHandler, {'path': '../'}),
    (r'/(rest_api_example.png)', web.StaticFileHandler, {'path': './'}),
])

if __name__ == '__main__':
    app.listen(8888)
    print "connecting to db and reading cube data"
    conn = pg8000.connect(user='postgres', host='ibs-andys-ux3', port=5432, database='wh', password='postgres')
    cursor = conn.cursor()
    cursor.execute("select data from data_cube_medium")
    results = cursor.fetchall()
    cube_data = np.asarray(results)[:,0]
    conn.close()
    print "done"
    ioloop.IOLoop.instance().start()

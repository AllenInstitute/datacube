from twisted.internet.defer import inlineCallbacks, returnValue
from klein import Klein, Plating
from autobahn.twisted.wamp import Application
from datetime import date, timedelta
import simplejson as json

app = Klein()
wampapp = Application()


@app.route('/', methods=['POST'])
@inlineCallbacks
def post_bridge(request):
    params = json.loads(request.content.read())
    procedure = params['procedure']
    args = params.get('args', [])
    kwargs = params.get('kwargs', {})

    res = yield wampapp.session.call(procedure, *args, **kwargs)
    request.responseHeaders.addRawHeader(b'content-type', b'application/json')
    request.responseHeaders.addRawHeader(b'Access-Control-Allow-Origin', b'*')
    returnValue(json.dumps(res, ignore_nan=True, separators=(',',':')))


@app.route('/<string:procedure>', methods=['GET'])
@inlineCallbacks
def get_bridge(request, procedure):
    args = request.args.get(b'args', [b'[]'])[0]
    args = json.loads(args.decode('utf-8'))
    kwargs = request.args.get(b'kwargs', [b'{}'])[0]
    kwargs = json.loads(kwargs.decode('utf-8'))

    res = yield wampapp.session.call(procedure, *args, **kwargs)

    request.responseHeaders.addRawHeader(b'content-type', b'application/json')
    request.responseHeaders.addRawHeader(b'Access-Control-Allow-Origin', b'*')
    #todo: check for some special key in the wamp containing custom headers that
    #  the callee would like to set.
    expiry = date.today() + timedelta(days=1)
    request.setHeader("Expires" , expiry.strftime("%a, %d %b %Y %H:%M:%S GMT"))
    request.responseHeaders.addRawHeader(b'Cache-Control', b'max-age=86400')

    returnValue(json.dumps(res, ignore_nan=True, separators=(',',':')))
 
 
if __name__ == "__main__":
    import sys
    from twisted.python import log
    from twisted.web.server import Site
    from twisted.internet import reactor
    log.startLogging(sys.stdout)

    reactor.listenTCP(9090, Site(app.resource()))
    wampapp.run("ws://localhost:8080/ws", "aibs")

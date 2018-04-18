import sys
import os
import requests
import json

router_uri = 'http://localhost:8080/call'

r=requests.post(router_uri, json={
    "procedure": "wamp.registration.list",
    "args": [],
    "kwargs": {}}).json()
r=map(lambda reg_id: requests.post(router_uri,
                                   json={"procedure": "wamp.registration.get", "args": [reg_id]}).json(),
   r['args'][0][u'prefix']+r['args'][0][u'exact']+r['args'][0][u'wildcard'])

s=''
for proc_uri in [ret['args'][0]['uri'] for ret in r]:
    if 'status' in proc_uri:
        try:
            ret=requests.post(router_uri, json={"procedure": proc_uri}, timeout=10.).json()
            s += '<p>{}: {}</p>'.format(proc_uri, 'up' if ret['args'][0] == True else 'down')
        except:
            s += '<p>{}: down</p>'.format(proc_uri)

print("Content-Type: text/html\n\n")

print("""<!doctype html>
<html>
    <body>
        {}
    </body>
</html>
""".format(s))

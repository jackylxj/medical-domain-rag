import json
import threading
import unittest
from http.server import HTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from scripts.serve_retrieval import handler_for


class FakeRuntime:
    def __init__(self):self.calls=0
    def health(self):return {'status':'ready','device':'cpu'}
    def search(self,question):
        self.calls+=1
        return {'question':question,'hits':[], 'timings':{'embedding_seconds':0.01}}


class ResidentHTTPTests(unittest.TestCase):
    def test_reuse_runtime_and_validate_input(self):
        runtime=FakeRuntime()
        server=HTTPServer(('127.0.0.1',0),handler_for(runtime))
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base+'/health') as r:self.assertEqual(json.load(r)['status'],'ready')
            for _ in range(2):
                request=Request(base+'/search',data=json.dumps({'question':'问题'}).encode(),headers={'Content-Type':'application/json'})
                with urlopen(request) as r:self.assertEqual(json.load(r)['question'],'问题')
            self.assertEqual(runtime.calls,2)
            with self.assertRaises(HTTPError) as ctx:
                urlopen(Request(base+'/search',data=b'{"question":" "}'))
            self.assertEqual(ctx.exception.code,400)
        finally:server.shutdown();server.server_close();worker.join()


if __name__=='__main__':unittest.main()

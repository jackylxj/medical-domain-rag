import json
import unittest
from unittest.mock import patch, MagicMock
from scripts.generate_rag_vllm import request_answer
from scripts.generate_rag_demo import SYSTEM


class VllmClientTests(unittest.TestCase):
    def test_request_and_response(self):
        response=MagicMock()
        response.__enter__.return_value.read.return_value=json.dumps({
            'model':'medical-qwen2','choices':[{'message':{'content':'答案[1]'},'finish_reason':'stop'}],
            'usage':{'completion_tokens':4}}).encode()
        with patch('scripts.generate_rag_vllm.urlopen',return_value=response) as call:
            data,elapsed=request_answer('http://127.0.0.1:8001/v1','medical-qwen2','参考资料')
        payload=json.loads(call.call_args.args[0].data)
        self.assertEqual(payload['messages'][0],{'role':'system','content':SYSTEM})
        self.assertEqual(payload['messages'][1]['content'],'参考资料')
        self.assertEqual(payload['temperature'],0)
        self.assertEqual(payload['max_tokens'],512)
        self.assertEqual(data['choices'][0]['message']['content'],'答案[1]')
        self.assertGreaterEqual(elapsed,0)

    def test_empty_response_rejected(self):
        response=MagicMock()
        response.__enter__.return_value.read.return_value=b'{"choices":[]}'
        with patch('scripts.generate_rag_vllm.urlopen',return_value=response):
            with self.assertRaises(ValueError):request_answer('http://127.0.0.1:8001/v1','medical-qwen2','test')


if __name__=='__main__':unittest.main()

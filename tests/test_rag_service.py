import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from io import BytesIO

from app.rag_service import JobService, BusyError


class ServiceTests(unittest.TestCase):
    def test_cache_hit_skips_generation_and_outage_falls_back(self):
        from app.answer_cache import AnswerCache
        from tests.test_answer_cache import FakeRedis
        calls = []
        def runner(command, **kwargs):
            calls.append(command)
            if 'scripts.generate_rag_vllm' in command:
                target = Path(command[command.index('--output')+1])
                target.mkdir()
                (target/'answers.jsonl').write_text(json.dumps({'answer':'回答','finish_reason':'stop','request_seconds':0.5,'usage':{'total_tokens':12}}),encoding='utf-8')
        with tempfile.TemporaryDirectory() as directory:
            s=JobService(directory,'python','python','bge','qwen','db',runner=runner,backend='vllm',
                         answer_cache=AnswerCache(FakeRedis()))
            with patch.object(s,'_cache_context',return_value={'version':'one'}):
                first=self.wait_done(s,s.submit('问题')['id'])
                count=len(calls)
                second=self.wait_done(s,s.submit('问题')['id'])
                self.assertEqual(second['cache']['status'],'hit')
                self.assertEqual(len(calls),count)
                self.assertNotIn('usage',second['result'])
                self.assertNotIn('vllm_request_seconds',second['timings'])
                self.assertEqual(second['cache']['source_job_id'],first['id'])
            with patch.object(s,'_cache_context',return_value={'version':'two'}):
                third=self.wait_done(s,s.submit('问题')['id'])
                self.assertEqual(third['cache']['status'],'miss')
            with patch.object(s,'_cache_context',return_value={'version':'two'}), patch.object(s.answer_cache,'get',side_effect=ConnectionError()):
                fourth=self.wait_done(s,s.submit('问题')['id'])
                self.assertEqual(fourth['status'],'completed')
                self.assertEqual(fourth['cache']['status'],'bypass_error')

    def test_resident_retrieval_saves_evidence_and_timings(self):
        commands = []
        def runner(command, **kwargs):
            commands.append(command)
            target = Path(command[command.index('--output') + 1])
            target.mkdir()
            (target/'answers.jsonl').write_text(json.dumps({'answer':'示例','request_seconds':0.5}), encoding='utf-8')
        reply = dict(question='问题', hits=[{'text':'证据'}], metadata={'database':str(Path('db').resolve())},
                     timings={'embedding_seconds':0.1})
        with tempfile.TemporaryDirectory() as directory:
            s = JobService(directory,'python','python','bge','qwen','db',runner=runner,
                           backend='vllm',retrieval_url='http://127.0.0.1:8002')
            with patch('app.rag_service.urlopen', return_value=BytesIO(json.dumps(reply).encode())):
                job = s.submit('问题')
                result = self.wait_done(s,job['id'])
            self.assertEqual(result['status'],'completed')
            self.assertEqual(len(commands),1)
            self.assertEqual(result['timings']['vllm_request_seconds'],0.5)
            saved = json.loads((Path(directory)/job['id']/'retrieval/results.json').read_text(encoding='utf-8'))
            self.assertEqual(saved[0]['hits'],reply['hits'])

    def wait_done(self, service, job_id):
        for _ in range(200):
            result = service.get(job_id)
            if result['status'] in ('completed', 'failed'):
                return result
            time.sleep(.01)
        self.fail('job did not finish')

    def test_busy_and_result(self):
        entered, release = threading.Event(), threading.Event()
        def runner(command, **kwargs):
            if 'scripts.retrieve_questions' in command:
                entered.set()
                release.wait(2)
            else:
                target = Path(command[command.index('--output') + 1])
                target.mkdir()
                (target / 'answers.jsonl').write_text(json.dumps({'answer':'示例[1]', 'evidence':[]}), encoding='utf-8')
        with tempfile.TemporaryDirectory() as directory:
            s = JobService(Path(directory), 'python', 'python', 'bge', 'qwen', 'db', runner=runner)
            first = s.submit('问题')
            self.assertTrue(entered.wait(2))
            with self.assertRaises(BusyError):
                s.submit('另一个问题')
            release.set()
            result = self.wait_done(s, first['id'])
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['result']['answer'], '示例[1]')

    def test_failure_releases_slot(self):
        def runner(*args, **kwargs):
            raise RuntimeError('test failure')
        with tempfile.TemporaryDirectory() as directory:
            s = JobService(Path(directory), 'python', 'python', 'bge', 'qwen', 'db', runner=runner)
            for _ in range(2):
                result = self.wait_done(s, s.submit('问题')['id'])
                self.assertEqual(result['status'], 'failed')
                self.assertIn('error', result)
            self.assertIsNone(s.get('missing'))

    def test_vllm_uses_cpu_retrieval_and_http_client(self):
        commands=[]
        def runner(command, **kwargs):
            commands.append(command)
            if 'scripts.generate_rag_vllm' in command:
                target=Path(command[command.index('--output')+1])
                target.mkdir()
                (target/'answers.jsonl').write_text(json.dumps({'answer':'示例','evidence':[]}),encoding='utf-8')
        with tempfile.TemporaryDirectory() as directory:
            s=JobService(Path(directory),'python','python','bge','qwen','db',runner=runner,backend='vllm')
            result=self.wait_done(s,s.submit('问题')['id'])
            self.assertEqual(result['status'],'completed')
            self.assertEqual(commands[0][-2:],['--device','cpu'])
            self.assertIn('scripts.generate_rag_vllm',commands[1])
            self.assertFalse(any('scripts.generate_rag_demo' in c for c in commands))


if __name__ == '__main__':
    unittest.main()

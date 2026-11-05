import json
import unittest
from app.answer_cache import AnswerCache, cache_key


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.expiry = None
    def get(self, key):
        return self.values.get(key)
    def set(self, key, value, ex):
        self.values[key] = value
        self.expiry = ex


class CacheTests(unittest.TestCase):
    def test_keys_separate_question_and_versions(self):
        context = dict(corpus='one', model='base', prompt='v2', deployment='a')
        original = cache_key('问题？', context)
        self.assertNotEqual(original, cache_key('问题', context))
        for field in context:
            changed = dict(context, **{field:'different'})
            self.assertNotEqual(original, cache_key('问题？', changed))
        self.assertEqual(original, cache_key('问题？', dict(reversed(list(context.items())))))

    def test_roundtrip_and_ttl(self):
        client = FakeRedis()
        cache = AnswerCache(client, ttl=3600)
        self.assertIsNone(cache.get('missing'))
        entry = {'result':{'answer':'回答'}, 'source_job_id':'original'}
        cache.put('key', entry)
        self.assertEqual(cache.get('key'), entry)
        self.assertEqual(client.expiry, 3600)

    def test_corrupt_entry_is_rejected(self):
        client = FakeRedis()
        client.values['bad'] = json.dumps({'result':{'answer':''}})
        with self.assertRaises(ValueError):
            AnswerCache(client).get('bad')

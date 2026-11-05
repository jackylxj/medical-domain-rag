"""Optional Redis cache. Errors propagate so the caller can record a bypass."""
import hashlib
import json


def cache_key(question, context):
    payload = json.dumps({'question':question,'context':context}, ensure_ascii=False,
                         sort_keys=True, separators=(',',':')).encode('utf-8')
    return 'ragdemo:answer:v1:' + hashlib.sha256(payload).hexdigest()


class AnswerCache:
    def __init__(self, client, ttl=3600):
        if ttl <= 0:
            raise ValueError('TTL must be positive')
        self.client, self.ttl = client, ttl

    @classmethod
    def from_url(cls, url, ttl=3600):
        import redis
        from redis.retry import Retry
        from redis.backoff import NoBackoff
        return cls(redis.Redis.from_url(url, decode_responses=True, socket_timeout=0.3,
                   socket_connect_timeout=0.3, retry=Retry(NoBackoff(),0)), ttl)

    def get(self, key):
        raw = self.client.get(key)
        if raw is None:
            return None
        entry = json.loads(raw)
        result = entry.get('result') if isinstance(entry,dict) else None
        if not isinstance(result,dict) or not isinstance(result.get('answer'),str) or not result['answer'].strip() or not entry.get('source_job_id'):
            raise ValueError('Invalid cached answer')
        return entry

    def put(self, key, entry):
        self.client.set(key,json.dumps(entry,ensure_ascii=False),ex=self.ttl)

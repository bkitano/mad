# Setup

1. Opencode
> `http://opencode.briankitano.com` should work?

Start opencode server where you have the env var set. It also needs to be wherever your actual file system is.

```bash
opencode serve
```

This spins up opencode on 4096.

2. Start the API.

> `http://mad.briankitano.com` should work?

```bash
uv run uvicorn service.api:app --host 0.0.0.0 --port 8001
```

3. List proposals.

```bash
curl -X GET "http://mad.briankitano.com/proposals" -H "accept: application/json"
```

4. Run an experiment worker on a single proposal.

```bash
uv run python -m service.worker --once
```

# TODO

- [ ] Setup GPT 5.2 Flex for opencode.
- [ ] Move the workers onto separate compute (first containers, then separate machines).
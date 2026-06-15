# Example K8s deployment

```
kubectl apply -f .
```

Then,

- no bearer key is needed as the demo endpoints are not actually protected.
- `--insecure` might be necessary until the certificate is issued.

```
curl --insecure https://temp-example-deployment.breithorn.svc.cscs.ch/v1/chat/completions \ 
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "swiss-ai/Apertus-8B-Instruct-2509",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain what vLLM does in one sentence."}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```
#!/bin/bash

IPADDR=$1

ts=$(date +%s%N)

##curl -X POST http://$IPADDR:11434/api/chat \
curl -X POST http://$IPADDR:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy" \
  -d '{
        "model": "qwen3.5:9b",
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [{
            "type": "function",
            "function": {
               "name": "get_weather",
               "description": "Get current weather",
               "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}
            }
        }]
  }'

echo "elapsed(ms): $((($(date +%s%N) - $ts)/1000000))"

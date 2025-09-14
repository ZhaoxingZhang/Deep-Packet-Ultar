mkdir data
cd data
featurize dataset download a7de404e-7b66-49e7-b193-bb3ae5e24332
unzip processed_all.zip
cd ../work
git clone https://github.com/ZhaoxingZhang/Deep-Packet-Ultar.git
cd Deep-Packet-Ultar 
#conda env create -f env_linux_cuda113.yaml
conda activate deep_packet
npm install -g @anthropic-ai/claude-code
npm install -g @musistudio/claude-code-router

mkdir -p ~/.claude-code-router && echo '{
  "LOG": false,
  "LOG_LEVEL": "info",
  "CLAUDE_PATH": "",
  "HOST": "127.0.0.1",
  "PORT": 3456,
  "APIKEY": "",
  "API_TIMEOUT_MS": "600000",
  "PROXY_URL": "",
  "transformers": [],
  "Providers": [
    {
      "name": "openrouter",
      "api_base_url": "https://openrouter.ai/api/v1/chat/completions",
      "api_key": "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",
      "models": [
        "google/gemini-2.5-pro",
        "anthropic/claude-sonnet-4",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3.7-sonnet:thinking",
        "z-ai/glm-4.5-air:free",
        "z-ai/glm-4.5",
        "openai/gpt-oss-20b:free",
        "openai/o3"
      ],
      "transformer": {
        "use": [
          "openrouter"
        ]
      }
    }
  ],
  "StatusLine": {
    "enabled": false,
    "currentStyle": "default",
    "default": {
      "modules": []
    },
    "powerline": {
      "modules": []
    }
  },
  "Router": {
    "default": "openrouter,z-ai/glm-4.5",
    "background": "openrouter,z-ai/glm-4.5-air:free",
    "think": "openrouter,openai/o3",
    "longContext": "openrouter,z-ai/glm-4.5",
    "longContextThreshold": 60000,
    "webSearch": "openrouter,google/gemini-2.5-pro"
  },
  "stream": false
}' > ~/.claude-code-router/config.json
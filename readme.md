cs_study_voice_agent/
├── main.py                      # entry point — run this for the voice agent
├── api.py                       # optional: read-only HTTP API over history
├── config/
│   ├── settings.py              # all env vars / constants, single source of truth
│   └── database.py              # Mongo connection + Beanie init
├── models/
│   └── conversation.py          # ConversationSession / ConversationMessage (Beanie)
├── services/
│   ├── history_service.py       # all DB reads/writes + past-session context builder
│   ├── search_service.py        # DuckDuckGo web_search tool
│   └── gemini_service.py        # system prompt + LiveConnectConfig builder
├── controllers/
│   └── voice_controller.py      # audio I/O, send/receive loop, tool dispatch
└── routes/
    └── history_routes.py        # GET /sessions, GET /sessions/{id}
# docubot

a discord bot designed to help users learn from training documentation. it uses rag (retrieval-augmented generation) to pull relevant info from your docs and answer questions with proper source citations.

## what it does

- **document retrieval**: indexes pdf, txt, md, and csv files from a training docs folder. when users ask questions, it searches for relevant chunks and includes them in the response with page numbers and source citations
- **image extraction**: automatically pulls diagrams and images from pdfs and can show them when users ask to "see" or "show" something
- **long-term memory**: remembers facts about users across sessions (job, skills, preferences, etc.) with semantic deduplication to avoid storing the same info twice
- **web search**: can search the web for current events or real-time info when needed
- **multi-user support**: tracks conversation context in group channels and handles multiple users talking at once
- **bot-to-bot protection**: has anti-loop logic to prevent infinite back-and-forth with other bots

## setup

### requirements

```
discord.py
langchain
langchain-community
langchain-openai
chromadb
sentence-transformers
ollama
pymupdf (fitz)
beautifulsoup4
requests
numpy
pillow
```

### configuration

edit the config section at the top of `professional-disbot.py`:

```python
myModel = "deepseek/deepseek-r1-distill-llama-70b"  # main text model (openrouter)
USE_OPENROUTER = True  # set to False to use local ollama instead
modelName = "Assistant"  # display name for the bot
imageModel = "llava:7b"  # local ollama model for image analysis

botToken = ""  # your discord bot token
openrouter_api_key = ""  # your openrouter api key
```

### directory structure

```
./training_docs/    # put your training documents here (pdf, txt, md, csv)
./training_images/  # extracted images get saved here automatically
./vector_store/     # chromadb vector store for document retrieval
```

### running

1. add your discord bot token and openrouter api key to the config
2. drop your training docs into `./training_docs/`
3. run `python professional-disbot.py`
4. the bot will index your documents on startup

## usage

### talking to the bot

- **in group channels**: @mention the bot or reply to one of its messages
- **in dms**: just send a message directly

the bot will:
- search your training docs for relevant info
- cite sources with page numbers
- show relevant diagrams if available
- remember things about you for future conversations

### slash commands

| command | description |
|---------|-------------|
| `/userinfo` | shows info about a discord user |
| `/serverinfo` | shows info about the current server |
| `/clear` | clears chat history for the current channel |
| `/save` | saves dm chat history to a json file |
| `/load` | loads previously saved chat history |
| `/remember` | shows what the bot remembers about you |
| `/forget` | clears all memories about you |
| `/tasks` | shows your recent task history |
| `/doc_status` | check how many documents are loaded |
| `/reload_docs` | reload training docs (admin only) |
| `/cleanup_memories` | deduplicate stored memories (admin only) |

## how it works

### document retrieval (rag)

1. on startup, loads all docs from `./training_docs/`
2. splits them into chunks using langchain's recursive text splitter
3. embeds chunks using `all-MiniLM-L6-v2` sentence transformer
4. stores embeddings in chromadb for fast similarity search
5. when a user asks something, retrieves the top 3 most relevant chunks
6. includes the chunks in the prompt so the model can cite sources

### memory system

the bot has two types of memory:

- **short-term**: recent conversation history (last ~100 messages per channel)
- **long-term**: persistent facts about users stored in `disbot_user_memories.json`

facts are extracted automatically using an llm and deduplicated using semantic similarity to avoid storing "works at google" and "employed by google" as separate facts.

### intent detection

before responding, the bot classifies messages into:

- `standard` - normal conversation, uses docs + memory
- `internet` - needs web search for current info
- `show-images` - user wants to see diagrams/images

## notes

- the bot uses openrouter by default for the main text model, but you can switch to local ollama by setting `USE_OPENROUTER = False`
- image analysis always uses local ollama (llava model)
- the system prompt is designed for a training assistant role - edit `BOT_SYSTEM_PROMPT` if you want different behavior
- anti-repetition logic tries to catch when the model gets stuck in loops

## license

do whatever you want with it

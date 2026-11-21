# rem

**A local-first memory for your AI assistants.**

AI coding assistants forget everything when a session ends. You end up repeating the same things over and over: which tools you like, how your project is set up, what you decided last week.

rem fixes that. It quietly listens to your conversations with an assistant such as Claude Code, picks out the facts worth keeping, and hands them back at the start of the next session. Everything stays on your own computer.

## Why local-first

Your conversations with an AI assistant are full of personal and work details. rem is built so they never have to leave your machine.

- **Your data stays on your computer.** Memories are stored in a local folder in your home directory. There is no account, no cloud service and no sync.
- **No external AI service needed.** rem reads your conversations with a small model that runs on your own computer through [Ollama](https://ollama.com). Search uses a local model too.
- **Nothing is sent out by accident.** rem refuses to talk to any online AI service unless you switch that on yourself in the settings.
- **Only you can reach it.** rem runs in the background and accepts connections only from your own user account. It opens no network port.
- **Secrets are scrubbed.** Things like API keys and passwords are removed from conversations before anything is saved.

## Features

- **Remembers automatically.** It picks up facts from your conversations with Claude Code without you doing anything.
- **Knows what to trust.** Things you said yourself count as true. Things the assistant guessed are kept as "unconfirmed" until you agree with them.
- **Understands your projects.** Each memory is tied to a project, to your computer, or to you in general, so one project's details don't leak into another.
- **Smart search.** Finds memories by meaning and by keyword, with the current project first.
- **Knowledge map.** Links memories through the people, tools and projects they mention, and shows them in a desktop window you can explore, as a slowly turning 3D sphere or as a flat 2D graph.
- **Extraction review.** Another tab in that window shows, in plain words, what rem saved from your recent messages, what it already knew, and anything worth a second look.
- **Works with your assistant.** Plugs into Claude Code, and any assistant that supports MCP can search and add memories.
- **Simple command line.** Add, search, list, confirm and delete memories from the terminal.

## Requirements

- Linux (for example Fedora or Ubuntu)
- [uv](https://docs.astral.sh/uv/), which installs Python and everything else rem needs
- [Ollama](https://ollama.com), to run the local model
- Git

## Installation

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then close the terminal and open a new one.

### 2. Install Ollama and download the model

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.5:4b
```

The model is about 3 GB, so the download can take a while.

### 3. Clone the repository

```bash
git clone https://github.com/ArnabHira03/rem.git
cd rem
```

### 4. Install rem

```bash
uv sync --extra graph
```

This installs rem together with the desktop window. Use plain `uv sync` if you don't want the window.

## Setup

### 1. Create the settings file

This tells rem to use the local model you downloaded:

```bash
mkdir -p ~/.config/rem && cat > ~/.config/rem/config.toml <<'EOF'
[llm]
provider = "ollama"
model = "qwen3.5:4b"
think = false
timeout_seconds = 120
EOF
```

### 2. Start rem when you log in

```bash
uv run rem install-service --enable
```

You can also start and stop it by hand with `uv run rem start` and `uv run rem stop`.

### 3. Connect rem to Claude Code

```bash
uv run rem install claude-code
```

Restart Claude Code afterwards. From now on rem remembers what matters from your sessions and reminds Claude of it next time.

### 4. Check that it works

```bash
uv run rem status
```

## Using rem

Run these from the `rem` folder.

```bash
uv run rem add "I prefer short answers"     # tell rem something
uv run rem search "answers"                 # look something up
uv run rem list                             # see everything rem remembers
uv run rem confirm <id>                     # agree with something the assistant suggested
uv run rem delete <id>                      # forget something
uv run --extra graph rem graph              # open the knowledge map
```

Run `uv run rem --help` to see every command.

### The knowledge map

`uv run --extra graph rem graph` opens a window with three tabs. It only reads from rem: nothing you do there changes your memories.

- **Sphere** (the tab it opens on) puts the people, tools and projects rem knows about on the surface of a slowly
  turning sphere, coloured by kind. Things that come up in the same memories sit close together. Each memory is a
  small dot inside the sphere, joined to the things it mentions. Memories that don't mention anything shown circle
  the sphere on a ring.
  - Filled, glowing dots and bright memory dots are confirmed. Hollow rings and dim memory dots aren't yet.
  - Drag to turn the sphere, and scroll to zoom.
  - Point at something to highlight it and its neighbours. Click it, or search and press Enter, to bring it to the
    front and see its details.
  - **auto-rotate** turns the spinning on or off. **Reset view** puts the camera back. **Reset layout** arranges
    everything from scratch.
  - It needs a graphics card with WebGL. Without one, the window shows the Graph tab instead.
- **Graph** shows the same things as a flat 2D graph.
- **Extraction** lists each time rem read your new messages, grouped by day. Pick one to see what rem saved,
  what it confirmed and what it left out, with a short reason for each. Facts marked **⚠ Worth a look** may be
  wrong (for example, filed as something you said when you didn't). **Show** and **Worth a look only** at the top
  filter the list, and **Details**, **Show the messages** and **Technical details** open the rest. In a memory's
  panel on the Sphere or Graph, **See the conversation this came from** jumps to the fact that made it.

The project, scope and search controls at the top apply to both Sphere and Graph.

## Learn more

The design and architecture are described in [`docs/design/`](docs/design/).

## License

rem is released under the [MIT License](LICENSE).

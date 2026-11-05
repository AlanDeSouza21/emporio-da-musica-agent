from datetime import datetime
from pathlib import Path

from src.agent import ask, create_chat

EXAMPLES_DIR = Path("examples")
EXAMPLES_DIR.mkdir(exist_ok=True)
log_path = EXAMPLES_DIR / f"session_{datetime.now():%Y%m%d_%H%M%S}.md"

chat = create_chat()
print("Empório da Música assistant. Type 'exit' to quit.")

while True:
    message = input("\nYou: ").strip()
    if message.lower() in {"exit", "sair"}:
        break
    if not message:
        continue
    try:
        answer, tools = ask(chat, message)
    except Exception as exc:
        print(f"\n[error] Could not get an answer right now: {exc}")
        continue
    print(f"[model: {chat.current_model}]")
    print(f"[tools: {', '.join(tools) or 'none'}]")
    print(f"\nAgent: {answer or '[empty response, try again]'}")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"**Customer:** {message}\n\n")
        f.write(f"*Tools used: {', '.join(tools) or 'none'}*\n\n")
        f.write(f"**Agent:** {answer}\n\n---\n\n")
    
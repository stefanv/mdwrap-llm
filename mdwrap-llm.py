# /// script
# dependencies = [
#   "huggingface-hub",
#   "llama-cpp-python"
# ]
# ///

import argparse
import re
from contextlib import redirect_stderr, contextmanager
import os
import sys

import llama_cpp


SYSTEM_MESSAGE = """
You are a text processor who takes an input document in markdown
format and returns a syntactically identical document, but wrapped so
that there is one sentence per line, without removing any blank
lines. Do not add quotes around the document.
Do not add newlines before or after the document.
Do not remove comments, references, or any other markup.
"""

parser = argparse.ArgumentParser(description="Rewrap a Markdown file to one sentence per line.")
parser.add_argument("markdown_file", help="Path to the markdown file")
args = parser.parse_args()


@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as null:
        with redirect_stderr(null):
            yield


with open(args.markdown_file) as f:
    markdown = f.read()

with suppress_stderr():
    llm = llama_cpp.Llama.from_pretrained(
        repo_id='bartowski/Phi-3.5-mini-instruct-GGUF',
        filename='Phi-3.5-mini-instruct-Q4_K_M.gguf',
        n_ctx=512
    )

pattern = r'(\n\n+.+?)(?=(\n\n+|\Z))'
chunks = [section[0] for section in re.findall(pattern, markdown, re.DOTALL)]

out = []
for chunk in chunks:
    if not chunk:
        continue

    with suppress_stderr():
        result = llm.create_chat_completion(
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE
                },
                {
                    "role": "user",
                    "content": chunk,
                }
            ]
        )

    out.append(result['choices'][0]['message']['content'])

print('\n'.join(out))

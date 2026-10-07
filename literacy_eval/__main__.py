"""Allow running the module with: python -m literacy_eval

Convenience entry point so you can run:
    python -m literacy_eval --model airavata --level low --sample 3

Instead of:
    python -m literacy_eval.run_eval --model airavata --level low --sample 3
"""

from literacy_eval.run_eval import main

main()

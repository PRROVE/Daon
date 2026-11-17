import hashlib
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PromptCheckoutTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("git"), "Requires Git")
    def test_windows_checkout_keeps_pinned_prompt_hash(self):
        # Use the current files, including edits not committed to HEAD yet.
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            (checkout / "prompts").mkdir()
            shutil.copyfile(ROOT / ".gitattributes", checkout / ".gitattributes")
            shutil.copyfile(ROOT / "prompts/story.md", checkout / "prompts/story.md")
            subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
            subprocess.run(["git", "-c", "core.autocrlf=true", "add", ".gitattributes", "prompts/story.md"], cwd=checkout, check=True)
            prompt = subprocess.check_output(
                ["git", "-c", "core.autocrlf=true", "cat-file", "--filters", ":prompts/story.md"], cwd=checkout,
            )
        config = (ROOT / "config/config.yaml").read_text(encoding="utf-8-sig")
        expected = re.search(r"^\s*prompt_sha256:\s*([0-9a-f]{64})", config, re.M).group(1)
        self.assertEqual(hashlib.sha256(prompt).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()

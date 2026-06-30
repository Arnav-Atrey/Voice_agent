import tempfile
import unittest
from pathlib import Path

from services.copilot_service import should_use_copilot, write_generated_code


class CopilotServiceTests(unittest.TestCase):
    def test_detects_autopilot_requests(self) -> None:
        self.assertTrue(should_use_copilot("activate autopilot and write a program"))
        self.assertTrue(should_use_copilot("ask Copilot to generate code for me"))
        self.assertTrue(should_use_copilot("can you write a script to rename files"))

    def test_ignores_regular_chat(self) -> None:
        self.assertFalse(should_use_copilot("explain recursion to me"))
        self.assertFalse(should_use_copilot("what is the capital of France"))

    def test_writes_generated_code_to_output_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = write_generated_code(
                "def greet():\n    return 'hi'\n",
                "hello world",
                base_dir=Path(tmpdir),
            )
            self.assertTrue(file_path.exists())
            self.assertEqual(file_path.parent.name, "copilot code outputs")
            self.assertRegex(file_path.name, r"^program_\d+\.py$")
            self.assertIn("def greet", file_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

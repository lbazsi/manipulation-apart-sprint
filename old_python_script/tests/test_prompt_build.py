import unittest

from deeb_mech_interp.mech.config import ExtractionConfig
from deeb_mech_interp.mech.dataio import build_prompt


class TestPromptBuild(unittest.TestCase):
    def test_prompt_from_question(self):
        cfg = ExtractionConfig()
        row = {"qid": "q1", "frame": "C", "question": "Hi?"}
        pr = build_prompt(row, cfg)
        self.assertEqual(pr.qid, "q1")
        self.assertEqual(pr.frame, "C")
        self.assertIn("Hi?", pr.prompt)

    def test_prompt_from_full_prompt(self):
        cfg = ExtractionConfig()
        row = {"id": "q2", "framing": "evaluation", "prompt": "PROMPT"}
        pr = build_prompt(row, cfg)
        self.assertEqual(pr.qid, "q2")
        self.assertEqual(pr.frame, "E")
        self.assertEqual(pr.prompt, "PROMPT")


if __name__ == "__main__":
    unittest.main()

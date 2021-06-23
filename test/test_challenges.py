import shutil
from pathlib import Path
import unittest
from datasets.config import HF_DATASETS_CACHE
from fewshot.challenges import registry
from fewshot import make_challenge


class TestChallenge(unittest.TestCase):
    def test_challenge_hashes(self):
        shutil.rmtree(Path(HF_DATASETS_CACHE) / 'flex_challenge', ignore_errors=True)
        specs = registry.specs
        wrong_hash_msgs = []
        for k in specs:
            # Ignore sanity for now.
            if 'sanity' not in k:
                try:
                    make_challenge(k, ignore_verification=False)
                except ValueError as e:
                    wrong_hash_msgs.append(str(e))
        self.assertEqual(len(wrong_hash_msgs), 0, '\n'.join(wrong_hash_msgs))


if __name__ == '__main__':
    unittest.main()

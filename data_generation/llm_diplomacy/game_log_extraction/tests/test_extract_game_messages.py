import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "extract_game_messages.py"
spec = importlib.util.spec_from_file_location("extract_game_messages", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExtractMessagesTests(unittest.TestCase):
    def test_private_only_and_summary_filenames(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            game = root / "results" / "game_000"
            game.mkdir(parents=True)
            (game / "lmvsgame.json").write_text(json.dumps({"phases": [{"messages": [
                {"sender": "FRANCE", "recipient": "GERMANY", "phase": "S1901M", "message": "Alliance?"},
                {"sender": "GERMANY", "recipient": "GLOBAL", "phase": "S1901M", "message": "Hello all"},
                {"sender": "ITALY", "recipient": "FRANCE", "phase": "F1901M", "message": "Agreed"}
            ]}]}), encoding="utf-8")
            module.process_game_logs(str(root / "results"), str(root / "messages"))
            p = root / "messages" / "game_000_messages.txt"
            self.assertTrue(p.exists())
            data = p.read_text(encoding="utf-8")
            self.assertIn("FRANCE, Spring 1901 to GERMANY: Alliance?", data)
            self.assertIn("ITALY, Fall 1901 to FRANCE: Agreed", data)
            self.assertNotIn("Hello all", data)

    def test_legacy_naming(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); inp=root/'results'; g=inp/'game_010';g.mkdir(parents=True)
            (g/'lmvsgame.json').write_text(json.dumps({'phases':[{'name':'S1901M','messages':[{'sender':'FRANCE','recipient':'GERMANY','message':'Test alliance'}]}]}),encoding='utf-8')
            module.process_game_logs(str(inp),str(root/'messages'),naming='original')
            self.assertTrue((root/'messages'/'game_010.txt').exists())

    def test_missing_path_does_not_create_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaises(FileNotFoundError):
                module.process_game_logs(str(root/'missing'),str(root/'messages'))
            self.assertFalse((root/'missing').exists())

if __name__ == '__main__': unittest.main()

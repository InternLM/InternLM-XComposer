import importlib.util
import sys
import types
import unittest


def load_choice():
    torch = types.ModuleType("torch")
    utils = types.ModuleType("torch.utils")
    data = types.ModuleType("torch.utils.data")

    class Dataset:
        pass

    data.Dataset = Dataset
    utils.data = data
    torch.utils = utils
    sys.modules["torch"] = torch
    sys.modules["torch.utils"] = utils
    sys.modules["torch.utils.data"] = data
    sys.modules["torchvision"] = types.ModuleType("torchvision")
    pil = types.ModuleType("PIL")
    pil.Image = types.ModuleType("Image")
    sys.modules["PIL"] = pil
    sys.modules["numpy"] = types.ModuleType("numpy")
    spec = importlib.util.spec_from_file_location(
        "seed_utils",
        "InternLM-XComposer-2.0/evaluation/SEED/utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.choice_letter


class SeedChoiceTest(unittest.TestCase):
    def test_a_reply_with_no_choice_is_not_marked_e(self):
        choice_letter = load_choice()
        self.assertEqual(choice_letter("I don't know"), "")

    def test_a_choice_of_e_is_kept(self):
        choice_letter = load_choice()
        self.assertEqual(choice_letter("The answer is E"), "E")

    def test_a_choice_of_b_is_kept(self):
        choice_letter = load_choice()
        self.assertEqual(choice_letter("B"), "B")


if __name__ == "__main__":
    unittest.main()

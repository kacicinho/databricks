from unittest import mock, TestCase
from unittest.mock import patch

from databricks_jobs.jobs.utils import affinity_lib
import importlib


class TestAffinityUDF(TestCase):
    def setUp(self):

        # We need some python black magic in order to be able to mock the decorator of an already imported
        # page : add a patch for the decorator and reload the module, add a cleanup phase
        def kill_patches():
            patch.stopall()
            importlib.reload(affinity_lib)
        self.addCleanup(kill_patches)

        # We patch the decorator as a decorator doing nothing to the function provided
        mock.patch('databricks_jobs.jobs.utils.spark_utils.typed_udf', lambda *args, **kwargs: lambda x: x).start()
        # Reloads the module which applies our patched decorator
        importlib.reload(affinity_lib)

    def test_unique_words_udf(self):
        results = affinity_lib.unique_words_udf("voiture voiture drft bière SUPER 23")
        self.assertSetEqual(set(results), {"voiture", "drft", "bière", "super"})

    def test_unique_words_udf(self):
        results = affinity_lib.token_lemma_udf(["la", "voiture",  "est",  "véritablement", "superbe"])
        self.assertSetEqual(set(results), {"voiture", "être", "véritablement", "superbe"})

    def test_lemma_udf(self):
        results = affinity_lib.token_lemma_udf(["la", "voiture", "est", "véritablement", "superbe"])
        self.assertSetEqual(set(results), {"voiture", "être", "véritablement", "superbe"})

    def test_inventory_fn(self):
        category_keys, keywords_roots, count_keywords =\
            affinity_lib.keywords_inventory("Les anges de la télé-réalité", "Les anges vont faire du surf à Dublin.")
        self.assertSetEqual(set(category_keys), {"ange", "surf", "télé", "réalité"})

    def test_select_info_udf(self):
        row = {"CATEGORY": "Divertissement", "KIND": "Divers",
               "SUMMARY": "Les anges partent à Dallas", "PROGRAM": "Les anges font du karaoké avec un zèbre"}
        rez = affinity_lib.select_info_udf(row, "subinfo")
        # We need an eval as the object has been turned to string
        self.assertSetEqual(set(eval(rez)), {"ange", "karaoké", "zèbre"})
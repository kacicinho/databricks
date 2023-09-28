from unittest import TestCase
from databricks_jobs.jobs.utils.channels.channel import Channel


class TestChannel(TestCase):

    def test_channel_enums(self):
        self.assertEqual(Channel.id_set_to_channel_set({1}), {Channel.CINE_FX})

        lci = Channel.LCI_LA_CHAINE_INFO
        self.assertEqual(lci.id, 5)
        self.assertEqual(lci.name, 'LCI - La Chaîne Info')

    def test_id_set_to_channel_set(self):
        assert(Channel.id_set_to_channel_set({1, 2}) == {Channel(1), Channel(2)})

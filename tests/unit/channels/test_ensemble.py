from unittest import TestCase
from databricks_jobs.jobs.utils.channels.channel import Channel
from databricks_jobs.jobs.utils.channels.ensemble import ChannelEnsemble, M6_TNT_CHANNELS, ChannelEnsembleWithDbInfos


class TestChannelEnsemble(TestCase):

    def test_channel_ensmble__eq__(self):
        ens_1 = ChannelEnsemble(Channel.id_set_to_channel_set({1, 2, 3, 4, 5, 6}))
        ens_2 = ChannelEnsemble(Channel.id_set_to_channel_set({1, 2, 3, 4, 5, 6}))
        self.assertEqual(ens_1, ens_2)

    def test_custom_constant_ensemble(self):
        self.assertEqual(M6_TNT_CHANNELS, ChannelEnsemble(Channel.id_set_to_channel_set({51, 44, 45})))

    def test_channel_ensmble_with_info__eq__(self):
        ens_with_info_1 = ChannelEnsembleWithDbInfos('ensemble 1', 1, Channel.id_set_to_channel_set({1, 2, 3, 4, 5, 6}))
        ens_with_info_2 = ChannelEnsembleWithDbInfos('ensemble 1', 1, Channel.id_set_to_channel_set({1, 2, 3, 4, 5, 6}))
        ens_with_info_3 = ChannelEnsembleWithDbInfos('ensemble 3', 3, Channel.id_set_to_channel_set({1, 2, 3, 4, 5, 6}))
        self.assertEqual(ens_with_info_1, ens_with_info_2)
        self.assertNotEqual(ens_with_info_1, ens_with_info_3)

    def test_channel_ensemble__add__(self):
        ens_with_info_1 = ChannelEnsembleWithDbInfos('ensemble 1', 1, Channel.id_set_to_channel_set({1, 2}))
        ens_with_info_2 = ChannelEnsembleWithDbInfos('ensemble 2', 2, Channel.id_set_to_channel_set({1, 2, 3, 4}))
        ens_with_info_3 = ens_with_info_1 + ens_with_info_2
        self.assertEqual(ens_with_info_1 + ens_with_info_2, ens_with_info_3)

    def test_channel_ensemble__sub__(self):
        ens_with_info_1 = ChannelEnsembleWithDbInfos('ensemble 1', 1, Channel.id_set_to_channel_set({1, 2}))
        ens_with_info_2 = ChannelEnsembleWithDbInfos('ensemble 2', 2, Channel.id_set_to_channel_set({1, 2, 3, 4}))
        ens_3 = ChannelEnsemble(Channel.id_set_to_channel_set({3, 4}))
        self.assertEqual(ens_with_info_2 - ens_with_info_1, ens_3)

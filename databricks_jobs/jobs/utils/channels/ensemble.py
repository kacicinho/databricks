from databricks_jobs.jobs.utils.channels.channel import Channel
from typing import Set, List


class ChannelEnsemble:
    """This class is basically a set of channels plus methods"""

    def __init__(self, channels: Set[Channel]):
        self._channels = channels

    @property
    def channels(self) -> Set[Channel]:
        return self._channels

    def sql_chnl_set(self) -> str:
        """returns a string like (1,2,3,4) which is used in sql queries, like "where CHANNEL in (1,2,3,4). The channel
        ids are sorted """
        chn_list = [str(ch.id) for ch in self._channels]
        chn_list.sort()
        return '(' + ', '.join(chn_list) + ')'

    def chnl_ids(self) -> List[int]:
        st = [ch.id for ch in self._channels]
        return st

    def chnl_names(self) -> Set[str]:
        return {ch.name for ch in self._channels}

    def __sub__(self, other_channel_ensemble):
        return ChannelEnsemble(self.channels.difference(other_channel_ensemble.channels))

    def __add__(self, other_channel_ensemble):
        return ChannelEnsemble(self.channels.union(other_channel_ensemble.channels))

    def __eq__(self, other_channel_ensemble):
        return self.chnl_ids() == other_channel_ensemble.chnl_ids()

    def __str__(self) -> str:
        res = f'{len(self._channels)} channels\n'
        res += f'chnl_names : {self.chnl_names()}\n'
        res += f'chnl_id_list : {self.chnl_ids()}\nsql_chnl_set : {self.sql_chnl_set()}'
        return res


class ChannelEnsembleWithDbInfos(ChannelEnsemble):
    """Used by bundles and groups, and has name and id, corresponding respectively to DataBases
    PROD.BACKEND.CHANNEL_GROUP and PROD.BACKEND.REL_TVBUNDLE_CHANNEL """

    def __init__(self, name: str, ensemble_id: int, channels: Set[Channel]) -> None:
        self._name = name
        self._id = ensemble_id
        self._channels = channels

    @property
    def name(self) -> str:
        return self._name

    @property
    def id(self) -> int:
        return self._id

    def __sub__(self, other_channel_ensemble):
        print('Returning an object of type ChannelEnsemble (no Ensemble.id nor Ensemble.name)')
        return super().__sub__(other_channel_ensemble)

    def __add__(self, other_channel_ensemble):
        print('Returning an object of type ChannelEnsemble (no Ensemble.id nor Ensemble.name)')
        return super().__add__(other_channel_ensemble)

    def __eq__(self, other_channel_ensemble_with_db_infos):
        same_channels = (self.chnl_ids() == other_channel_ensemble_with_db_infos.chnl_ids())
        same_ids = (self.id == other_channel_ensemble_with_db_infos.id)
        same_names = (self.name == other_channel_ensemble_with_db_infos.name)
        return same_channels and same_ids and same_names

    def __str__(self) -> str:
        res = f'{len(self._channels )} channels'
        res += f'id : {self.id}\nname : {self.name}'
        res += super().__str__()
        return res


FREE_BUNDLE_CHANNELS = ChannelEnsemble({Channel.CNEWS,
                                        Channel.LCI_LA_CHAINE_INFO,
                                        Channel.LA_CHAINE_PARLEMENTAIRE,
                                        Channel.CHERIE_25,
                                        Channel.RMC_DECOUVERTE,
                                        Channel.FRANCE_4,
                                        Channel.FRANCE_2,
                                        Channel.NRJ_12,
                                        Channel.ARTE,
                                        Channel.BFMTV,
                                        Channel.TMC,
                                        Channel.CSTAR,
                                        Channel.TFX,
                                        Channel.FRANCE_5,
                                        Channel.SIX_TER,
                                        Channel.M6,
                                        Channel.TF1,
                                        Channel.RMC_STORY,
                                        Channel.FRANCE_3,
                                        Channel.W9,
                                        Channel.L_EQUIPE,
                                        Channel.GULLI,
                                        Channel.C8,
                                        Channel.TF1_SERIES_FILMS,
                                        Channel.BFM_BUSINESS,
                                        Channel.PUBLIC_SENAT_24_24,
                                        Channel.LCP_100,
                                        Channel.FRANCEINFO_136,
                                        Channel.INA,
                                        Channel.BFM_PARIS,
                                        Channel.BFM_LYON_METROPOLE,
                                        Channel.FIGAROLIVE,
                                        Channel.RTFRANCE,
                                        Channel.SPORT_EN_FRANCE,
                                        Channel.FRANCE_INTER,
                                        Channel.SPLASH,
                                        Channel.MANGO_ZYLO,
                                        Channel.MANGO,
                                        Channel.MANGO_ZED,
                                        Channel.MANGO_SONAR,
                                        Channel.MANGO_MEDIAWAN,
                                        Channel.MANGO_ITV,
                                        Channel.MANGO_FIP,
                                        Channel.MANGO_AMPERSAND,
                                        Channel.MANGO_ACI,
                                        Channel.MANGO_ACE,
                                        Channel.BRUT,
                                        Channel.MANGO_CROME,
                                        Channel.MANGO_SONY,
                                        Channel.MANGO_LIONSGATE,
                                        Channel.MANGO_TRADEMEDIADYNAMIC,
                                        Channel.MANGO_EVENTS_FREE,
                                        Channel.MANGO_MK2,
                                        Channel.MANGO_UNIVERSAL_MUSIC,
                                        Channel.MANGO_CINEMA,
                                        Channel.MANGO_SERIES,
                                        Channel.MANGO_DOCS,
                                        Channel.MANGO_KIDS,
                                        Channel.MANGO_FILMRISE,
                                        Channel.MANGO_DYNAMIC,
                                        Channel.MANGO_CINEFLIX,
                                        Channel.MANGO_WILDSIDETV,
                                        Channel.MANGO_ALSHANA,
                                        Channel.WILDSIDETV,
                                        Channel.MANGO_EONE,
                                        Channel.CGTN,
                                        Channel.CGTNFR,
                                        Channel.MANGO_BANIJAY,
                                        Channel.MANGO_FREEDOLPHIN,
                                        Channel.MANGO_UNDERTHEMILKYWAY,
                                        Channel.MANGO_MYDIGITALCOMPANY,
                                        Channel.MANGO_GRB,
                                        Channel.MANGO_MEDIATOON,
                                        Channel.MANGO_FAMILYFILMS})
M6_TNT_CHANNELS = ChannelEnsemble({Channel.SIX_TER,
                                   Channel.M6,
                                   Channel.W9})
TF1_TNT_CHANNELS = ChannelEnsemble({Channel.TF1,
                                    Channel.TF1_SERIES_FILMS,
                                    Channel.TFX,
                                    Channel.TMC})
MANGO_CHANNELS = ChannelEnsemble({Channel.MANGO_ZYLO,
                                  Channel.MANGO,
                                  Channel.MANGO_ZED,
                                  Channel.MANGO_SONAR,
                                  Channel.MANGO_MEDIAWAN,
                                  Channel.MANGO_ITV,
                                  Channel.MANGO_FIP,
                                  Channel.MANGO_AMPERSAND,
                                  Channel.MANGO_ACI,
                                  Channel.MANGO_ACE,
                                  Channel.MANGO_RED_ARROW,
                                  Channel.MANGO_CROME,
                                  Channel.MANGO_SONY,
                                  Channel.MANGO_LIONSGATE,
                                  Channel.MANGO_TRADEMEDIADYNAMIC,
                                  Channel.MANGO_EVENTS_FREE,
                                  Channel.MANGO_MK2,
                                  Channel.MANGO_UNIVERSAL_MUSIC,
                                  Channel.MANGO_CINEMA,
                                  Channel.MANGO_SERIES,
                                  Channel.MANGO_DOCS,
                                  Channel.MANGO_KIDS,
                                  Channel.MANGO_VINTAGE,
                                  Channel.MANGO_FILMRISE,
                                  Channel.MANGO_DYNAMIC,
                                  Channel.MANGO_CINEFLIX,
                                  Channel.MANGO_WILDSIDETV,
                                  Channel.MANGO_ALSHANA,
                                  Channel.MANGO_EONE,
                                  Channel.MANGO_BANIJAY,
                                  Channel.MANGO_FREEDOLPHIN,
                                  Channel.MANGO_UNDERTHEMILKYWAY,
                                  Channel.MANGO_MYDIGITALCOMPANY,
                                  Channel.MANGO_GRB,
                                  Channel.MANGO_MEDIATOON,
                                  Channel.MANGO_FAMILYFILMS,
                                  Channel.MANGO_HISTOIRE})

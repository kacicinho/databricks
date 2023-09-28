from databricks_jobs.jobs.utils.channels.channel import Channel
from databricks_jobs.jobs.utils.channels.ensemble import ChannelEnsembleWithDbInfos

"""
On 2021-12-01 :
ChannelGroup aims at having a correspondence to the data found in PROD.BACKEND.CHANNEL 
A ChannelGroup is basically a ChannelGroup with a name and an bundle_id, which correspond to PROD.BACKEND.CHANNEL_GROUP
"""


class ChannelGroup(ChannelEnsembleWithDbInfos):
    pass


AB = ChannelGroup('AB', 4, {
    Channel.AB1_MA,
    Channel.AB_1,
    Channel.ACTION,
    Channel.ACTION_MA,
    Channel.ALAMAISON,
    Channel.ANIMAUX,
    Channel.ANIMAUX_MA,
    Channel.AUTOMOTO,
    Channel.AUTOMOTO_MA,
    Channel.CHASSE_ET_PECHE,
    Channel.GOLF_CHANNEL,
    Channel.LUCKY_JACK,
    Channel.MANGAS,
    Channel.MANGAS_MA,
    Channel.RTL_9,
    Channel.SCIENCEVIE_MA,
    Channel.SCIENCE_ET_VIE_TV,
    Channel.TOUTE_LHISTOIRE_MA,
    Channel.TOUTE_L_HISTOIRE,
    Channel.TREK})
ALSHANA_TV = ChannelGroup('Alshana TV', 20, {
    Channel.PASSIONS_TV,
    Channel.PASSIONS_TV_CI,
    Channel.SAVANNAHTV,
    Channel.SAVANNAHTV_MA})
CANAL_PLUS = ChannelGroup('Canal Plus', 15, {
    Channel.C8,
    Channel.CINE_PLUS_CLASSIC,
    Channel.CINE_PLUS_CLUB,
    Channel.CINE_PLUS_EMOTION,
    Channel.CINE_PLUS_FAMIZ,
    Channel.CINE_PLUS_FRISSON,
    Channel.CINE_PLUS_PREMIER,
    Channel.CNEWS,
    Channel.CSTAR})
COTE_OUEST = ChannelGroup('Cote Ouest', 25, {
    Channel.NINATV,
    Channel.NINATV_MA})
DIGITAL_VIRGO = ChannelGroup('Digital Virgo', 22, {
    Channel.WAX_ACTION_CI,
    Channel.WAX_COMEDY_CI,
    Channel.WAX_DOCS_CI,
    Channel.WAX_DOCS_SE,
    Channel.WAX_DRAMA_CI,
    Channel.WAX_KIDS_CI,
    Channel.WAX_SERIES_CI})
DISNEY = ChannelGroup('Disney', 7, {
    Channel.DISNEY_CHANNEL,
    Channel.DISNEY_CHANNEL_PLUS_1})
DORCEL = ChannelGroup('Dorcel', 29, {
    Channel.DORCEL_TV,
    Channel.DORCEL_TV_AFRICA,
    Channel.DORCEL_XXX,
    Channel.PLAYBOY_TV,
    Channel.VIXEN_TV})
EURONEWS = ChannelGroup('Euronews', 24, {
    Channel.AFRICANEWS,
    Channel.AFRICANEWS_MA,
    Channel.EURONEWS,
    Channel.EURONEWS_CI,
    Channel.EURONEWS_MA})
FRANCE_TV_PLUS_FRANCE_MEDIA_MONDE = ChannelGroup('France TV + France Media Monde', 2, {
    Channel.BOINGMAX,
    Channel.BOOMERANGMAX,
    Channel.BRUT,
    Channel.CULTUREBOX,
    Channel.FRANCE24_AFRIQUE,
    Channel.FRANCE24_MA,
    Channel.FRANCE2_MA,
    Channel.FRANCE3_MA,
    Channel.FRANCE5_MA,
    Channel.FRANCEINFO_136,
    Channel.FRANCEINFO_MA,
    Channel.FRANCE_2,
    Channel.FRANCE_24,
    Channel.FRANCE_24_ARABIC_MA,
    Channel.FRANCE_24_CI,
    Channel.FRANCE_3,
    Channel.FRANCE_4,
    Channel.FRANCE_5,
    Channel.FRANCE_INTER,
    Channel.FRANCE_O,
    Channel.FTVSVOD,
    Channel.FTVSVODENFANTS,
    Channel.TESTUHD1,
    Channel.TESTUHD2,
    Channel.TOONAMIMAX,
    Channel.TV5MONDE,
    Channel.TV5MONDE_CI,
    Channel.TV5MONDE_MA})
GONG = ChannelGroup('Gong', 8, {
    Channel.GONG,
    Channel.GONG_MAX})
LCP = ChannelGroup('LCP', 5, {
    Channel.LA_CHAINE_PARLEMENTAIRE,
    Channel.LCP_100,
    Channel.PUBLIC_SENAT_24_24})
LCTVI = ChannelGroup('LCTVI', 6, {
    Channel.CAMPAGNES_TV,
    Channel.L_ENORME_TV})
LAGARDERE = ChannelGroup('Lagardère', 17, {
    Channel.CANALJ_MA,
    Channel.CANAL_J,
    Channel.GULLI,
    Channel.GULLI_MA,
    Channel.MEDICI,
    Channel.MEZZO,
    Channel.MEZZO_LIVE_HD,
    Channel.TIJI,
    Channel.TIJI_MA})
M6 = ChannelGroup('M6', 3, {
    Channel.GIRONDINS_TV,
    Channel.M6,
    Channel.M6MUSIC_MA,
    Channel.M6_BOUTIQUE,
    Channel.M6_MA,
    Channel.M6_MUSIC,
    Channel.PARIS_PREMIERE,
    Channel.SIX_TER,
    Channel.TEVA,
    Channel.W9,
    Channel.W9_MA})
MEDI1_TV = ChannelGroup('Medi1 TV', 26, {
    Channel.MEDI1TV_AFRIQUE_CI,
    Channel.MEDI1TV_AFRIQUE_MA,
    Channel.MEDI1TV_ARABIC_CI,
    Channel.MEDI1TV_ARABIC_MA,
    Channel.MEDI1TV_MAGHREB_CI,
    Channel.MEDI1TV_MAGHREB_MA})
NRJ = ChannelGroup('NRJ', 14, {
    Channel.CHERIE_25,
    Channel.NRJMUSICTOUR,
    Channel.NRJ_12})
NEXTRADIO = ChannelGroup('Nextradio', 11, {
    Channel.BFMTV,
    Channel.BFM_BUSINESS,
    Channel.RMC_DECOUVERTE,
    Channel.RMC_SPORT_1,
    Channel.RMC_SPORT_1_UHD,
    Channel.RMC_SPORT_2,
    Channel.RMC_SPORT_3,
    Channel.RMC_SPORT_4,
    Channel.RMC_SPORT_NEWS,
    Channel.RMC_STORY})
ORANGE = ChannelGroup('Orange', 16, {
    Channel.OCS_CHOC,
    Channel.OCS_CITY,
    Channel.OCS_GEANTS,
    Channel.OCS_MAX})
ROTANA_TV = ChannelGroup('Rotana TV', 19, {
    Channel.ROTANA_AFLAM,
    Channel.ROTANA_AFLAM_PLUS_CI,
    Channel.ROTANA_AFLAM_PLUS_MA,
    Channel.ROTANA_CINEMA,
    Channel.ROTANA_CINEMA_MA,
    Channel.ROTANA_CLASSIC_CI,
    Channel.ROTANA_CLIP_CI,
    Channel.ROTANA_CLIP_MA,
    Channel.ROTANA_COMEDY_CI,
    Channel.ROTANA_COMEDY_MA,
    Channel.ROTANA_DRAMA_CI,
    Channel.ROTANA_DRAMA_MA,
    Channel.ROTANA_KIDS_CI,
    Channel.ROTANA_KIDS_MA,
    Channel.ROTANA_MUSIC_CI})
SPI_INTERNATIONAL = ChannelGroup('SPI International', 18, {
    Channel.DOCUBOX,
    Channel.DOCUBOX_MA,
    Channel.FASHIONBOX,
    Channel.FASHIONBOX_MA,
    Channel.FASTFUNBOX_MA,
    Channel.FAST_ET_FUNBOX,
    Channel.FIGHTBOX,
    Channel.FIGHTBOX_MA,
    Channel.FILMBOX,
    Channel.FILMBOX_MA,
    Channel.FILM_BOX,
    Channel.TROIS_CENT_SOIXANTE_TUNEBOX,
    Channel.TROIS_CENT_SOIXANTE_TUNEBOX_MA})
TF1 = ChannelGroup('TF1', 1, {
    Channel.HISTOIRE_MA,
    Channel.HISTOIRE_TV,
    Channel.LCI_LA_CHAINE_INFO,
    Channel.LCI_MA,
    Channel.TF1,
    Channel.TF1_MA,
    Channel.TF1_SERIES_FILMS,
    Channel.TFX,
    Channel.TMC,
    Channel.TVBREIZH,
    Channel.USHUAIATV_MA,
    Channel.USHUAIA_TV})
TRACE = ChannelGroup('Trace', 10, {
    Channel.TRACEPLUS_AFRICA,
    Channel.TRACEPLUS_AFRICA_CI,
    Channel.TRACEPLUS_AFRICA_FR,
    Channel.TRACEPLUS_AYITI,
    Channel.TRACEPLUS_BRAZUCA,
    Channel.TRACEPLUS_CARIBBEAN,
    Channel.TRACEPLUS_LATINA,
    Channel.TRACEPLUS_SPORTSTARS,
    Channel.TRACEPLUS_SPORT_STARS,
    Channel.TRACEPLUS_URBAN,
    Channel.TRACEPLUS_URBAN_CI,
    Channel.TRACEPLUS_URBAN_FR,
    Channel.TRACEURBANAFRICA,
    Channel.TRACE_AFRICA,
    Channel.TRACE_AFRICA_CI,
    Channel.TRACE_CARIBBEAN,
    Channel.TRACE_LATINA,
    Channel.TRACE_PLUS,
    Channel.TRACE_SPORT_STARS,
    Channel.TRACE_URBAN,
    Channel.TRACE_URBAN_AFRIQUE,
    Channel.TRACE_URBAN_CI,
    Channel.TRACE_VANILLA})
TURNER = ChannelGroup('Turner', 12, {
    Channel.BOING,
    Channel.BOING_MA,
    Channel.BOOMERANG,
    Channel.BOOMERANG_MA,
    Channel.CARTOONNETWORK_MA,
    Channel.CARTOON_NETWORK,
    Channel.CNN,
    Channel.CNNINTERNATIONAL_MA,
    Channel.TCM_CINEMA,
    Channel.TCM_MA,
    Channel.TOONAMI})
VIACOM = ChannelGroup('Viacom', 13, {
    Channel.BET,
    Channel.GAME_ONE,
    Channel.J_ONE,
    Channel.MTV,
    Channel.MTV_HITS_FRANCE,
    Channel.NICKELODEON,
    Channel.NICKELODEON_JUNIOR,
    Channel.NICKELODEON_PLUS_1,
    Channel.NICKELODEON_TEEN,
    Channel.PARAMOUNT_CHANNEL,
    Channel.PARAMOUNT_CHANNEL_DECALE})

# Databricks notebook source
import re

import nltk
import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, StringType

from databricks_jobs.jobs.utils.spark_utils import typed_udf

nltk.download('stopwords')
stop_words = set(stopwords.words("french"))
stemmer = nltk.stem.SnowballStemmer("french")
SPACY_MODEL = None

sub_category = [
    "Culture",
    "Divertissement",
    "Documentaires",
    "Informations",
    "Sport",
]

sub_kind = [
    "Aventure",
    "Biopic",
    "Challenge",
    "Coaching",
    "Documentaire",
    "Divers",
    "Divertissement",
    "Faits divers",
    "Film documentaire",
    "Information"
]

keep_kind = [
    "Adaptations",
    "Animaux",
    "Arts martiaux",
    "Athlétisme",
    "Automobile",
    "Autres sports",
    "Basket-ball",
    "Beaux arts",
    "Boxe",
    "Biopic",
    "Catastrophe",
    "Catch",
    "Chasse",
    "Cinéma",
    "Cinéma Asiatique",
    "Civilisations anciennes",
    "Classe de Cinquième",
    "Classe de Première",
    "Classe de Quatrième",
    "Classe de Seconde",
    "Classe de Sixième",
    "Classe de Terminale",
    "Classe de Troisième",
    "Clips",
    "Comédies",
    "Comédies burlesques",
    "Comédies dramatiques",
    "Coupe du Monde de Rugby 2019",
    "Comédies françaises",
    "Comédies musicales",
    "Comédies romantiques",
    "Concerts",
    "Consommation",
    "Conte",
    "Courts métrages",
    "Culture",
    "Cultures du monde",
    "Cyclisme",
    "Cyclo-cross",
    "Danse",
    "Danse classique",
    "Dessins animés",
    "Dessins animés cultes",
    "Déco",
    "Débats",
    "Drames",
    "Drames psychologiques",
    "Drames sentimentaux",
    "Drames sociaux",
    "E-sport",
    "Economie",
    "Environnement",
    "Equitation",
    "Evasion",
    "Famille",
    "Fantastiques",
    "Films",
    "Films de cinéma",
    "Finance",
    "Fitness",
    "Football",
    "Football américain",
    "Gastronomie",
    "Golf",
    "Grands conflits",
    "Guerre",
    "Handball",
    "Histoire",
    "Historique",
    "Hockey sur glace",
    "Horreur",
    "Humour",
    "Infos et journaux télévisés",
    "Investigation",
    "Jardinage",
    "Jazz",
    "Jeunesse",
    "Jeux",
    "Judo",
    "Karaté",
    "Kick-boxing",
    "Kung-fu",
    "Littérature",
    "Mangas",
    "Médicales",
    "Mode",
    "Moyen métrage",
    "Multisports",
    "Musique",
    "Musique classique",
    "Natation",
    "Nature",
    "Opéras",
    "Paris sportifs",
    "Patinage artistique",
    "Pétanque",
    "Poker",
    "Policiers",
    "Politique",
    "Pour les enfants en Maternelle",
    "Replay Roland Garros",
    "Pour les élèves de Primaire",
    "Pour les Collégiens et Lycéens",
    "Quad",
    "Rallye",
    "Recettes à la maison",
    "Rugby",
    "Santé",
    "Science et technologie",
    "Science-fiction",
    "Sentimentales",
    "Ski",
    "Société",
    "Société française",
    "Société internationale",
    "Spectacles",
    "Spécial Emmy Awards",
    "Spécial Championnats du monde d'athlétisme",
    "Spécial Championnats européens",
    "Spécial Cannes",
    "Spécial Coupe du Monde de football",
    "Spécial Education",
    "Spécial Elections",
    "Spécial Elections Américaines",
    "Spécial Élections municipales",
    "Spécial Golden Globes",
    "Sport",
    "Sports - Divers",
    "Sport à la maison",
    "Sports de combat",
    "Sports de glisse",
    "Spécial Roland Garros",
    "Sports extrêmes",
    "Sports mécaniques",
    "Sports nautiques",
    "Surf",
    "Séries",
    "Talk shows",
    "Téléachat",
    "Tennis",
    "Téléfilm",
    "Téléfilms",
    "Téléfilms dramatiques",
    "Téléfilms sentimentaux",
    "Téléréalité",
    "Théâtre",
    "Thriller",
    "Tournoi des VI Nations de Rugby",
    "UEFA Champions League",
    "UEFA Europa League",
    "Volley-ball",
    "Voyage",
    "Web & Gaming",
    "Westerns",
]

ignored_kind = [
    "Aventure",
    "Divers",
    "Interview",
    "Magazines",
    "Emissions spéciales",
    "Cirque",
    "Espionnage",
    "Formule 1",
    "Hippisme",
    "International",
    "Jeep ELITE",
    "Judiciaire",
    "Les grands classiques",
    "Liga Portugaise",
    "MMA",
    "Mélodrame",
    "People",
    "Premier League",
    "Péplum",
    "Rodéo",
    "Services",
    "Spécial 14 juillet",
    "Spécial BD",
    "Spécial Commémoration Attentats de Nice",
    "Spécial D-Day",
    "Spécial Disney",
    "Spécial Droits des femmes",
    "Spécial Euro",
    "Spécial Eurovision",
    "Spécial Hellfest",
    "Spécial JO d'hiver",
    "Spécial Jeux paralympiques",
    "Spécial Journées du Patrimoine",
    "Spécial Noël",
    "Spécial Rentrée",
    "Spécial Saint Valentin",
    "Spécial Scorsese/De Niro",
    "Spécial Sidaction",
    "Spécial Summer",
    "Spécial Téléthon",
    "Spécial Tour de France",
    "Spécial Vendée Globe",
    "Spécial années 80",
    "Spécial vitesse",
    "Spécial été",
    "Test Jul",
    "Tir à l'arc",
    "Turkish Airlines EuroLeague",
    "Téléphone VS Indochine",
    "Variétés",
    "Vie pratique",
    "Voile",
    "Voir le replay - JO d'hiver",
    "Événements",
]


def load_affinities_df(ss):
    return ss.read.format("csv") \
        .load("dbfs:/FileStore/AffinitySegment/affinity_categories_simplified.csv", header='True')


def get_spacy_model():
    global SPACY_MODEL
    if not SPACY_MODEL:
        _model = spacy.load("fr_core_news_lg")
        SPACY_MODEL = _model
    return SPACY_MODEL


def keywords_inventory(title, summary):
    nlp = get_spacy_model()
    stemmer = nltk.stem.SnowballStemmer("french")
    keywords_roots = dict()  # collect the words / root
    category_keys = []
    count_keywords = dict()
    reg = re.compile(r"\b([a-zA-ZÉéèêàâëïîçôö]+)\b")

    if pd.notnull(summary) or pd.notnull(title):
        if pd.isnull(title):
            doc = nlp(" ".join(reg.findall(summary)))
            doc_lem = []
        elif pd.isnull(summary):
            doc = nlp(" ".join(reg.findall(title)))
            doc_lem = []
        else:
            doc = nlp(" ".join(reg.findall(" ".join([title, summary]))))
            doc_lem = []

        for token in doc:
            doc_lem.append(str(token.lemma_))
        doc_lem = nlp(" ".join(doc_lem))
        nouns = [
            str(token) for token in doc_lem if token.pos_ == "NOUN" and len(token) > 2
        ]

        for t in nouns:
            racine = stemmer.stem(t)
            if racine in keywords_roots:
                keywords_roots[racine].add(t)
                count_keywords[racine] += 1
            else:
                keywords_roots[racine] = {t}
                count_keywords[racine] = 1

        for s in keywords_roots.keys():
            if len(keywords_roots[s]) > 1:
                for k in keywords_roots[s]:
                    category_keys.append(k)

            else:
                category_keys.append(list(keywords_roots[s])[0])

    return category_keys, keywords_roots, count_keywords


@typed_udf(StringType())
def select_info_udf(x, type):
    nlp = get_spacy_model()

    reg = re.compile(r"[a-zA-ZÉéèàêâëïîçôö]+")
    category = x["CATEGORY"]
    kind = x["KIND"]
    stop_words = set(stopwords.words("french"))
    info = []
    sub_info = []

    def word_is_ok(word):
        return word not in stop_words and len(word) > 2

    def lemma_tf(word):
        return word.lemma_

    def nlp_process(field):
        return nlp(" ".join(reg.findall(field)).lower())

    cat_lem = list(map(lemma_tf, nlp_process(category)))
    kind_lem = list(map(lemma_tf, nlp_process(kind)))

    if category in sub_category:

        info = list(filter(word_is_ok, cat_lem))

        if kind in keep_kind:
            info = list(filter(word_is_ok, cat_lem))
            sub_info = list(filter(word_is_ok, kind_lem))
        elif kind in sub_kind:
            info = list(filter(word_is_ok, cat_lem))

            summary = x["SUMMARY"]
            title = x["PROGRAM"]
            keywords, _, _ = keywords_inventory(title, summary)
            sub_info += keywords

        elif kind in ignored_kind:
            info = list(filter(word_is_ok, cat_lem))
        else:
            info = list(filter(word_is_ok, cat_lem + kind_lem))

    else:
        if kind in ignored_kind:
            info = list(filter(word_is_ok, cat_lem))
        else:
            info = list(filter(word_is_ok, cat_lem + kind_lem))

    if type == "info":
        return str(info)
    if type == "subinfo":
        return str(sub_info)


@typed_udf(StringType())
def unique_words_udf(x):
    reg = re.compile(r"[a-zA-ZÉéèàêâëïîçôö]+")
    return np.unique([word.lower() for word in list(reg.findall(x))])


@typed_udf(StringType())
def token_lemma_udf(x):
    nlp = get_spacy_model()
    return [str(token.lemma_) for token in nlp(" ".join(x)) if str(token.lemma_) not in stop_words]


@typed_udf(IntegerType())
def count_subcategories_udf(x):
    return int(str(x).count("/") - 1)


def aff_conditions(name, params):
    for key, value in params.items():
        if key == "gender":
            condition = (col("GENDER") == value)
            name += "_" + value

        if key == "low_age":
            if condition is not None:
                condition = condition & (col("AGE") >= value)
            else:
                condition = (col("AGE") >= value)

            name += "_" + str(value) + "+"

        if key == "high_age":
            if condition is not None:
                condition = condition & (col("AGE") <= value)
            else:
                condition = (col("AGE") <= value)

            name += "_" + str(value) + "-"

        if key == "refresh":
            if condition is not None:
                condition = condition & (col("REFRESH") <= value)
            else:
                condition = (col("REFRESH") <= value)

            name += "_" + "R" + str(value)

    return name, condition

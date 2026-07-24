#!/usr/bin/env python3
"""
sret_words.py  —  trait-adjective word bank for the "self" calibration block
-----------------------------------------------------------------------------
Words for the self-referential encoding task (SRET) block ("Does this word describe you?"),
ported from pab2163/mindful_brain_project/self_reference/word_list_splits — a personality-trait
word bank split into positive/negative valence. We shuffle at runtime rather than using the
original repo's 183 pre-generated counterbalanced orders, since this is a single-session live
calibration, not a multi-run fMRI study.
"""
import random

POSITIVE_WORDS = [
    "kind", "active", "healthy", "confident", "consistent", "calm", "sophisticated", "modest",
    "wise", "mature", "productive", "competent", "polite", "ethical", "innovative", "jolly",
    "talented", "hopeful", "energetic", "outgoing", "sincere", "humorous", "dignified", "courageous",
    "perceptive", "considerate", "truthful", "artistic", "glorious", "dependable", "responsible",
    "positive", "patient", "bright", "capable", "sweet", "grateful", "pleasant", "creative",
    "reliable", "smart", "respectable", "decisive", "cheerful", "thorough", "humble", "thoughtful",
    "knowledgeable", "courteous", "attentive", "adaptable", "joyful", "fearless", "forgiving",
    "trustworthy", "daring", "good-natured", "proficient", "self-disciplined", "comical", "devoted",
    "moral", "efficient", "organized", "clever", "helpful", "warm", "lucky", "entertaining",
    "intelligent", "amusing", "charitable", "terrific", "brave", "sympathetic", "optimistic",
    "enthusiastic", "charming", "respectful", "admirable", "trusting", "self-confident",
    "compassionate", "gracious", "humane", "inventive", "observant", "agreeable", "affectionate",
    "sociable", "fun", "happy", "friendly", "nice", "generous", "brilliant", "outstanding",
    "understanding", "honest", "strong", "composed", "gentle", "mighty", "skilled", "lively",
    "cooperative", "admired", "loyal", "elegant", "ambitious", "versatile", "expressive",
    "adventurous", "purposeful", "playful", "selfless", "witty", "diligent", "down-to-earth",
    "communicative",
]

NEGATIVE_WORDS = [
    "guilty", "controlling", "boring", "harsh", "cruel", "rude", "ignorant", "ruthless", "reckless",
    "irrational", "annoying", "irresponsible", "dishonest", "pompous", "unproductive", "sloppy",
    "unfriendly", "unruly", "cowardly", "obnoxious", "grumpy", "lethargic", "bad", "moody", "stupid",
    "stubborn", "combative", "suspicious", "mediocre", "childish", "terrible", "sad", "unhappy",
    "dull", "hostile", "dreadful", "wicked", "rotten", "selfish", "gloomy", "inefficient", "careless",
    "discouraged", "self-conscious", "dreary", "envious", "intolerant", "bossy", "wasteful",
    "hesitant", "defensive", "indecisive", "brooding", "fickle", "unreasonable", "demanding",
    "devious", "awkward", "disdainful", "insincere", "foolish", "weak", "aggressive", "bitter",
    "lonely", "toxic", "nasty", "violent", "useless", "unpleasant", "helpless", "unpopular", "lazy",
    "unwise", "unreliable", "hopeless", "jealous", "messy", "stern", "unhealthy", "unkind",
    "uninspiring", "disagreeable", "disobedient", "unsympathetic", "resentful", "deceptive",
    "prejudiced", "self-critical", "materialistic", "mean", "shallow", "withdrawn", "disturbed",
    "troubled", "awful", "detached", "inadequate", "offensive", "rejected", "immoral", "vicious",
    "cynical", "greedy", "timid", "insecure", "feeble", "troublesome", "insensitive", "fearful",
    "superficial", "thoughtless", "unsophisticated", "compulsive", "untrustworthy", "manipulative",
    "unethical", "absent-minded", "uninteresting", "conceited",
]


def word_deck(rng=None):
    """Endless generator of shuffled (word, valence) pairs, valence in {"+", "-"}.

    Reshuffles a fresh positive+negative deck each time it runs out, so any block duration gets
    enough trials without repeats within a pass through the deck."""
    rng = rng or random.Random()
    pool = [(w, "+") for w in POSITIVE_WORDS] + [(w, "-") for w in NEGATIVE_WORDS]
    while True:
        deck = list(pool)
        rng.shuffle(deck)
        yield from deck

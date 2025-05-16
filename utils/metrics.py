import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def character_accuracy(pred_sequences, target_sequences, pad_id=0):
    """Calculate character-level accuracy."""
    correct = 0
    total = 0
    
    for pred, target in zip(pred_sequences, target_sequences):
        for p, t in zip(pred, target):
            if t != pad_id:  # Skip pad tokens
                total += 1
                if p == t:
                    correct += 1
    
    return correct / total if total > 0 else 0

def word_accuracy(pred_words, target_words):
    """Calculate word-level accuracy."""
    correct = sum(1 for p, t in zip(pred_words, target_words) if p == t)
    total = len(pred_words)
    return correct / total if total > 0 else 0

def bleu_score(pred_words, target_words):
    """Calculate BLEU score for transliteration."""
    smoother = SmoothingFunction().method2
    
    scores = []
    for pred, target in zip(pred_words, target_words):
        score = sentence_bleu([list(target)], list(pred), smoothing_function=smoother)
        scores.append(score)
    
    return np.mean(scores) if scores else 0

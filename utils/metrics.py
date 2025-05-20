import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def character_accuracy(pred_ids, target_ids, pad_id=0):
    """Calculate character-level accuracy for token ID sequences.
    
    Args:
        pred_ids (list): List of lists of predicted token IDs
        target_ids (list): List of lists of target token IDs
        pad_id (int): ID used for padding tokens (to be ignored)
        
    Returns:
        float: Accuracy as fraction of correctly predicted non-pad tokens.
    """
    correct = 0
    total = 0

    for pred_seq, tgt_seq in zip(pred_ids, target_ids):
        for p, t in zip(pred_seq, tgt_seq):
            if t != pad_id:
                total += 1
                if p == t:
                    correct += 1
    return correct / total if total > 0 else 0.0


def word_accuracy(pred_seqs, target_seqs):
    """Calculate exact-match sequence accuracy for decoded string sequences.
    
    Args:
        pred_seqs (list): List of decoded strings (predictions)
        target_seqs (list): List of decoded strings (references)
        
    Returns:
        float: Fraction of sequences that exactly match the reference.
    """
    correct = sum(1 for p, t in zip(pred_seqs, target_seqs) if p == t)
    total = len(pred_seqs)
    return correct / total if total > 0 else 0.0


def bleu_score(pred_texts, target_texts):
    """Calculate character-level BLEU score for transliteration outputs.
    
    Splits each text into a list of characters for BLEU computation.
    
    Args:
        pred_texts (list): List of predicted strings
        target_texts (list): List of reference strings
        
    Returns:
        float: Average BLEU score over the input pairs.
    """
    smoother = SmoothingFunction().method2
    scores = []

    for pred, tgt in zip(pred_texts, target_texts):
        pred_tokens = list(pred)
        tgt_tokens = list(tgt)
        score = sentence_bleu([tgt_tokens], pred_tokens, smoothing_function=smoother)
        scores.append(score)

    return np.mean(scores) if scores else 0.0

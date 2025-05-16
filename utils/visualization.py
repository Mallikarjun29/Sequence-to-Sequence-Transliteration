import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from wordcloud import WordCloud
from collections import Counter
from colour import Color
from IPython.display import HTML, display
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def visualize_attention(word, translated_word, attention_weights, ax=None, font_path=None):
    """Visualize attention weights as a heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    attention_heatmap = np.array(attention_weights)
    
    ax.imshow(attention_heatmap, cmap='viridis')
    
    # Set title and labels
    ax.set_title(f"Attention heatmap: '{word}' -> '{translated_word}'")
    
    # Set ticks
    ax.set_xticks(np.arange(len(word)))
    ax.set_yticks(np.arange(len(translated_word)))
    
    # Set tick labels
    ax.set_xticklabels(list(word))
    
    # Use special font for non-Latin script if provided
    if font_path:
        font_prop = FontProperties(fname=font_path, size=18)
        ax.set_yticklabels(list(translated_word), fontproperties=font_prop)
    else:
        ax.set_yticklabels(list(translated_word))
    
    plt.tight_layout()
    return ax

def visualize_model_outputs(inputs, targets, outputs, font_path=None):
    """Visualize input, target, and model output words using word clouds."""
    # Generate color mappings based on BLEU score similarity
    smoother = SmoothingFunction().method2
    
    # Calculate BLEU scores for each output
    scores = []
    for target, output in zip(targets, outputs):
        score = sentence_bleu([list(target)], list(output), smoothing_function=smoother)
        scores.append(score)
    
    # Create color gradient
    red = Color("red")
    colors = list(red.range_to(Color("violet"), len(inputs)))
    colors = [c.hex for c in colors]
    
    # Map scores to colors
    score_to_index = {score: idx for idx, score in enumerate(sorted(scores))}
    input_colors = {inp: colors[score_to_index[score]] for inp, score in zip(inputs, scores)}
    target_colors = {tgt: colors[score_to_index[score]] for tgt, score in zip(targets, scores)}
    output_colors = {out: colors[score_to_index[score]] for out, score in zip(outputs, scores)}
    
    # Create word clouds
    input_text = Counter(inputs)
    target_text = Counter(targets)
    output_text = Counter(outputs)
    
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    
    # Generate and display word clouds
    wc_in = WordCloud(width=800, height=400, background_color="white")
    wc_in.generate_from_frequencies(input_text)
    
    wc_target = WordCloud(width=800, height=400, background_color="white", 
                          font_path=font_path if font_path else None)
    wc_target.generate_from_frequencies(target_text)
    
    wc_out = WordCloud(width=800, height=400, background_color="white",
                       font_path=font_path if font_path else None)
    wc_out.generate_from_frequencies(output_text)
    
    # Display word clouds
    axs[0].imshow(wc_in.recolor(color_func=lambda word, **kwargs: input_colors.get(word, "black")))
    axs[0].set_title("Input Words", fontsize=22)
    axs[0].axis("off")
    
    axs[1].imshow(wc_target.recolor(color_func=lambda word, **kwargs: target_colors.get(word, "black")))
    axs[1].set_title("Target Words", fontsize=22)
    axs[1].axis("off")
    
    axs[2].imshow(wc_out.recolor(color_func=lambda word, **kwargs: output_colors.get(word, "black")))
    axs[2].set_title("Model Outputs", fontsize=22)
    axs[2].axis("off")
    
    plt.tight_layout()
    plt.show()

def visualize_connectivity(word, translated_word, gradients):
    """Visualize connectivity between input and output characters."""
    # Create HTML for visualization
    print(f"Original Word: {word}")
    print(f"Transliterated Word: {translated_word}")
    
    # Create a gradient color map
    colors = ['#85c2e1', '#89c4e2', '#95cae5', '#99cce6', '#a1d0e8',
              '#b2d9ec', '#baddee', '#c2e1f0', '#eff7fb', '#f9e8e8',
              '#f9e8e8', '#f9d4d4', '#f9bdbd', '#f8a8a8', '#f68f8f',
              '#f47676', '#f45f5f', '#f34343', '#f33b3b', '#f42e2e']
              
    def get_color(value):
        idx = min(int(value * (len(colors) - 1)), len(colors) - 1)
        return colors[idx]
    
    # Normalize gradients to [0, 1]
    norm_gradients = []
    for grad_vec in gradients:
        # Get L2 norm for each character position
        grad_norms = [np.linalg.norm(g) for g in grad_vec]
        # Normalize to [0, 1]
        norm_sum = sum(grad_norms)
        if norm_sum > 0:
            grad_norms = [g / norm_sum for g in grad_norms]
        norm_gradients.append(grad_norms)
    
    # Visualize each output character's connection to input
    for i, output_char in enumerate(translated_word):
        print(f"Connectivity for '{output_char}':")
        html_parts = []
        
        for j, input_char in enumerate(word):
            if j < len(norm_gradients[i]):
                bg_color = get_color(norm_gradients[i][j])
                html_parts.append(f'<span style="color:black;background-color:{bg_color}">{input_char}</span>')
        
        display(HTML(''.join(html_parts)))

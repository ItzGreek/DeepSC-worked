import os
import re
import matplotlib.pyplot as plt
import argparse
import numpy as np
from datetime import datetime

checkpoint_tested = "20240731_144404_CDL_MMSE_32x2_80ep"


def process_data(directory):
    file_path = os.path.join(directory, 'simresults.log')
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
    
    epochs = []
    train_losses = []
    loss_ce = []
    loss_mi = []
    mi = []
    snr = []
    val_losses = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "Type: Train;" in line:
                epoch = int(re.search(r'Epoch: (\d+);', line).group(1))
                loss = float(re.search(r'Loss: ([\d.]+);', line).group(1))
                ce = float(re.search(r'Loss_CE ([\d.]+);', line).group(1))
                mi_loss = float(re.search(r'Loss_MI ([\d.-]+);', line).group(1))
                mutual_info = float(re.search(r'MI ([\d.-]+);', line).group(1))
                snr_value = float(re.search(r'SNR ([\d.]+)', line).group(1))
                
                epochs.append(epoch)
                train_losses.append(loss)
                loss_ce.append(ce)
                loss_mi.append(mi_loss)
                mi.append(mutual_info)
                snr.append(snr_value)

            elif "Type: VAL;" in line:
                epoch = int(re.search(r'Epoch: (\d+);', line).group(1))
                avg_loss = float(re.search(r'Average Loss: ([\d.]+);', line).group(1))
                
                if epoch not in epochs:
                    epochs.append(epoch)
                val_losses.append(avg_loss)

    plot_metric(epochs, train_losses, 'Train Loss', directory)
    plot_metric(epochs, loss_ce, 'Loss CE', directory)
    plot_metric(epochs, loss_mi, 'Loss MI', directory)
    plot_metric(epochs, mi, 'MI', directory)
    plot_metric(epochs, snr, 'SNR', directory)
    plot_metric(epochs[:len(val_losses)], val_losses, 'Validation Loss', directory)

    # Aggiunta per creare il grafico scatter di MI vs SNR
    plot_scatter(mi, snr, epochs, 'MI vs SNR', directory)

    # Cerca il file di performance più recente nella directory
    performance_files = [f for f in os.listdir(directory) if f.endswith('_performance_results.txt')]
    if not performance_files:
        print("No performance_results.txt files found.")
        return    

    # Trova il file più recente basato sulla data nel prefisso
    performance_files.sort(key=lambda x: datetime.strptime(x.split('_performance_results.txt')[0], "%Y%m%d_%H%M%S"))
    latest_performance_file = performance_files[-1]
    performance_file_path = os.path.join(directory, latest_performance_file)

    # Lettura del file performance_results.txt
    with open(performance_file_path, 'r') as file:
        lines = file.readlines()
        performance_snr = eval(lines[1].split(': ')[1])
        bleu_scores = eval(lines[2].split(': ')[1])
        similarity_scores = eval(lines[3].split(': ')[1])
    
    # Aggiunta per creare i grafici bleu_score vs SNR e similarity_score vs SNR
    plot_metric(performance_snr, bleu_scores, 'BLEU Score', directory, 'SNR [dB]')
    plot_metric(performance_snr, similarity_scores, 'Similarity Score', directory, 'SNR [dB]')


def plot_metric(epochs, values, title, directory, xlabel='Epoch'):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, values, marker='o', linestyle='-', label=title)
    plt.title(f'{title} vs {xlabel}')
    plt.xlabel(xlabel)
    plt.ylabel(title)
    plt.legend()
    plt.grid(True)
    
    # Save the figure as a PNG file in the specified directory
    file_name = os.path.join(directory, f'{title.replace(" ", "_").lower()}.png')
    plt.savefig(file_name)

    # Show the plot without blocking
    plt.show(block=False)

# Funzione per plottare il grafico scatter di MI vs SNR
def plot_scatter(mi_values, snr_values, epochs, title, directory):
    mi_values = [-mi for mi in mi_values]  # Invert the sign of MI
    
    # Normalize epochs to range [0, 1] for colormap
    norm_epochs = (np.array(epochs) - min(epochs)) / (max(epochs) - min(epochs))
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(snr_values, mi_values, c=norm_epochs, cmap='RdYlGn', marker='o')
    cbar = plt.colorbar(scatter, label='Epoch')  # Add colorbar to indicate epochs
    # Update colorbar ticks to match actual epochs with reduced frequency
    num_ticks = 5  # Number of ticks to show
    tick_locs = np.linspace(0, 1, num_ticks)
    tick_labels = np.linspace(min(epochs), max(epochs), num_ticks, dtype=int)
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(tick_labels)

    plt.title(title)
    plt.xlabel('SNR [dB]')
    plt.ylabel('(-)MI')
    plt.grid(True)
    
    # Save the figure as a PNG file in the specified directory
    file_name = os.path.join(directory, f'{title.replace(" ", "_").lower()}.png')
    plt.savefig(file_name)

    # Show the plot without blocking
    plt.show(block=False)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process simulation results and plot metrics.')
    parser.add_argument('-d', '--directory', default=f'./checkpoints/{checkpoint_tested}/deepsc', help='Directory containing simresults.log (default: current directory)')
    
    args = parser.parse_args()
    process_data(args.directory)
    
    # Keep the plots open
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

RESULTS_FILENAME_I7 = 'results/i7_9700'
RESULTS_FILENAME_XEON_E5 = 'results/xeon_e5'

def main():
    # ================ Intel Core i7-9700 8-core ================

    # Load the data from each file
    data_c = np.loadtxt(f'{RESULTS_FILENAME_I7}/result_c.txt')
    data_rust = np.loadtxt(f'{RESULTS_FILENAME_I7}/result_rs.txt')
    data_cpp = np.loadtxt(f'{RESULTS_FILENAME_I7}/result_cpp.txt')

    # Combine the data into a list
    data = [data_c, data_rust, data_cpp]
    title = 'GPT-2 Training Times with Intel Core i7-9700 8-core'
    y_label = 'Training Time Per Step (s)'
    x_ticks = ['C', 'Rust', 'C++']

    generate_plot(data, title, y_label, x_ticks = x_ticks, highlight_idx = 1)

    # ================ Intel Xeon E5-2690 v3 12-core ================
    
    # Load the data from each file
    data_c = np.loadtxt(f'{RESULTS_FILENAME_XEON_E5}/result_c.txt')
    data_rust = np.loadtxt(f'{RESULTS_FILENAME_XEON_E5}/result_rs.txt')
    data_cpp = np.loadtxt(f'{RESULTS_FILENAME_XEON_E5}/result_cpp.txt')
    data_mojo = np.loadtxt(f'{RESULTS_FILENAME_XEON_E5}/result_mojo.txt')

    # Combine the data into a list
    data = [data_c, data_rust, data_cpp, data_mojo]
    title = 'GPT-2 Training Times with Intel Xeon E5-2690 v3 12-core'
    y_label = 'Training Time Per Step (s)'
    x_ticks = ['C', 'Rust', 'C++', 'Mojo']

    generate_plot(data, title, y_label, x_ticks = x_ticks, highlight_idx = 1)

def generate_plot(data, title, y_label = None, x_label = None, x_ticks = None, highlight_idx = None):
    # Create a vertical box plot
    plt.figure(figsize=(10, 6))
    box = plt.boxplot(data, patch_artist=True)

    # Customize the plot
    plt.title(title)

    if y_label is not None:
        plt.ylabel(y_label)
    if x_label is not None:
        plt.xlabel(x_label)
    if x_ticks is not None:
        plt.xticks(list(range(1, len(x_ticks) + 1)), x_ticks)

    plt.grid(axis='y')

    if highlight_idx is not None:
        ax = plt.gca()  # Get current axis
        xtick_labels = ax.get_xticklabels()  # Get the x-tick labels

        # Customize Rust's x-axis label
        xtick_labels[highlight_idx].set_color('red')        # Set Rust label color to red
        xtick_labels[highlight_idx].set_fontsize(11)        # Increase font size
        xtick_labels[highlight_idx].set_fontweight('bold')  # Make Rust label bold

    # Display the plot
    plt.show()

if __name__ == '__main__':
    main()
# @title helper functions

def plot_training(losses):
    # Plot the loss
    plt.plot(losses)
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, kept_classes):
    dim = len(kept_classes)
    labels = [class_names[i] for i in kept_classes]
    # Plot the confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred)
    norm_conf_mat = conf_mat / np.sum(conf_mat, axis=1)
    # plot the matrix
    fig, ax = plt.subplots()
    plt.imshow(norm_conf_mat)
    plt.title('Confusion Matrix')
    plt.xlabel('Predictions')
    plt.ylabel('Labels')
    plt.xticks(range(dim), labels, rotation=45)
    plt.yticks(range(dim), labels)
    plt.colorbar()
    # Put number of each cell in plot
    for i in range(dim):
        for j in range(dim):
            c = conf_mat[j, i]
            color = 'black' if c > 500 else 'white'
            ax.text(i, j, str(int(c)), va='center', ha='center', color=color)
    plt.show()


def get_data(filter_classes):
    fashion_mnist = fetch_openml("Fashion-MNIST", parser='auto')
    x, y = fashion_mnist['data'], fashion_mnist['target'].astype(int)
    # Remove classes
    filtered_indices = np.isin(y, filter_classes)
    x, y = x[filtered_indices].to_numpy(), y[filtered_indices]
    # Normalize the pixels to be in [-1, +1] range
    x = ((x / 255.) - .5) * 2
    removed_class_count = 0
    for i in range(10):  # Fix the labels
        if i in filter_classes and removed_class_count != 0:
            y[y == i] = i - removed_class_count
        elif i not in filter_classes:
            removed_class_count += 1
    # Do the train-test split
    return train_test_split(x, y, test_size=10_000)


def onehot_encoder(y, num_labels):
    one_hot = np.zeros(shape=(y.size, num_labels), dtype=int)
    one_hot[np.arange(y.size), y] = 1
    return one_hot


def plot_batch_size(vanila, stochastic, mini_batch):
    fig, axes = plt.subplots(2, 2)
    # Plot the loss
    axes[0, 0].plot(vanila[0], label='Gradient Descent')
    axes[0, 0].plot(stochastic[0], label='Stochastic Gradient Descent')
    axes[0, 0].plot(mini_batch[0], label='Mini-Batch Gradient Descent')
    axes[0, 0].set_xlabel('Epoch'), axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss'), axes[0, 0].legend()
    # Plot the accuracy
    axes[0, 1].plot(vanila[2], label='Gradient Descent')
    axes[0, 1].plot(stochastic[2], label='Stochastic Gradient Descent')
    axes[0, 1].plot(mini_batch[2], label='Mini-Batch Gradient Descent')
    axes[0, 1].set_xlabel('Epoch'), axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Test Accuracy'), axes[0, 1].legend()
    # Plot SGD batch loss
    axes[1, 0].plot(stochastic[1], label='Stochastic Gradient Descent')
    axes[1, 0].set_xlabel('Batch'), axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Stochastic Gradient Descent')
    # Plot MBGD batch loss
    axes[1, 1].plot(mini_batch[1], label='Mini-Batch Gradient Descent')
    axes[1, 1].set_xlabel('Batch'), axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Mini-Batch Gradient Descent')

    fig.set_size_inches(16, 12)
    plt.show()
import matplotlib.pyplot as plt

def plot_history(history, save_fig_name = None):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,6))

    ax[0].plot(epochs, acc, 'bo', label='Training acc')
    ax[0].plot(epochs, val_acc, 'b', label='Validation acc')
    ax[0].set_title('Training and validation accuracy')
    ax[0].legend()
    ax[0].set_ylim([0, 1.2])

    ax[1].plot(epochs, loss, 'bo', label='Training loss')
    ax[1].plot(epochs, val_loss, 'b', label='Validation loss')
    ax[1].set_title('Training and validation loss')
    ax[1].legend()
    ax[1].set_ylim([0, 1.2])

    plt.tight_layout()

    if not save_fig_name == None:
        plt.savefig(save_fig_name, dpi=200)

    plt.show()

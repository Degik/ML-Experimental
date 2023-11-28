import torch
import numpy as np
import matplotlib.pyplot as plt

def plotModel(tensor_input, tensor_output, net, trainer):
    # Genera una griglia di punti
    x_min, x_max = tensor_input[:, 0].min() - 1, tensor_input[:, 0].max() + 1
    y_min, y_max = tensor_input[:, 1].min() - 1, tensor_input[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Calcola i confini di decisione
    Z_input = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()])
    if Z_input.shape[1] != 10:
        Z_input = torch.cat([Z_input, torch.zeros(Z_input.size(0), 10 - Z_input.size(1))], dim=1)

    Z = net(Z_input.double().to(trainer.device))
    Z = Z.argmax(1).cpu().numpy()
    Z = Z.reshape(xx.shape)

    # Visualizza il grafico
    plt.contourf(xx, yy, Z, alpha=0.8)

    # Se tensor_output è un tensore PyTorch, convertilo in un array numpy
    if isinstance(tensor_output, torch.Tensor):
        tensor_output = tensor_output.cpu().numpy()

    # Usa una colormap per mappare i valori numerici a colori
    plt.scatter(tensor_input[:, 0].cpu(), tensor_input[:, 1].cpu(), c=tensor_output, cmap='viridis', edgecolors='k', s=20)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision boundaries")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # Salva il grafico come immagine
    plt.savefig('decision_boundaries.png')

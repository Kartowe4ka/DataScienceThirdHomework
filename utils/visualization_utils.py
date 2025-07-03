import matplotlib.pyplot as plt

def getLearningCurve(epochs, losses, accs, title):
    plt.plot(epochs, losses)
    plt.plot(epochs, accs)
    plt.title(title)
    plt.show()

def getAccuracyCurve(epochs, accs, title):
    plt.plot(epochs, accs)
    plt.title(title)
    plt.show()

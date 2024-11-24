from PA2_net import ThreeLayerNet
from PA2_trainer import OnlyPythonTrainer


lr = 0.01
epoch = 40

threelayer_net = ThreeLayerNet(784, 100, 50, 10, lr)
threelayer_trainer = OnlyPythonTrainer(threelayer_net, lr, epoch)
threelayer_trainer.train()
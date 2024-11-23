import numpy as np
import matplotlib.pyplot as plt
import torch.backends
from dataloader import *
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from PA1_pytorch_net import PytorchCNN, PytorchThreeLayerNet


class PytorchTrainer:
    def __init__(self, net: nn.Module, lr=0.01, epoch=10):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("mps") if torch.backends.mps.is_available() else self.device
        
        self.data_path = "./"
        self.batch_size_train = 300
        self.net = net.to(self.device)
        self.lr = lr
        self.epoch = epoch

        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.softmax = nn.Softmax()

        self.net_name = repr(self.net).split('\n')[0].strip('(')

    
    def __get_epoch_loss(self):
        train_loader = Dataloader(path=self.data_path, is_train=True, shuffle=False, batch_size=60000)
        test_loader = Dataloader(path=self.data_path, is_train=False, shuffle=False, batch_size=10000)

        for train_image, train_label in train_loader:
            break
    
        for test_image, test_label in test_loader:
            break

        with torch.no_grad():
            
            if self.net_name == "PytorchThreeLayerNet": train_image: np.ndarray = train_image.reshape(60000, -1)
            
            train_image = torch.tensor(train_image).to(self.device)
            train_label_fornet = torch.tensor(np.argmax(train_label, axis=1)).to(self.device)
            train_output = self.net(train_image)
            train_loss = self.loss_func(train_output, train_label_fornet)
            train_output = train_output.cpu().detach().numpy()
            train_loss = train_loss.cpu().detach().numpy()

            if self.net_name == "PytorchThreeLayerNet": test_image: np.ndarray = test_image.reshape(10000, -1)
            test_image = torch.tensor(test_image).to(self.device)
            test_label_fornet = torch.tensor(np.argmax(test_label, axis=1)).to(self.device)
            test_output = self.net(test_image)
            test_loss = self.loss_func(test_output, test_label_fornet)
            test_output = test_output.cpu().detach().numpy()
            test_loss = test_loss.cpu().detach().numpy()
            
    
        return train_loss, test_loss
    

    def __plot_train_test_loss(self, train_loss_list, test_loss_list):
        save_path = "./plot_results/" + self.net_name + "_loss.png"
        plt.clf()
        
        train_x = list(range(len(train_loss_list)))
        train_y = np.array(train_loss_list)
        test_x = list(range(len(test_loss_list)))
        test_y = np.array(test_loss_list)
        
        plt.figure(figsize=(8,8))

        plt.plot(train_x, train_y)
        plt.plot(test_x, test_y)

        plt.title('train/test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(["train loss", "test loss"])
        plt.xticks(np.arange(0, self.epoch, 2))

        plt.savefig(save_path)


    def __get_confusion_matrix_prob_image(self):
        print("..... get confusion matrix .....")

        confusion_matrix = np.zeros((10, 10))
        num_of_numbers = [0 for _ in range(10)]

        image_lists = [[] for _ in range(10)]

        pytorch_test_loader = Dataloader(self.data_path, is_train=False, shuffle=False, batch_size=1)

        for test_image, test_label in tqdm(pytorch_test_loader):
            if self.net_name == "PytorchThreeLayerNet": test_image = test_image.reshape(1, -1)
            test_image = torch.tensor(test_image).to(self.device)
            
            output = self.net(test_image)
            output = self.softmax(output)
            output = output.cpu().detach().numpy()

            output_label = np.argmax(output, axis=1)[0]
            answer_label = np.argmax(test_label, axis=1)[0]
            output_p = output[0][answer_label]

            confusion_matrix[answer_label][output_label] += 1
            num_of_numbers[answer_label] += 1

            test_image = test_image.reshape(28,28) * 255
            test_image = test_image.cpu().detach().numpy()

            image_lists[answer_label].append((output_p, test_image))

        for i, num in enumerate(num_of_numbers): confusion_matrix[i] /= num

        image_lists = [sorted(image_lists[key], key=lambda x:x[0], reverse=True) for key in range(len(image_lists)) ]

        return confusion_matrix, image_lists
    

    def __plot_confusion_matrix(self, confusion_matrix: np.ndarray):
        save_path = "./plot_results/" + self.net_name + "_confusion.png"
        plt.clf()
        
        plt.figure(figsize=(8,8))

        plot = plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(plot)
        plt.ylabel('answer label')
        plt.xlabel('prediction')
        plt.xticks(np.arange(0, 10, 1))
        plt.yticks(np.arange(0, 10, 1))

        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                plt.text(j, i, format(confusion_matrix[i, j], ".2f"),
                        horizontalalignment="center",
                        color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black")
                
        plt.savefig(save_path)
            

    def __plot_prob_images(self, image_lists: list):
        save_path = "./plot_results/" + self.net_name + "_topthree.png"
        plt.clf()

        fig, axes = plt.subplots(10, 3, figsize=(8,12))
        axes = axes.flatten()

        num_idx = 0
        img_idx = 0
        for ax in axes:
            ax.imshow(image_lists[num_idx][img_idx][1], cmap='gray_r')
            ax.set_title(f"prob: {image_lists[num_idx][img_idx][0]:.5f}")
            ax.axis('off')

            img_idx += 1
            if img_idx % 3 == 0:
                img_idx = 0
                num_idx += 1
        
        plt.tight_layout()
        plt.savefig(save_path)


    def train(self):
        print(f"... {self.net_name} training ...")

        train_loss_list = []
        test_loss_list = []

        epoch_train_loss, epoch_test_loss = self.__get_epoch_loss()
        train_loss_list.append(epoch_train_loss)
        test_loss_list.append(epoch_test_loss)

        for epoch in tqdm(range(self.epoch)):
            train_loader = Dataloader(path=self.data_path, is_train=True, shuffle=True, batch_size=self.batch_size_train)

            for train_images, train_labels in train_loader:
                if self.net_name == "PytorchThreeLayerNet": 
                    train_images = train_images.reshape(self.batch_size_train, -1)
                
                train_images = torch.tensor(train_images).to(self.device)
                train_labels = torch.tensor(np.argmax(train_labels, axis=1)).to(self.device)

                self.optimizer.zero_grad()

                outputs = self.net(train_images)
                
                train_loss: torch.Tensor = self.loss_func(outputs, train_labels)
                
                train_loss.backward()
                
                self.optimizer.step()

            epoch_train_loss, epoch_test_loss = self.__get_epoch_loss()
            train_loss_list.append(epoch_train_loss)
            test_loss_list.append(epoch_test_loss)


        confusion_matrix, image_lists = self.__get_confusion_matrix_prob_image()
        self.__plot_train_test_loss(train_loss_list, test_loss_list)
        self.__plot_confusion_matrix(confusion_matrix)
        self.__plot_prob_images(image_lists)

        print("... training ends ...")



def main():
    lr = 0.01
    epoch = 20
    threelayer_net = PytorchThreeLayerNet(784,200,100,10)
    CNN_net = PytorchCNN()

    threelayer_trainer = PytorchTrainer(threelayer_net, lr, epoch)
    CNN_trainer = PytorchTrainer(CNN_net, lr, epoch)

    threelayer_trainer.train()
    CNN_trainer.train()


if __name__ == "__main__":
    main()
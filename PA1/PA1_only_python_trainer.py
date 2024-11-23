import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
from tqdm import tqdm

from PA1_only_python_layers import only_python_softmax
from PA1_only_python_net import OnlyPythonThreeLayerNet, OnlyPythonCNN


class OnlyPythonTrainer:
    def __init__(self, net, lr=0.01, epoch=10):
        self.data_path = "./"
        self.batch_size_train = 100
        self.net = net
        self.lr = lr
        self.epoch = epoch

        self.net_name = repr(self.net)
        self.softmax = only_python_softmax

    def __get_epoch_loss(self):
        only_python_train_loader_acc = Dataloader(path=self.data_path, is_train=True, shuffle=True, batch_size=100)
        only_python_test_loader_acc = Dataloader(path=self.data_path, is_train=False, shuffle=True, batch_size=100)

        tmp_train_loss = []
        tmp_test_loss = []

        for train_image, train_label in only_python_train_loader_acc:
            if self.net_name == "OnlyPythonThreeLayerNet": train_image: np.ndarray = train_image.reshape(100, -1)
            train_output = self.net.forward(train_image)
            res_train_loss = self.net.loss(train_output, train_label)
            tmp_train_loss.append(res_train_loss)
        
        for test_image, test_label in only_python_test_loader_acc:
            if self.net_name == "OnlyPythonThreeLayerNet": test_image: np.ndarray = test_image.reshape(100, -1)
            test_output = self.net.forward(test_image)
            res_test_loss = self.net.loss(test_output, test_label)
            tmp_test_loss.append(res_test_loss)

        train_loss = sum(tmp_train_loss) / len(tmp_train_loss)
        test_loss = sum(tmp_test_loss) / len(tmp_test_loss)
        

        return train_loss, test_loss


    def __plot_train_test_loss(self, train_loss_list, test_loss_list):
        plt.clf()
        save_path = "./plot_results/" + self.net_name + "_loss.png"

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

        only_python_test_loader = Dataloader(self.data_path, is_train=False, shuffle=False, batch_size=1)

        for test_image, test_label in tqdm(only_python_test_loader):
            if self.net_name == "OnlyPythonThreeLayerNet": test_image = test_image.reshape(1, -1)
            
            output = self.net.forward(test_image)
            output = self.softmax(output)
            output_label = np.argmax(output, axis=1)[0]
            answer_label = np.argmax(test_label, axis=1)[0]
            output_p = output[0][answer_label]

            confusion_matrix[answer_label][output_label] += 1
            num_of_numbers[answer_label] += 1

            test_image = test_image.reshape(28,28) * 255
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

    
        total_counter = 0
        for epoch in range(self.epoch):
            train_loader = Dataloader(path=self.data_path, is_train=True, shuffle=True, batch_size=self.batch_size_train)
            
            for train_images, train_labels in train_loader:
                # training part
                if self.net_name == "OnlyPythonThreeLayerNet": 
                    train_images = train_images.reshape(self.batch_size_train, -1)
                output = self.net.forward(train_images)

                loss = self.net.loss(output, train_labels)
                
                if total_counter % 200 == 0: 
                    print(f"Loss {loss} at epoch: {epoch} total_counter: {total_counter}")

                self.net.optimizer()

                total_counter += 1
            
            print(f"epoch {epoch} loss calculation : ")
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
    epoch = 40

    threelayer_net = OnlyPythonThreeLayerNet(784, 100, 50, 10, lr)
    # CNN_net = OnlyPythonCNN(lr, 3, 16)

    threelayer_trainer = OnlyPythonTrainer(threelayer_net, lr, epoch)
    # CNN_trainer = OnlyPythonTrainer(CNN_net, lr, 10)

    threelayer_trainer.train()
    # CNN_trainer.train()


if __name__ == "__main__":
    main()
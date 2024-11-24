import numpy as np
import matplotlib.pyplot as plt
from PA1_dataloader import *
from tqdm import tqdm

from PA2_functions import softmax
from PA2_optimizer import SGD
from PA2_dataloader import dataloader, train_set_shuffle
from emo_utils import label_to_emoji


# test the LinearLayer with PA1's trainer
class OnlyPythonTrainer:
    def __init__(self, net, lr=0.01, epoch=10):
        self.data_path = "./"
        self.batch_size_train = 100
        self.net = net
        self.lr = lr
        self.epoch = epoch

        self.net_name = "linear-test"
        self.softmax = softmax

    def __get_epoch_loss(self):
        only_python_train_loader_acc = Dataloader(path=self.data_path, is_train=True, shuffle=True, batch_size=100)
        only_python_test_loader_acc = Dataloader(path=self.data_path, is_train=False, shuffle=True, batch_size=100)

        tmp_train_loss = []
        tmp_test_loss = []

        for train_image, train_label in only_python_train_loader_acc:
            train_image: np.ndarray = train_image.reshape(100, -1)
            train_output = self.net.forward(train_image)
            res_train_loss = self.net.loss(train_output, train_label)
            tmp_train_loss.append(res_train_loss)
        
        for test_image, test_label in only_python_test_loader_acc:
            test_image: np.ndarray = test_image.reshape(100, -1)
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
            test_image = test_image.reshape(1, -1)
            
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
        print(f"... lienar layer training ...")

        train_loss_list = []
        test_loss_list = []

        optimizer = SGD()

        total_counter = 0
        for epoch in range(self.epoch):
            train_loader = Dataloader(path=self.data_path, is_train=True, shuffle=True, batch_size=self.batch_size_train)
            
            for train_images, train_labels in train_loader:
                # training part
                train_images = train_images.reshape(self.batch_size_train, -1)
                
                output = self.net.forward(train_images)
                loss = self.net.loss(output, train_labels)
                
                if total_counter % 200 == 0: 
                    print(f"Loss {loss} at epoch: {epoch} total_counter: {total_counter}")

                self.net.backward()
                optimizer.update(self.net.params, self.net.grads)

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



class Trainer():
    def __init__(self, net, optimizer, epoch, trainer_name=""):
        self.net = net
        self.optimizer = optimizer
        self.epoch = epoch

        self.trainer_name = trainer_name
    
    def __plot_training_loss_graph(self, train_loss_list):
        plt.clf()
        save_path = "./plot_results/" + self.trainer_name + "_loss.png"

        train_x = list(range(len(train_loss_list)))
        train_y = np.array(train_loss_list)
        
        plt.figure(figsize=(8,8))

        plt.plot(train_x, train_y)

        plt.title('train loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')

        plt.savefig(save_path)


    def __get_test_accuracy(self, test_sentences, test_labels):
        correct_predicts = 0
        for i, sentence in enumerate(test_sentences):
            predict_label = np.argmax(self.net.predict(sentence))
            if predict_label == test_labels[i]: correct_predicts += 1

        return correct_predicts / len(test_sentences)


    def __save_test_set_emojis(self, test_sentences, test_labels):
        save_path = "./emoji_results/" + self.trainer_name + "_test_set_emoji.txt"
        with open(save_path, "w") as save_f:
            correct_predicts = 0
            for i, sentence in enumerate(test_sentences):
                predict_label = np.argmax(self.net.predict(sentence))
                if predict_label == test_labels[i]: correct_predicts += 1
                

                print_str = f"\n{predict_label == test_labels[i]} :: "
                print_str += f"test sentence: {sentence} // emoji predicted: {label_to_emoji(predict_label)}"
                save_f.write(f"{print_str} // emoji answer {label_to_emoji(test_labels[i])} \n")

            save_f.write(f"\naccuracy : {correct_predicts / len(test_sentences)}")


    def train(self, is_print=True):
        if is_print: print(f"... {self.trainer_name} training ...")

        train_loss_list = []

        train_sentences, train_labels, test_sentences, test_labels = dataloader()
    
        for epoch in range(self.epoch):
            shuffle_train_sentences, shuffle_train_labels = train_set_shuffle(train_sentences, train_labels)
            epoch_loss = 0.0
            for i, sentence in enumerate(shuffle_train_sentences):
                forward_res = self.net.forward(sentence)
                loss = self.net.loss(forward_res, shuffle_train_labels[i])

                self.net.backward()
                self.optimizer.update(self.net.params, self.net.grads)

                epoch_loss += loss

            epoch_loss /= len(shuffle_train_sentences)
            print(f"> {self.trainer_name} :: Loss {epoch_loss} at epoch : {epoch}")
            train_loss_list.append(epoch_loss)

            # gradually decrease learning rate
            # self.optimizer.lr /= 4.0

        
        self.__plot_training_loss_graph(train_loss_list)
        test_accuracy = self.__get_test_accuracy(test_sentences, test_labels)
        if is_print: print(f"> test set accuracy : {test_accuracy}")
        self.__save_test_set_emojis(test_sentences, test_labels)

        if is_print: print("... training ends ...")

            
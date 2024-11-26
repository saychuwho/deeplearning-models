import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from PA2_functions import softmax
from PA2_optimizer import SGD
from PA2_dataloader import dataloader, train_set_shuffle
from emo_utils import label_to_emoji


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

            
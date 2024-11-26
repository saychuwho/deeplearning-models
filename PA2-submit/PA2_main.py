from PA2_net import PA2VanilaRNN, PA2LSTM
from PA2_trainer import Trainer
from PA2_dataloader import get_map
from PA2_optimizer import SGD, Adam

import multiprocessing

word_to_vec_map_50d = get_map()
word_to_vec_map_100d = get_map("100d")

epoch = 1000         # global epoch
lstm_sgd_lr = 5.0
lstm_decay_step = 10

processes = []

""" case 1: RNN + SGD + 50d """
case_1_net = PA2VanilaRNN(word_to_vec_map=word_to_vec_map_50d,
                          word_vec_size=50,
                          is_dropout=False)
trainer_name = "RNN_SGD_50d"
print(f"\n... {trainer_name} start ...")
lr = 5.0

rnn_epoch = epoch
optimizer = SGD(lr=lr, decay_step=10)
case_1_trainer = Trainer(net=case_1_net,
                         optimizer=optimizer,
                         epoch=rnn_epoch,
                         trainer_name=trainer_name)

case_1_trainer.train()
process = multiprocessing.Process(target=case_1_trainer.train, args=(False,))
processes.append(process)
process.start()


""" case 2: LSTM + SGD + 50d """
case_2_net = PA2LSTM(word_to_vec_map=word_to_vec_map_50d,
                     word_vec_size=50,
                     is_adam=False,
                     is_dropout=False)

trainer_name = "LSTM_SGD_50d"
print(f"\n... {trainer_name} start ...")

optimizer = SGD(lr=lstm_sgd_lr, decay_step=lstm_decay_step)
case_2_trainer = Trainer(net=case_2_net,
                         optimizer=optimizer,
                         epoch=epoch,
                         trainer_name=trainer_name)

# case_2_trainer.train()
process = multiprocessing.Process(target=case_2_trainer.train, args=(False,))
processes.append(process)
process.start()


""" case 3: LSTM + Adam + 50d """
case_3_net = PA2LSTM(word_to_vec_map=word_to_vec_map_50d,
                     word_vec_size=50,
                     is_adam=True,
                     is_dropout=False)
trainer_name = "LSTM_Adam_50d"
print(f"\n... {trainer_name} start ...")
lr = 0.1
epoch_adam = 200
optimizer = Adam(lr=lr)
case_3_trainer = Trainer(net=case_3_net,
                         optimizer=optimizer,
                         epoch=epoch_adam,
                         trainer_name=trainer_name)
# case_3_trainer.train()
process = multiprocessing.Process(target=case_3_trainer.train, args=(False,))
processes.append(process)
process.start()


""" case 4: LSTM + SGD + 100d """
case_4_net = PA2LSTM(word_to_vec_map=word_to_vec_map_100d,
                     word_vec_size=100,
                     is_adam=False,
                     is_dropout=False)
trainer_name = "LSTM_SGD_100d"
print(f"\n... {trainer_name} start ...")

optimizer = SGD(lr=lstm_sgd_lr, decay_step=lstm_decay_step)
case_4_trainer = Trainer(net=case_4_net,
                         optimizer=optimizer,
                         epoch=epoch,
                         trainer_name=trainer_name)
# case_4_trainer.train()
process = multiprocessing.Process(target=case_4_trainer.train, args=(False,))
processes.append(process)
process.start()


""" case 5: LSTM + SGD + 50d + dropout """
case_5_net = PA2LSTM(word_to_vec_map=word_to_vec_map_100d,
                     word_vec_size=100,
                     is_adam=False,
                     is_dropout=True)
trainer_name = "LSTM_SGD_100d_Dropout"
print(f"\n... {trainer_name} start ...")

optimizer = SGD(lr=lstm_sgd_lr, decay_step=lstm_decay_step)
case_5_trainer = Trainer(net=case_5_net,
                         optimizer=optimizer,
                         epoch=epoch,
                         trainer_name=trainer_name)
# case_5_trainer.train()
process = multiprocessing.Process(target=case_5_trainer.train, args=(False,))
processes.append(process)
process.start()

for i, process in enumerate(processes):
    process.join()
    print(f"... Process {i} ends ...")

print("... training end ...")
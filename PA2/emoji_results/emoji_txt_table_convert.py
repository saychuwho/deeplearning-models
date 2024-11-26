def convert_func(file_name, markdown_file):
    with open(file_name, 'r') as f:
        file_content = f.readlines()

    with open(markdown_file, 'w') as f:
        f.write("|Line number|Achieved|Predicted label|Answer label|\n")
        f.write("|:---:|:---:|:---:|:---:|\n")
        line_idx = 0
        for line in file_content:
            tmp_line = line.split()
            
            if not tmp_line: continue
            elif tmp_line[0] == "accuracy": continue

            tmp_str = f"|{line_idx}|{tmp_line[0]}|{tmp_line[-5]}|{tmp_line[-1]}|\n"
            line_idx += 1
            f.write(tmp_str)


file_name_list = ["LSTM_Adam_50d", "LSTM_SGD_50d", "LSTM_SGD_100d", "LSTM_SGD_100d_Dropout", "RNN_SGD_50d"]

for file_n in file_name_list:
    file_name = f"./{file_n}_test_set_emoji.txt"
    markdown_file = f"./{file_n}_test_set_emoji.md"
    convert_func(file_name, markdown_file)
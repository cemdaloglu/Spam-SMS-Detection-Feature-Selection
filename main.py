import csv
import numpy
import math

total_arr = []
uniq_arr = []
with open('tokenized_corpus.csv', 'r') as csvfile:
    readcem = list(csv.reader(csvfile))
for x in readcem:
    for y in x:
        total_arr.append(y)
for x in total_arr:
    if x not in uniq_arr:
        uniq_arr.append(x)


def write_featureset(readcem, uniq_arr):
    row_num = -1
    s = (len(readcem), len(uniq_arr))
    feature_matrix = numpy.zeros(s, dtype = int)
    for x in readcem:
        row_num += 1
        for y in x:
            repeat = 0
            if y in x:
                repeat += 1
            feature_matrix[row_num][uniq_arr.index(y)] += repeat

    with open("feature_set.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(feature_matrix)

def read_featureset():

    with open('feature_set.csv', 'r') as csvfile:
        read_set = list(csv.reader(csvfile))
    with open('labels.csv', 'r') as csvfile:
        read_labels = list(csv.reader(csvfile))
    train_arr = read_set[0:4459][:]
    test_arr = read_set[4460:][:]
    train_label = read_labels[0:4459]
    test_label = read_labels[4460:]

    t1 = len(train_arr[0])
    T_jspam = numpy.zeros(t1, dtype = int)
    T_jham = numpy.zeros(t1, dtype = int)

    n_spam = 0
    row_train = -1
    for x in train_arr:
        row_train += 1
        column_train = -1
        if train_label[row_train] == ['1']:
            n_spam += 1
            for y in x:
                column_train += 1
                T_jspam[column_train] = T_jspam[column_train] + int(y)
        elif train_label[row_train] == ['0']:
            for y in x:
                column_train += 1
                T_jham[column_train] = T_jham[column_train] + int(y)


    pi_yspam = n_spam / len(train_arr)
    sum_T_jspam = sum(T_jspam)
    teta_j_yspam = numpy.zeros(t1)
    column_teta_spam = 0
    sum_T_jham = sum(T_jham)
    teta_j_yham = numpy.zeros(t1)
    column_teta_ham = 0
    for x in T_jspam:
        teta_j_yspam[column_teta_spam] = x/sum_T_jspam
        column_teta_spam += 1

    for x in T_jham:
        teta_j_yham[column_teta_ham] = x/sum_T_jham
        column_teta_ham += 1


    y1 = (len(test_arr), 1)
    y_i = numpy.zeros(y1, dtype=str)
    for x in range (0, len(test_arr)):
        second_sum_spam = 0.0
        second_sum_ham = 0.0
        for y in range (0, len(T_jspam)):
            temp = 0
            temp1 = 0
            if teta_j_yspam[y] > 0:
                temp = math.log(teta_j_yspam[y])
            second_sum_spam += int(test_arr[x][y]) * temp
            if teta_j_yham[y] > 0:
                temp1 = math.log(teta_j_yham[y])
            second_sum_ham += int(test_arr[x][y]) * temp1
        y_i_spam = math.log(pi_yspam) + second_sum_spam
        y_i_ham = math.log(1-pi_yspam) + second_sum_ham
        if y_i_ham > y_i_spam:
            y_i[x] = '0'
        else:
            y_i[x] = '1'
    count = 0
    for x in range(0, len(y_i)):
        if y_i[x] == test_label[x]:
            count += 1
    accuracy = [count / len(y_i) * 100]
    with open("test_accuracy.csv", 'w', newline='') as csvfile:
        csv_writer_2 = csv.writer(csvfile)
        csv_writer_2.writerows([accuracy])

    teta_j_yspam_map = numpy.zeros(t1)
    column_teta_spam_map = 0
    teta_j_yham_map = numpy.zeros(t1)
    column_teta_ham_map = 0
    for x in T_jspam:
        teta_j_yspam_map[column_teta_spam_map] = (x+1)/(sum_T_jspam+1*len(T_jspam))
        column_teta_spam_map += 1

    for x in T_jham:
        teta_j_yham_map[column_teta_ham_map] = (x+1)/(sum_T_jham+1*len(T_jham))
        column_teta_ham_map += 1

    y_i_map = numpy.zeros(y1, dtype=str)
    for x in range (0, len(test_arr)):
        second_sum_spam = 0.0
        second_sum_ham = 0.0
        for y in range (0, len(T_jspam)):
            temp = 0
            temp1 = 0
            if teta_j_yspam_map[y] > 0:
                temp = math.log(teta_j_yspam_map[y])
            second_sum_spam += int(test_arr[x][y]) * temp
            if teta_j_yham_map[y] > 0:
                temp1 = math.log(teta_j_yham_map[y])
            second_sum_ham += int(test_arr[x][y]) * temp1
        y_i_spam = math.log(pi_yspam) + second_sum_spam
        y_i_ham = math.log(1-pi_yspam) + second_sum_ham
        if y_i_ham > y_i_spam:
            y_i_map[x] = '0'
        else:
            y_i_map[x] = '1'
    count = 0
    for x in range(0, len(y_i)):
        if y_i_map[x] == test_label[x]:
            count += 1
    accuracy_laplace = [count / len(y_i) * 100]
    with open("test_accuracy_laplace.csv", 'w', newline='') as csvfile:
        csv_writer_3 = csv.writer(csvfile)
        csv_writer_3.writerows([accuracy_laplace])

with open('labels.csv', 'r') as csvfile:
    read_labels = list(csv.reader(csvfile))
train_label = read_labels[0:4459]
test_label = read_labels[4460:]

# Question 3
# Question 3.1

uniq_feature_arr = []
for x in uniq_arr:
    count_feat = 0
    for y in total_arr:
        if x == y:
            count_feat += 1
    if count_feat >= 10:
        if x not in uniq_feature_arr:
            uniq_feature_arr.append(x)
def forward_selection():
    s2 = (len(readcem), len(uniq_feature_arr))
    feature_selection_matrix = numpy.zeros(s2, dtype = int)
    for x in uniq_feature_arr:
        row_num = -1
        for y in readcem:
            row_num += 1
            repeat = 0
            if x in y:
                repeat += 1
            feature_selection_matrix[row_num][uniq_feature_arr.index(x)] += repeat
    train_arr = feature_selection_matrix[0:4459][:]
    test_arr = feature_selection_matrix[4460:][:]
    temp_feature_selection_matrix = numpy.zeros((len(train_arr), len(train_arr[0])), dtype = int)
    temp_test_arr = numpy.zeros((len(test_arr), len(test_arr[0])), dtype = int)
    temp_selection = 0
    best_index = 0
    temp_select_arr =[]
    temp_selection_prev = [-1]
    t1 = 0
    T_jspam_saved = numpy.zeros(len(train_arr))
    T_jham_saved = numpy.zeros(len(train_arr))

    while temp_selection_prev[0] < temp_selection:
        temp_selection_prev[0] = temp_selection
        t1 += 1

        for x1 in range(0, len(feature_selection_matrix[0])):
            if x1 not in temp_select_arr:
                T_jspam = numpy.zeros(t1, dtype=int)
                T_jham = numpy.zeros(t1, dtype=int)
                if t1 > 1:
                    for x in range(0, t1 - 1):
                        T_jham[x] += T_jham_saved[x]
                        T_jspam[x] += T_jspam_saved[x]
                temp_feature_selection_matrix[:, t1-1] = train_arr[:, x1]

                n_spam = 0
                for x in range(0,len(temp_feature_selection_matrix)):
                    if train_label[x] == ['1']:
                        n_spam += 1
                        T_jspam[t1-1] += temp_feature_selection_matrix[x, t1-1]
                    elif train_label[x] == ['0']:
                        T_jham[t1-1] += temp_feature_selection_matrix[x, t1-1]

                pi_yspam = n_spam / len(temp_feature_selection_matrix)
                sum_T_jspam = sum(T_jspam)
                teta_j_yspam = numpy.zeros(t1)
                column_teta_spam = 0
                sum_T_jham = sum(T_jham)
                teta_j_yham = numpy.zeros(t1)
                column_teta_ham = 0
                for x in T_jspam:
                    teta_j_yspam[column_teta_spam] = (x+1)/(sum_T_jspam+1*len(T_jspam))
                    column_teta_spam += 1
                for x in T_jham:
                    teta_j_yham[column_teta_ham] = (x+1)/(sum_T_jham+1*len(T_jham))
                    column_teta_ham += 1
                temp_test_arr[:, t1-1] = test_arr[:, x1]

                y1 = (len(temp_test_arr), 1)
                y2 = (len(temp_test_arr), len(teta_j_yspam))
                y_i = numpy.zeros(y1, dtype=str)
                y_i_spam = temp_test_arr[:, 0:t1].dot(numpy.log(numpy.transpose(teta_j_yspam))) + math.log(pi_yspam)
                y_i_ham = temp_test_arr[:, 0:t1].dot(numpy.log(numpy.transpose(teta_j_yham))) + math.log(1 - pi_yspam)
                for x in range (0, len(temp_test_arr)):
                    if y_i_ham[x] > y_i_spam[x]:
                        y_i[x] = '0'
                    else:
                        y_i[x] = '1'
                count = 0
                for x in range(0, len(y_i)):
                    if y_i[x] == test_label[x]:
                        count += 1
                accuracy = count / len(y_i) * 100

                if temp_selection == 0:
                    temp_selection = accuracy
                    best_index = x1
                    T_jspam_saved[0] = T_jspam[0]
                    T_jham_saved[0] = T_jham[0]
                else:
                    if temp_selection < accuracy:
                        temp_selection = accuracy
                        best_index = x1
                        T_jspam_saved[t1 - 1] = 0
                        T_jham_saved[t1 - 1] = 0
                        T_jspam_saved[t1 - 1] += T_jspam[t1-1]
                        T_jham_saved[t1 - 1] += T_jham[t1-1]
        temp_feature_selection_matrix[:, t1-1] = train_arr[:, best_index]
        temp_test_arr[:, t1-1] = test_arr[:, best_index]
        temp_select_arr.append(best_index)
    with open("forward_selection.csv", 'w', newline='') as csvfile:
        csv_writer_4 = csv.writer(csvfile)
        for line_by_line in temp_select_arr:
            csv_writer_4.writerow([line_by_line])

# question 3.2

def frequency_selection():
    uniq_arr_freq = []
    descending_uniq_arr_freq = []
    descending_uniq_arr = []
    for x in range(len(uniq_feature_arr)):
        count = 0
        for y in range(len(total_arr)):
            if uniq_feature_arr[x] == total_arr[y]:
                count += 1
        uniq_arr_freq.append(count)
    for x in range(len(uniq_feature_arr)):
        index = numpy.argmax(uniq_arr_freq)
        descending_uniq_arr_freq.append(uniq_arr_freq[index])
        descending_uniq_arr.append(uniq_feature_arr[index])
        uniq_arr_freq[index] = -1
    size = (len(readcem), len(descending_uniq_arr_freq))
    descending_feature_matrix = numpy.zeros(size, dtype=int)
    for x in uniq_feature_arr:
        row_num = -1
        for y in readcem:
            row_num += 1
            repeat = 0
            if x in y:
                repeat += 1
            descending_feature_matrix[row_num][descending_uniq_arr.index(x)] += repeat

    train_arr = descending_feature_matrix[0:4459][:]
    test_arr = descending_feature_matrix[4460:][:]
    temp_feature_selection_matrix = numpy.zeros((len(train_arr), len(train_arr[0])), dtype=int)
    temp_test_arr = numpy.zeros((len(test_arr), len(test_arr[0])), dtype=int)
    temp_selection = []
    t1 = 0
    T_jspam_saved = numpy.zeros(len(train_arr))
    T_jham_saved = numpy.zeros(len(train_arr))

    for x1 in range(0, len(descending_feature_matrix[0])):
        t1 += 1
        T_jspam = numpy.zeros(t1, dtype=int)
        T_jham = numpy.zeros(t1, dtype=int)
        if t1 > 1:
            for x in range(0, t1 - 1):
                T_jham[x] += T_jham_saved[x]
                T_jspam[x] += T_jspam_saved[x]
        temp_feature_selection_matrix[:, t1-1] = train_arr[:, x1]
        n_spam = 0
        for x in range(0, len(temp_feature_selection_matrix)):
            if train_label[x] == ['1']:
                n_spam += 1
                T_jspam[t1-1] += temp_feature_selection_matrix[x, t1-1]
            elif train_label[x] == ['0']:
                T_jham[t1-1] += temp_feature_selection_matrix[x, t1-1]

        pi_yspam = n_spam / len(temp_feature_selection_matrix)
        sum_T_jspam = sum(T_jspam)
        teta_j_yspam = numpy.zeros(t1)
        column_teta_spam = 0
        sum_T_jham = sum(T_jham)
        teta_j_yham = numpy.zeros(t1)
        column_teta_ham = 0
        for x in T_jspam:
            teta_j_yspam[column_teta_spam] = (x+1)/(sum_T_jspam+1*len(T_jspam))
            column_teta_spam += 1
        for x in T_jham:
            teta_j_yham[column_teta_ham] = (x+1)/(sum_T_jham+1*len(T_jham))
            column_teta_ham += 1
        temp_test_arr[:, t1-1] = test_arr[:, x1]

        y1 = (len(temp_test_arr), 1)
        y2 = (len(temp_test_arr), len(teta_j_yspam))
        y_i = numpy.zeros(y1, dtype=str)
        y_i_spam = temp_test_arr[:, 0:t1].dot(numpy.log(numpy.transpose(teta_j_yspam))) + math.log(pi_yspam)
        y_i_ham = temp_test_arr[:, 0:t1].dot(numpy.log(numpy.transpose(teta_j_yham))) + math.log(1 - pi_yspam)
        for x in range (0, len(temp_test_arr)):
            if y_i_ham[x] > y_i_spam[x]:
                y_i[x] = '0'
            else:
                y_i[x] = '1'
        count = 0
        for x in range(0, len(y_i)):
            if y_i[x] == test_label[x]:
                count += 1
        accuracy = count / len(y_i) * 100

        if temp_selection == 0:
            temp_selection = accuracy
            T_jspam_saved[0] = T_jspam[0]
            T_jham_saved[0] = T_jham[0]
        else:
            temp_selection.append(accuracy)
            T_jspam_saved[t1 - 1] = 0
            T_jham_saved[t1 - 1] = 0
            T_jspam_saved[t1 - 1] += T_jspam[t1-1]
            T_jham_saved[t1 - 1] += T_jham[t1-1]
        temp_feature_selection_matrix[:, t1-1] = train_arr[:, t1-1]
        temp_test_arr[:, t1-1] = test_arr[:, t1-1]
    with open("frequency_selection.csv", 'w', newline='') as csvfile:
        csv_writer_5 = csv.writer(csvfile)
        for line_by_line in temp_selection:
            csv_writer_5.writerow([line_by_line])

# question 2.1
write_featureset(readcem, uniq_arr)
# question 2.2 & 2.3
read_featureset()
# question 3.1
forward_selection()
# question 3.2
frequency_selection()

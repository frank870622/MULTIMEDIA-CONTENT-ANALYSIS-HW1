import matplotlib.pyplot as plt
import numpy as np

def readfile(file_pos):
    f = open(file_pos, "r")
    lines = f.readlines()
    count = 0
    for line in lines:
        if '[' in line:
            if count == 0:
                recall_array = np.array(eval(line))
                count += 1
            elif count == 1:
                precision_array = np.array(eval(line))
                count += 1
            elif count == 2:
                false_array = np.array(eval(line))
                count += 1
            elif count == 3:
                truth_array = np.array(eval(line))
                count += 1
    f.close()

    return recall_array, precision_array, false_array, truth_array


if __name__ == '__main__':
    video_number = 3

    parewise_precision = np.array([])
    parewise_recall = np.array([])
    parewise_truth = np.array([])
    parewise_false = np.array([])

    parewise_color_precision = np.array([])
    parewise_color_recall = np.array([])
    parewise_color_truth = np.array([])
    parewise_color_false = np.array([])

    histogram_precision = np.array([])
    histogram_recall = np.array([])
    histogram_truth = np.array([])
    histogram_false = np.array([])

    histogram_twin_precision = np.array([])
    histogram_twin_recall = np.array([])
    histogram_twin_truth = np.array([])
    histogram_twin_false = np.array([])

    histogram_color_precision = np.array([])
    histogram_color_recall = np.array([])
    histogram_color_truth = np.array([])
    histogram_color_false = np.array([])

    histogram_color_twin_precision = np.array([])
    histogram_color_twin_recall = np.array([])
    histogram_color_twin_truth = np.array([])
    histogram_color_twin_false = np.array([])

    likelihood_precision = np.array([])
    likelihood_recall = np.array([])
    likelihood_truth = np.array([])
    likelihood_false = np.array([])

    edge_precision = np.array([])
    edge_recall = np.array([])
    edge_truth = np.array([])
    edge_false = np.array([])

    edge_twin_precision = np.array([])
    edge_twin_recall = np.array([])
    edge_twin_truth = np.array([])
    edge_twin_false = np.array([])
    ###################################################################################
    parewise_recall, parewise_precision, parewise_false, parewise_truth = readfile("video" + str(video_number) + "file_parewise.txt")
    parewise_color_recall, parewise_color_precision, parewise_color_false, parewise_color_truth = readfile("video" + str(video_number) + "file_parewise_color.txt")
    histogram_recall, histogram_precision, histogram_false, histogram_truth = readfile("video" + str(video_number) + "file_histogram.txt")
    histogram_twin_recall, histogram_twin_precision, histogram_twin_false, histogram_twin_truth = readfile("video" + str(video_number) + "file_histogram_twin.txt")
    histogram_color_recall, histogram_color_precision, histogram_color_false, histogram_color_truth = readfile("video" + str(video_number) + "file_histogram_color.txt")
    histogram_color_twin_recall, histogram_color_twin_precision, histogram_color_twin_false, histogram_color_twin_truth = readfile("video" + str(video_number) + "file_histogram_color_twin.txt")
    likelihood_recall, likelihood_precision, likelihood_false, likelihood_truth = readfile("video" + str(video_number) + "file_likelihood.txt")
    edge_recall, edge_precision, edge_false, edge_truth = readfile("video" + str(video_number) + "file_edge.txt")
    edge_twin_recall, edge_twin_precision, edge_twin_false, edge_twin_truth = readfile("video" + str(video_number) + "file_edge_twin.txt")

    ###################################################################################
    plt.figure(1)
    plt.title('pr curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])

    plt.plot(parewise_recall, parewise_precision, label='parewise')
    plt.plot(parewise_color_recall, parewise_color_precision, label='parewise color')

    plt.legend()
    plt.savefig("video" + str(video_number) + "_pr_part1" + ".png")
    ######################################
    plt.figure(2)
    plt.title('pr curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])

    plt.plot(histogram_recall, histogram_precision, label='histogram')
    plt.plot(histogram_twin_recall, histogram_twin_precision, label='histogram twin')
    plt.plot(histogram_color_recall, histogram_color_precision, label='histogram color')
    plt.plot(histogram_color_twin_recall, histogram_color_twin_precision, label='histogram color twin')

    plt.legend()
    plt.savefig("video" + str(video_number) + "_pr_part2" + ".png")
    ######################################
    plt.figure(3)
    plt.title('pr curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    
    plt.plot(likelihood_recall, likelihood_precision, label='likelihood')
    plt.plot(edge_recall, edge_precision, label='edge')
    plt.plot(edge_twin_recall, edge_twin_precision, label='edge twin')

    plt.legend()
    plt.savefig("video" + str(video_number) + "_pr_part3" + ".png")
    ######################################
    plt.figure(4)
    plt.title('roc curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])

    plt.plot(parewise_false, parewise_truth, label='parewise')
    plt.plot(parewise_color_false, parewise_color_truth, label='parewise color')

    plt.legend()
    plt.savefig("video" + str(video_number) + "_roc_part1" + ".png")
    ######################################
    plt.figure(5)
    plt.title('roc curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])

    plt.plot(histogram_false, histogram_truth, label='histogram')
    plt.plot(histogram_twin_false, histogram_twin_truth, label='histogram twin')
    plt.plot(histogram_color_false, histogram_color_truth, label='histogram color')
    plt.plot(histogram_color_twin_false, histogram_color_twin_truth, label='histogram color twin')

    plt.legend()
    plt.savefig("video" + str(video_number) + "_roc_part2" + ".png")
    ######################################
    plt.figure(6)
    plt.title('roc curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])

    plt.plot(likelihood_false, likelihood_truth, label='likelihood')
    plt.plot(edge_false, edge_truth, label='edge')
    plt.plot(edge_twin_false, edge_twin_truth, label='edge twin')

    plt.legend()
    plt.savefig("video" + str(video_number) + "_roc_part3" + ".png")
    plt.show()

    
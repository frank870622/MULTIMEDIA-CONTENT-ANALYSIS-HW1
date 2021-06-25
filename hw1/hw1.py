import cv2
import numpy as np

f = open("logfile.txt", "w")

class image_dataset:
    def __init__(self, file_pos, gray_flag):
        self.image_array = []
        cap = cv2.VideoCapture(file_pos)
        count = 0
        while cap.isOpened():
            ret, gray = cap.read()
            # if frame is read correctly ret is True
            if not ret: 
                print("Can't receive frame (stream end?). Exiting ...")
                break
            if gray_flag is True:
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            self.image_array.append(gray)
            count += 1
            if file_pos == 'ngc.mpeg' and count >= 1060:
                #print('file is ngc and count >= 1060')
                break

        self.image_array = np.array(self.image_array)
        self.array_len = len(self.image_array)

    def change_int32(self):
        self.image_array = self.image_array.astype('int32')

def smooth_image(input_array):
    #M, N = input_array.image_array.shape[1] , input_array.image_array.shape[2]
    for i in range(0, input_array.array_len):
        if i % 50 == 0:
            print("smooth picture : " + str(i))
        input_array.image_array[i] = cv2.GaussianBlur(input_array.image_array[i], (3, 3), 0)

    return input_array


def parewise(image_array, array_len, gray_flag, threshold1, threshold2):
    answer_parewise = np.array([])
    M, N = image_array.shape[1] ,  image_array.shape[2]

    for i in range(array_len-1):
        if gray_flag:
            this_image = np.reshape(image_array[i], (M*N)).astype('int32')
            next_image = np.reshape(image_array[i+1], (M*N)).astype('int32')
            DP = (abs(this_image - next_image)*3 > threshold1).astype(int)
            parewise_ans = (np.sum(DP) / (M*N)) * 100
            if parewise_ans > threshold2:
                print(i+1, parewise_ans)
                answer_parewise = np.append(answer_parewise, i+1)
        else:
            this_image = image_array[i].astype('int32')
            next_image = image_array[i+1].astype('int32')
            DP = np.absolute(this_image - next_image)
            DP = ((DP.sum(axis = 2)) > threshold1).astype(int)
            
            parewise_ans = (np.sum(DP) / (M*N)) * 100
            if parewise_ans > threshold2:
                print(i+1, parewise_ans)
                answer_parewise = np.append(answer_parewise, i+1)
            #f.writelines(str(this_image.tolist()) + '\r\n')
    return answer_parewise

def histogram(image_array, array_len, gray_flag, threshold):
    answer_histogram = np.array([])

    M, N = image_array.shape[1] ,  image_array.shape[2]
    histogram_array = np.array([])
    #answer_array = np.array([])

    for i in range(array_len-1):
        if gray_flag:
            this_image = np.reshape(image_array[i], (image_array[i].shape[0] * image_array[i].shape[1]))
            next_image = np.reshape(image_array[i+1], (image_array[i+1].shape[0] * image_array[i+1].shape[1]))

            this_histogram = cv2.calcHist([this_image],[0],None,[256],[0,256]) 
            next_histogram = cv2.calcHist([next_image],[0],None,[256],[0,256]) 

            SD = (np.sum((np.absolute(this_histogram - next_histogram))) / (M*N)) * 3
            histogram_array = np.append(histogram_array, SD)

            if SD > threshold:
                print(i+1, SD)
                answer_histogram = np.append(answer_histogram, i+1)
        else:
            this_image = image_array[i]
            next_image = image_array[i+1]

            this_histogram_r = cv2.calcHist([this_image],[0],None,[256],[0,256]) 
            this_histogram_g = cv2.calcHist([this_image],[1],None,[256],[0,256]) 
            this_histogram_b = cv2.calcHist([this_image],[2],None,[256],[0,256]) 
            next_histogram_r = cv2.calcHist([next_image],[0],None,[256],[0,256]) 
            next_histogram_g = cv2.calcHist([next_image],[1],None,[256],[0,256]) 
            next_histogram_b = cv2.calcHist([next_image],[2],None,[256],[0,256]) 

            CHD = (np.absolute(this_histogram_r - next_histogram_r) + np.absolute(this_histogram_g - next_histogram_g) + np.absolute(this_histogram_b - next_histogram_b)).sum()/(M*N)
            histogram_array = np.append(histogram_array, CHD)
            if CHD > threshold:
                print(i+1, CHD)
                answer_histogram = np.append(answer_histogram, i+1)

    return histogram_array, answer_histogram
            
    
def likelihood(image_array, array_len, threshold):
    answer_likehood = np.array([])
    M, N = image_array.shape[1] ,  image_array.shape[2]

    for i in range(array_len-1):
        #this_image = np.reshape(image_array[i], (M*N)).astype('int32')
        #next_image = np.reshape(image_array[i+1], (M*N)).astype('int32')

        this_image = image_array[i].astype('int32')
        next_image = image_array[i+1].astype('int32')

        threshold_block = 0
        for j in range(16):
            m_block_start = (int)(M * (j / 16))
            m_block_end = (int)(M * ((j+1) / 16))
            n_block_start = (int)(N * (j / 16))
            n_block_end = (int)(N * ((j+1) / 16))

            S1 = np.std(this_image[m_block_start:m_block_end, n_block_start:n_block_end])
            S2 = np.std(next_image[m_block_start:m_block_end, n_block_start:n_block_end])
            M1 = np.mean(this_image[m_block_start:m_block_end, n_block_start:n_block_end])
            M2 = np.mean(next_image[m_block_start:m_block_end, n_block_start:n_block_end])

            """
            print('m_start m_end n_start n_end')
            print(m_block_start, m_block_end, n_block_start, n_block_end)
            print('S1 S2 M1 M2')
            print(S1, S2, M1, M2)
            """
            
            if S1 * S2 == 0:
                S1 = 1
                S2 = 1


            likelihood_ratio = pow(((S1 + S2)/2 + pow(((M1 - M2)/2), 2)), 2) / (S1 * S2)
            if likelihood_ratio > threshold:
                threshold_block += 1
            #print('likehood ration : ' + str(likelihood_ratio))

        if threshold_block >= 10 :
            print(i+1, threshold_block)
            answer_likehood = np.append(answer_likehood, i+1)
        
        #print(likelihood_ratio)
    return answer_likehood
        

def edge_change(image_array, array_len, threshold):
    answer_edge = np.array([])
    #M, N = image_array.shape[1] ,  image_array.shape[2]
    edge_array = np.array([])
    for i in range(array_len-1):
        this_image = image_array[i]
        next_image = image_array[i+1]

        this_canny = cv2.Canny(this_image, 30, 150).astype('int32')
        next_canny = cv2.Canny(next_image, 30, 150).astype('int32')

        """
        cv2.imshow('My Image', this_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        A1 = np.count_nonzero(this_canny) 
        A2 = np.count_nonzero(next_canny) 
        if A1 == 0:
            A1 = 1
        if A2 == 0:
            A2 = 1

        difference = this_canny - next_canny
        Xin = np.count_nonzero(difference == -255)
        Xout = np.count_nonzero(difference == 255)

        ECR = max(Xin/A2, Xout/A1)
        edge_array = np.append(edge_array, ECR)
        if ECR > threshold:
            print(i+1, ECR)
            answer_edge = np.append(answer_edge, i+1)

        #f.writelines(str(difference.tolist()) + '\r\n')
    return edge_array, answer_edge

"""
def moving_vector(image_array, array_len):
    M, N = image_array.shape[1] ,  image_array.shape[2]
    for i in range(array_len-1):
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        color = np.random.randint(0,255,(100,3))
        # Take first frame and find corners in it
        old_frame = image_array[i]
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

        frame = image_array[i+1]
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        f.writelines(str(i+1) + '\r\n')  
        f.writelines('good_new\r\n')
        f.writelines(str(good_new.shape) + '\r\n')

        f.writelines('good_old\r\n')
        f.writelines(str(good_old.shape) + '\r\n')

        # Now update the previous frame and previous points

        #_ = input("change`:")
"""

def twin_comparison(histogram_array, threshold1, threshold2, gray_flag, edge_flag):
    answer_twin = np.array([])
    remember_i = -1
    accumulation_diff = 0
    for i in range(len(histogram_array)):
        if remember_i == -1:
            if histogram_array[i] >= threshold2 and histogram_array[i] < threshold1:
                remember_i = i + 1
                accumulation_diff = histogram_array[i]
            elif histogram_array[i] >= threshold1:
                print(i+1, histogram_array[i])
                answer_twin = np.append(answer_twin, i+1)
        elif remember_i != -1:
            if histogram_array[i] >= threshold2:
                accumulation_diff += histogram_array[i] - threshold2
            elif histogram_array[i] < threshold2 or i == len(histogram_array):
                if accumulation_diff >= threshold1:
                    print(str(remember_i) + '~' + str(i+1))
                    for j in range(remember_i, i+2, 1):
                        answer_twin = np.append(answer_twin, j)
                remember_i = -1
                accumulation_diff = 0 
    return answer_twin
                
def evaluation(input_answer_array, video_flag):
    r = open("out_video" + str(video_flag) + ".txt", "a")

    input_answer_array = input_answer_array.astype(int)

    if video_flag == 1:
        total_frame_number = 1379
    elif video_flag == 2:
        total_frame_number = 864
    elif video_flag == 3:
        total_frame_number = 1059

    #1379 frame
    news_ground = np.array([73, 235, 301, 370, 452, 861, 1281])
    #864 frame
    soccer_ground = np.array([89, 90, 91, 92, 93, 94, 95, 96, 378, 379, 380, 381, 382, 383, 384, 385, 567, 568, 569, 570, 571, 572, 573])
    #1059 frame
    ngc_ground = np.array([285, 340, 383, 456, 683, 703, 722, 859, 868, 876, 885, 897, 909, 921, 933, 943, 958, 963, 965, 969, 976, 986, 1038])
    for i in range(127, 165, 1):
        ngc_ground = np.append(ngc_ground, i)
    for i in range(196, 254, 1):
        ngc_ground = np.append(ngc_ground, i)
    for i in range(384, 445, 1):
        ngc_ground = np.append(ngc_ground, i)
    for i in range(516, 536, 1):
        ngc_ground = np.append(ngc_ground, i)
    for i in range(540, 574, 1):
        ngc_ground = np.append(ngc_ground, i)
    for i in range(573, 623, 1):
        ngc_ground = np.append(ngc_ground, i)
    for i in range(622, 665, 1):
        ngc_ground = np.append(ngc_ground, i)
    for i in range(728, 749, 1):
        ngc_ground = np.append(ngc_ground, i)
    for i in range(760, 816, 1):
        ngc_ground = np.append(ngc_ground, i)
    for i in range(816, 839, 1):
        ngc_ground = np.append(ngc_ground, i)
    for i in range(840, 852, 1):
        ngc_ground = np.append(ngc_ground, i)
    for i in range(1003, 1010, 1):
        ngc_ground = np.append(ngc_ground, i)
    for i in range(1048, 1060, 1):
        ngc_ground = np.append(ngc_ground, i)

    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(input_answer_array)):
        if video_flag == 1:
            if input_answer_array[i] in news_ground:
                TP += 1
            if input_answer_array[i] not in news_ground:
                FP += 1
        elif video_flag == 2:
            if input_answer_array[i] in soccer_ground:
                TP += 1
            if input_answer_array[i] not in soccer_ground:
                FP += 1
        elif video_flag == 3:
            if input_answer_array[i] in ngc_ground:
                TP += 1
            if input_answer_array[i] not in ngc_ground:
                FP += 1
    for i in range(1, total_frame_number, 1):
        if video_flag == 1:
            if i not in input_answer_array and i not in news_ground:
                TN += 1
        elif video_flag == 2:
            if i not in input_answer_array and i not in soccer_ground:
                TN += 1
        elif video_flag == 3:
            if i not in input_answer_array and i not in ngc_ground:
                TN += 1
    FN = total_frame_number - TP - FP - TN
    if TP + FP != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0
    if TP + FN != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0
    if TP + FN != 0:
        truth_rate = TP / (TP + FN)
    else:
        truth_rate = 0
    if FP + TN != 0:
        false_rate = FP / (FP + TN)
    else:
        false_rate = 0

    print('TP: ' + str(TP) + ' , FP: ' + str(FP) + ', TN: ' + str(TN) + ', FN: ' + str(FN) + '\n')
    #print('precision: ' + str(precision) + ' , recall: ' + str(recall) + ', truth_rate: ' + str(truth_rate) + ', false_rate: ' + str(false_rate) + '\r\n')

    r.writelines(str(precision) + ', ' + str(recall) + ', ' + str(truth_rate) + ', ' + str(false_rate) + '\r\n')

    return precision, recall, truth_rate, false_rate

def use_algorithm(algorithm, input_image, video_number):
    precision_array = np.array([])
    recall_array = np.array([])
    truth_rate_array = np.array([])
    false_rate_array = np.array([])
    
    precision_array_twin = np.array([])
    recall_array_twin = np.array([])
    truth_rate_array_twin = np.array([])
    false_rate_array_twin = np.array([])
    if algorithm == 'parewise':
        print('parewise')
        if video_number == 1:
            range0 = 0
            range1 = 800
            range2 = 8
        elif video_number == 2:
            range0 = 0
            range1 = 800
            range2 = 8
        elif video_number == 3:
            range0 = 0
            range1 = 1000
            range2 = 10
        for threshold2 in range(range0, range1, range2):
            threshold1 = 42
            answer_parewise = parewise(input_image.image_array, input_image.array_len, True, threshold1, threshold2/10)
            return_precision, return_recall, return_truth, return_false = evaluation(answer_parewise, video_number)
            precision_array = np.append(precision_array, return_precision)
            recall_array = np.append(recall_array, return_recall)
            truth_rate_array = np.append(truth_rate_array, return_truth)
            false_rate_array = np.append(false_rate_array, return_false)

        output_file = open("video" + str(video_number) + "file_parewise.txt", "w")

    elif algorithm == 'parewise_color':
        print('parewise_color')
        if video_number == 1:
            range0 = 0
            range1 = 900
            range2 = 9
        elif video_number == 2:
            range0 = 0
            range1 = 800
            range2 = 8
        elif video_number == 3:
            range0 = 0
            range1 = 900
            range2 = 9
        for threshold2 in range(range0, range1, range2):
            threshold1 = 42
            answer_parewise_color = parewise(input_image.image_array, input_image.array_len, False, threshold1, threshold2/10)
            return_precision, return_recall, return_truth, return_false = evaluation(answer_parewise_color, video_number)
            precision_array = np.append(precision_array, return_precision)
            recall_array = np.append(recall_array, return_recall)
            truth_rate_array = np.append(truth_rate_array, return_truth)
            false_rate_array = np.append(false_rate_array, return_false)

        output_file = open("video" + str(video_number) + "file_parewise_color.txt", "w")

    elif algorithm == 'histogram & twin':
        print('histogram and twin')
        if video_number == 1:
            range0 = 400
            range1 = 2600
            range2 = 22
        elif video_number == 2:
            range0 = 400
            range1 = 1700
            range2 = 13
        elif video_number == 3:
            range0 = 300
            range1 = 5000
            range2 = 47
        for threshold in range(range0, range1, range2):
            gray_histogrom_array, answer_histogram = histogram(input_image.image_array, input_image.array_len, True, threshold/1000)
            return_precision, return_recall, return_truth, return_false = evaluation(answer_histogram, video_number)
            precision_array = np.append(precision_array, return_precision)
            recall_array = np.append(recall_array, return_recall)
            truth_rate_array = np.append(truth_rate_array, return_truth)
            false_rate_array = np.append(false_rate_array, return_false)
            
            answer_twin_histogram = twin_comparison(gray_histogrom_array, threshold/1000, max(threshold/1000 - 0.25, 0), True, False)
            return_precision, return_recall, return_truth, return_false = evaluation(answer_twin_histogram, video_number)
            precision_array_twin = np.append(precision_array_twin, return_precision)
            recall_array_twin = np.append(recall_array_twin, return_recall)
            truth_rate_array_twin = np.append(truth_rate_array_twin, return_truth)
            false_rate_array_twin = np.append(false_rate_array_twin, return_false)

        output_file = open("video" + str(video_number) + "file_histogram.txt", "w")
        output_file_twin = open("video" + str(video_number) + "file_histogram_twin.txt", "w")

    elif algorithm == 'histogram_color & twin':
        print('histogram_color and twin')
        if video_number == 1:
            range0 = 400
            range1 = 2600
            range2 = 22
        elif video_number == 2:
            range0 = 400
            range1 = 1600
            range2 = 12
        elif video_number == 3:
            range0 = 300
            range1 = 5000
            range2 = 47
        for threshold in range(range0, range1, range2):
            color_histogrom_array, answer_histogram_color = histogram(input_image.image_array, input_image.array_len, False, threshold/1000)
            return_precision, return_recall, return_truth, return_false = evaluation(answer_histogram_color, video_number)
            precision_array = np.append(precision_array, return_precision)
            recall_array = np.append(recall_array, return_recall)
            truth_rate_array = np.append(truth_rate_array, return_truth)
            false_rate_array = np.append(false_rate_array, return_false)

            answer_twin_histogram_color = twin_comparison(color_histogrom_array, threshold/1000, max(threshold/1000 - 0.25, 0), False, False)
            return_precision, return_recall, return_truth, return_false = evaluation(answer_twin_histogram_color, video_number)
            precision_array_twin = np.append(precision_array_twin, return_precision)
            recall_array_twin = np.append(recall_array_twin, return_recall)
            truth_rate_array_twin = np.append(truth_rate_array_twin, return_truth)
            false_rate_array_twin = np.append(false_rate_array_twin, return_false)

        output_file = open("video" + str(video_number) + "file_histogram_color.txt", "w")
        output_file_twin = open("video" + str(video_number) + "file_histogram_color_twin.txt", "w")

    elif algorithm == 'likelihood':
        print('likelihood')
        if video_number == 1:
            range0 = 1000
            range1 = 2000
            range2 = 10
        elif video_number == 2:
            range0 = 1000
            range1 = 2000
            range2 = 10
        elif video_number == 3:
            range0 = 900
            range1 = 3000
            range2 = 21
        for threshold in range(range0, range1, range2):
            answer_likehood = likelihood(input_image.image_array, input_image.array_len, threshold/1000)
            return_precision, return_recall, return_truth, return_false = evaluation(answer_likehood, video_number)
            precision_array = np.append(precision_array, return_precision)
            recall_array = np.append(recall_array, return_recall)
            truth_rate_array = np.append(truth_rate_array, return_truth)
            false_rate_array = np.append(false_rate_array, return_false)

        output_file = open("video" + str(video_number) + "file_likelihood.txt", "w")

    elif algorithm == 'edge & twin':
        print('edge')
        if video_number == 1:
            range0 = 100
            range1 = 600
            range2 = 5
        elif video_number == 2:
            range0 = 300
            range1 = 700
            range2 = 4
        elif video_number == 3:
            range0 = 200
            range1 = 1000
            range2 = 8
        for threshold in range(range0, range1, range2):
            edge_array, answer_edge = edge_change(input_image.image_array, input_image.array_len, threshold/1000)
            return_precision, return_recall, return_truth, return_false = evaluation(answer_edge, video_number)
            precision_array = np.append(precision_array, return_precision)
            recall_array = np.append(recall_array, return_recall)
            truth_rate_array = np.append(truth_rate_array, return_truth)
            false_rate_array = np.append(false_rate_array, return_false)

            answer_twin_edge = twin_comparison(edge_array, threshold/1000, max(threshold/1000 - 0.1, 0), True, True)
            return_precision, return_recall, return_truth, return_false = evaluation(answer_twin_edge, video_number)
            precision_array_twin = np.append(precision_array_twin, return_precision)
            recall_array_twin = np.append(recall_array_twin, return_recall)
            truth_rate_array_twin = np.append(truth_rate_array_twin, return_truth)
            false_rate_array_twin = np.append(false_rate_array_twin, return_false)

        output_file = open("video" + str(video_number) + "file_edge.txt", "w")
        output_file_twin = open("video" + str(video_number) + "file_edge_twin.txt", "w")

    output_file.writelines('recall_array\r\n')
    output_file.writelines(str(recall_array.tolist()) +'\r\n')
    output_file.writelines('precision_array\r\n')
    output_file.writelines(str(precision_array.tolist()) +'\r\n')
    output_file.writelines('false_rate_array\r\n')
    output_file.writelines(str(false_rate_array.tolist()) +'\r\n')
    output_file.writelines('truth_rate_array\r\n')
    output_file.writelines(str(truth_rate_array.tolist()) +'\r\n')

    if 'twin' in algorithm:
        output_file_twin.writelines('recall_array\r\n')
        output_file_twin.writelines(str(recall_array_twin.tolist()) +'\r\n')
        output_file_twin.writelines('precision_array\r\n')
        output_file_twin.writelines(str(precision_array_twin.tolist()) +'\r\n')
        output_file_twin.writelines('false_rate_array\r\n')
        output_file_twin.writelines(str(false_rate_array_twin.tolist()) +'\r\n')
        output_file_twin.writelines('truth_rate_array\r\n')
        output_file_twin.writelines(str(truth_rate_array_twin.tolist()) +'\r\n')

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)

    input_image_gray = image_dataset('news.mpg', True)
    input_image_color = image_dataset('news.mpg', False)

    video_number = 1

    use_algorithm('parewise', input_image_gray, video_number)
    use_algorithm('parewise_color', input_image_color, video_number)
    use_algorithm('histogram & twin', input_image_gray, video_number)
    use_algorithm('histogram_color & twin', input_image_color, video_number)
    use_algorithm('likelihood', input_image_gray, video_number)
    use_algorithm('edge & twin', input_image_gray, video_number)

    #print(input_image_gray.image_array.shape)
    #print(input_image_gray.array_len)
    #print(input_image_color.image_array.shape)
    #print(input_image_color.array_len)

    #answer_parewise = parewise(input_image_gray.image_array, input_image_gray.array_len, True, 42, 50)
    #gray_histogrom_array, answer_histogram = histogram(input_image_gray.image_array, input_image_gray.array_len, True, 1.0)
    #color_histogrom_array, answer_histogram_color = histogram(input_image_color.image_array, input_image_color.array_len, False, 1.0)
    #answer_likelihood = likelihood(input_image_gray.image_array, input_image_gray.array_len, 1.5)
    #edge_array, answer_edge = edge_change(input_image_gray.image_array, input_image_gray.array_len, 0.6)
    #answer_edge_twin = twin_comparison(edge_array, 0.6, 0.5, True, True)

    #parewise(input_image_color.image_array, input_image_color.array_len, False, 42, 50)
    #gray_histogrom_array = histogram(input_image_gray.image_array, input_image_gray.array_len, True, 1.0)
    #twin_comparison(gray_histogrom_array, 1.0, 0.8, True, False)
    #color_histogrom_array = histogram(input_image_color.image_array, input_image_color.array_len, False, 1.0)
    #twin_comparison(color_histogrom_array, 1.0, 0.8, True, False)
    #likelihood(input_image_gray.image_array, input_image_gray.array_len, 1.5)
    #edge_array = edge_change(input_image_gray.image_array, input_image_gray.array_len, 0.5)
    #twin_comparison(edge_array, 0.5, 0.4, True, True)

    #print(answer_edge_twin)
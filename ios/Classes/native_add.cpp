#include <opencv2/opencv.hpp>
#include <chrono>

#include<opencv2/core/core.hpp>
#include<opencv2/core/types_c.h>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<cmath>
#include <string>
#include<iostream>

#ifdef __ANDROID__
#include <android/log.h>
#endif

using namespace cv;
using namespace std;


long long int get_now() {
    return chrono::duration_cast<std::chrono::milliseconds>(
            chrono::system_clock::now().time_since_epoch()
    ).count();
}

//定义结构体
struct HSV_pixel {
    int H = 0;
    float S = 0;
    float V = 0;
}HSV_PIXEL;

void platform_log(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
#ifdef __ANDROID__
    __android_log_vprint(ANDROID_LOG_VERBOSE, "ndk", fmt, args);
#else
    vprintf(fmt, args);
#endif
    va_end(args);
}

// 过滤得到每个连续块的行或者列
vector<vector<int>> filterVector(vector<int> vector_before) {
    vector<vector<int>> vec;
    vector<int> vec_tmp;
    int vec_size = vector_before.size();
    for (int i = 0; i < (vec_size - 1); i++) {
        if (vec_tmp.size() == 0) {
            vec_tmp.push_back(vector_before[i]); //首先将首字母的第一个送进去
        }
        if ((vector_before[i] + 1) == vector_before[i+1]) {
            vec_tmp.push_back(vector_before[i+1]);
        }
        else {
            //if (vec_tmp.size() > 1) { //如果只有一行则不保存到里面，如果一行也要保存 删除本行
            vec.push_back(vec_tmp);
            //}
            vec_tmp.clear();
        }
    }
    if (vec.begin() == vec.end() or vec_tmp != vec.back()) {
        vec.push_back(vec_tmp);
        vec_tmp.clear();
    }
    return vec;
}

//判断元素是否存在在vector当中
bool isContainVec(vector<int> vector_before, int item) {
    vector<int>::iterator it;
    it = find(vector_before.begin(), vector_before.end(), item);
    if (it != vector_before.end())
    {
        return true;
    }
    else
    {
        return false;
    }
}

//RGB转为HSV 返回结果为[H,S,V]
vector<float> RGB2HSV(int red, int green, int blue) {
    vector<float> vec_tmp;
    vec_tmp.push_back(float((float)red * 2));
    vec_tmp.push_back(float((float)green / 255));
    vec_tmp.push_back(float((float)blue / 255));
    return vec_tmp;
}

void reduce_neverless_block(vector<vector<int>> vec, int*** one_hot_matrix) {
    for (int i = 0; i < vec.size(); i++) {
        *(one_hot_matrix[vec[i][0]][vec[i][1]]) = 0;
    }
}


void drawImg(int image_height, int image_width, int*** one_hot_matrix) {
    printf("\n\n\n");
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            printf("%d", *(one_hot_matrix[i][j]));
        }
        printf("\n");
    }
}

// Avoiding name mangling
extern "C" {
// Attributes to prevent 'unused' function from being removed and to make it visible
__attribute__((visibility("default"))) __attribute__((used))
int32_t native_add(int32_t x, int32_t y) {
    cv::Mat m = cv::Mat::zeros(x, y, CV_8UC3);
    return m.rows + m.cols;
}

__attribute__((visibility("default"))) __attribute__((used))
const char* version() {
    return CV_VERSION;
}

__attribute__((visibility("default"))) __attribute__((used))
int32_t process_image(char* image_url) {

    IplImage* image_original = cvLoadImage(image_url);

    //缩放图片
    IplImage* image;
    CvSize size;
    //将图片大小统一处理图片宽度为200pix大小
    int orginal_image_height = image_original->height;
    int orginal_image_width = image_original->width;
    if (orginal_image_width < 200) {
        int return_value = -2;
        printf("-2 图片不清晰，请重拍");
        return return_value;
    }
    //缩放的倍数
    double scale = 0.3; // double(200 / orginal_image_width);

    size.width = image_original->width * scale;
    size.height = image_original->height * scale;
    image = cvCreateImage(size, image_original->depth, image_original->nChannels);
    cvResize(image_original, image, CV_INTER_CUBIC);

    if (image == NULL) {    //open image error
        printf("打开图像文件失败!");
        return -1;
    }
    return 10;
    IplImage* image1 = cvCreateImage(cvSize(image->width, image->height), image->depth, image->nChannels);  //注意图像必须和输入图像的size，颜色位深度，通道一致
    cvZero(image1); //清空image_data数据
    cvCvtColor(image, image1, CV_BGR2HSV);//CV_BGR2HSV
    CvScalar scalar;    //scalar


    //获取二维的HSV矩阵
    int image_height = image1->height;
    int image_width = image1->width;
    struct HSV_pixel*** HSV_Arr = (struct HSV_pixel***)malloc(image_height * sizeof(struct treeBranch**));
    for (int i = 0; i < image_height; i++) {    //HSV
        struct HSV_pixel** HSV_Arr_1 = (struct HSV_pixel**)malloc(image_width * sizeof(struct treeBranch*));
        HSV_Arr[i] = HSV_Arr_1;
        for (int j = 0; j < image_width; j++) {
            struct HSV_pixel* HSV_Arr_2 = (struct HSV_pixel*)malloc(sizeof(struct HSV_pixel));
            scalar = cvGet2D(image1, i, j);    //获取像素点的RGB颜色分量
            //printf("第%d个像素的H：%d°，S：%lf，V：%lf\n", (i * image->width) + j, (int)scalar.val[0] * 2, scalar.val[1] / 255, scalar.val[2] / 255);
            HSV_Arr_2->H = (int)scalar.val[0] * 2;
            HSV_Arr_2->S = scalar.val[1] / 255;
            HSV_Arr_2->V = scalar.val[2] / 255;
            HSV_Arr[i][j] = HSV_Arr_2;
            //getchar();    //防止打印速度太快，暂停一次打印一次
        }
    }


    //生产01矩阵
    int*** one_hot_matrix = (int***)malloc(image_height * sizeof(int**));
    vector<vector<int>> heightValidList;  // 有效数组的集合 比如 [[1,2,3][7,8][11,12,14][17,18]]
    vector<vector<int>> widthValidList;
    vector<int>heightValidLines;  // 有效数组的列表
    vector<int>widthValidLines;
    for (int i = 0; i < image_height; i++) {
        int** one_hot_line = (int**)malloc(image_width * sizeof(int*));
        one_hot_matrix[i] = one_hot_line;
        for (int j = 0; j < image_width; j++) {
            int* one_hot = (int*)malloc(sizeof(int));
            if (((HSV_Arr[i][j]->H >= 0 and HSV_Arr[i][j]->H <= 60) or (HSV_Arr[i][j]->H >= 300 and HSV_Arr[i][j]->H <= 360)) and HSV_Arr[i][j]->S >= 0.1 and HSV_Arr[i][j]->V >= 0.3) {
                *one_hot = 1;
            }
            else {
                *one_hot = 0;
            }
            one_hot_matrix[i][j] = one_hot;
            //printf("第%d个像素的H：%d°，S：%lf，V：%lf，对应的O1值为：%d\n", (i* image_width) + j, HSV_Arr[i][j]->H, HSV_Arr[i][j]->S, HSV_Arr[i][j]->V, *one_hot);
        }
    }



    drawImg(image_height, image_width, one_hot_matrix);

    printf("总共有%d行，%d列", image_height, image_width);

    // 过滤图片中一些不符合条件的行
    for (int i = 0; i < image_height; i++) {
        int line_zero_count = 0;
        int line_one_count = 0;
        float proportion_one = 0;
        for (int j = 0; j < image_width; j++) {
            if (*(one_hot_matrix[i][j]) == 0) {
                line_zero_count = line_zero_count + 1;
            }
            else {
                line_one_count = line_one_count + 1;
            }
        }
        proportion_one = (float)line_one_count / ((float)line_one_count + (float)line_zero_count);
        if (proportion_one >= 0.05) {
            heightValidLines.push_back(i);
        }
    }
    heightValidList = filterVector(heightValidLines);
    // 过滤列

    for (int j = 0; j < image_width; j++) { // 列单位
        int col_zero_count = 0;
        int col_one_count = 0;
        float proportion_one = 0;
        for (int k = 0; k < heightValidList.size(); k++) {
            for (int i = 0; i < heightValidList[k].size(); i++) { // 行单位
                if (*(one_hot_matrix[(heightValidList[k][i])][j]) == 0) {
                    col_zero_count = col_zero_count + 1;
                }
                else {
                    col_one_count = col_one_count + 1;
                }
            }
        }
        proportion_one = (float)col_one_count / ((float)col_one_count + (float)col_zero_count);
        if (proportion_one >= 0.2) { // 如果列中1的比例大于0.2 则进入候选的有效段
            widthValidLines.push_back(j);
        }
    }
    //将有效的列处理成连续集合的形式  [1,2,3,5,8,9,10] => [[1,2,3][5][8,9,10]]
    widthValidList = filterVector(widthValidLines);

    //----------将行跟列串成连续的数据块，并过滤出有效连续数据块-----------  [[[2,3][2,4][2,5][3,3]...]]

    vector<vector<vector<int>>> vec;
    for (int k = 0; k < heightValidList.size(); k++) {
        for (int g = 0; g < widthValidList.size(); g++) {
            vector<vector<int>> vec1;  // 取每块的
            for (int i = 0; i < heightValidList[k].size(); i++) {
                for (int j = 0; j < widthValidList[g].size(); j++) {
                    vector<int> vec2;
                    vec2.push_back(heightValidList[k][i]);
                    vec2.push_back(widthValidList[g][j]);
                    vec1.push_back(vec2);
                }
            }
            vec.push_back(vec1);
        }
    }

    // 将连续数据块中整行的0过滤掉
    for (int i = 0; i < vec.size(); i++) {
        int height_index_begin = vec[i][0][0];
        int height_index_end = vec[i][(vec[i].size() - 1)][0];
        int width_index_begin = vec[i][0][1];
        int width_index_end = vec[i][(vec[i].size() - 1)][1];
        int each_width = width_index_end - width_index_begin + 1; //每行有多少个元素
        for (int j = height_index_end; j >= height_index_begin; j--) { //行
            bool flag_width = true;//如果全部为0 则在vec中删除对应行
            for (int k = width_index_end; k >= width_index_begin; k--) { //列
                if (*(one_hot_matrix[j][k]) == 1) {
                    flag_width = false;
                    break;
                }
            }
            if (flag_width == true) {
                //执行删除vec中对应行的操作
                for (int k = width_index_end; k >= width_index_begin; k--) { //列
                    int delete_index = ((j - height_index_begin) * each_width) + (k - width_index_begin);//获取要删除的元素对应的vec[i]中对应第几个
                    vec[i].erase(remove(vec[i].begin(), vec[i].end(), vec[i][delete_index]), vec[i].end()); //删除vec中值为vec[i]的元素
                }
            }
        }
    }
    // 将连续数据块中整列的0过滤掉
    for (int i = 0; i < vec.size(); i++) {
        int height_index_begin = vec[i][0][0];
        int height_index_end = vec[i][(vec[i].size() - 1)][0];
        int width_index_begin = vec[i][0][1];
        int width_index_end = vec[i][(vec[i].size() - 1)][1];
        int each_width = width_index_end - width_index_begin + 1; //每列有多少个元素
        for (int k = width_index_end; k >= width_index_begin; k--) { //列
            bool flag_height = true;//如果全部为0 则在vec中删除对应列
            for (int j = height_index_end; j >= height_index_begin; j--) { //行
                if (*(one_hot_matrix[j][k]) == 1) {
                    flag_height = false;
                    break;
                }
            }
            if (flag_height == true) {
                //执行删除vec中对应行的操作
                for (int j = height_index_end; j >= height_index_begin; j--) {
                    int delete_index = ((j - height_index_begin) * each_width) + (k - width_index_begin);//获取要删除的元素对应的vec[i]中对应第几个
                    vec[i].erase(remove(vec[i].begin(), vec[i].end(), vec[i][delete_index]), vec[i].end()); //删除vec中值为vec[i]的元素
                }
            }
        }
    }

    //根据地方去除一些杂块 只取中间位置
    for (int i = vec.size() - 1; i >= 0; i--) {
        for (int j = 0; j < vec[i].size(); j++) {
            //如果某一个连续块处于图片中间50%范围以外，则排除此连续块
            if ((vec[i][j][1] < image_width * 0.25) || (vec[i][j][1] > image_width * 0.75)) {
                reduce_neverless_block(vec[i], one_hot_matrix);
                vec.erase(remove(vec.begin(), vec.end(), vec[i]), vec.end()); //删除vec中值为vec[i]的元素
                break;
            }
        }
    }
    drawImg(image_height, image_width, one_hot_matrix);

    //两两对比 找到一对相似度高的区域块 同时排除最大的那块 标记为1的数量小于总像素块数量的2％
    vector<vector<int>> vec_match_pairs;  // [[1,2][5,6]]
    for (int i = 0; i < vec.size() - 1; i++) {
        int count_one_i = 0;//获取该数据块总共有多少个1
        int count_one_i_height = 0;//    获取该数据块总共含有1的行数有几个
        int count_one_i_width = 0;//    获取该数据块总共含有1的列数有几个
        //    计算数据块总共有多少个1
        vector<int> used_i_height_nums;
        vector<int> used_i_width_nums;
        for (int o = 0; o < vec[i].size(); o++) {
            if (*(one_hot_matrix[vec[i][o][0]][vec[i][o][1]]) == 1) {
                count_one_i = count_one_i + 1;
                if (!isContainVec(used_i_height_nums, vec[i][o][0])) {
                    used_i_height_nums.push_back(vec[i][o][0]);
                    count_one_i_height = count_one_i_height + 1;
                }
                if (!isContainVec(used_i_width_nums, vec[i][o][1])) {
                    used_i_width_nums.push_back(vec[i][o][1]);
                    count_one_i_width = count_one_i_width + 1;
                }
            }
        }

        for (int j = i + 1; j < vec.size(); j++) {
            int count_one_j = 0;//获取该数据块总共有多少个1
            int count_one_j_height = 0;//获取该数据块总共含有1的行数有几个
            int count_one_j_width = 0;//获取该数据块总共含有1的列数有几个
            bool flagTotalCount = false; //判断总数是否符合
            bool flagTotalRow = false; //判断行数是否符合
            bool flagTotalCol = false; //判断列数是否符合

            //计算数据块总共有多少个1
            vector<int> used_j_height_nums;
            vector<int> used_j_width_nums;
            for (int o = 0; o < vec[j].size(); o++) {
                if (*(one_hot_matrix[vec[j][o][0]][vec[j][o][1]]) == 1) {
                    count_one_j = count_one_j + 1;
                    if (!isContainVec(used_j_height_nums, vec[j][o][0])) {
                        used_j_height_nums.push_back(vec[j][o][0]);
                        count_one_j_height = count_one_j_height + 1;
                    }
                    if (!isContainVec(used_j_width_nums, vec[j][o][1])) {
                        used_j_width_nums.push_back(vec[j][o][1]);
                        count_one_j_width = count_one_j_width + 1;
                    }
                }
            }

            //判断总数是否符合
            if (float(float(count_one_i) / float(count_one_j)) >= 0.6 and (float(count_one_i) / float(count_one_j)) <= 1.66) {
                flagTotalCount = true;
            }
            //判断行数是否符合
            if (float(float(count_one_i_height) / float(count_one_j_height)) >= 0.6 and (float(count_one_i_height) / float(count_one_j_height)) <= 1.66) {
                flagTotalRow = true;
            }
            //判断列数是否符合
            if (float(float(count_one_i_width) / float(count_one_j_width)) >= 0.6 and (float(count_one_i_width) / float(count_one_j_width)) <= 1.66) {
                flagTotalCol = true;
            }
            if (flagTotalCount && flagTotalRow && flagTotalCol) {
                vector<int > vec_match_pair; //记录
                vec_match_pair.push_back(i);
                vec_match_pair.push_back(j);
                vec_match_pairs.push_back(vec_match_pair);
            }
        }
    }
    int return_value = -1;

    if (vec_match_pairs.size() == 1) {
        //结果对应为1-8的颜色转为HSV
        vector<float> HSV_1 = RGB2HSV(254, 254, 254);
        vector<float> HSV_2 = RGB2HSV(252, 180, 209);
        vector<float> HSV_3 = RGB2HSV(241, 128, 177);
        vector<float> HSV_4 = RGB2HSV(234, 69, 141);
        vector<float> HSV_5 = RGB2HSV(218, 39, 117);
        vector<float> HSV_6 = RGB2HSV(190, 38, 105);
        vector<float> HSV_7 = RGB2HSV(159, 36, 92);
        vector<float> HSV_8 = RGB2HSV(139, 29, 76);

        //将左边的块定为T 右边的块定为C
        int block_t = 0; //T线在vec中对应的index
        int block_c = 0; //C线在vec中对应的index
        if (vec[vec_match_pairs[0][0]][0][1] > vec[vec_match_pairs[0][1]][0][1]) {
            block_c = vec_match_pairs[0][0];
            block_t = vec_match_pairs[0][1];
        }


        // vec => [[[2,3][2,4][2,5][3,3]...]]
        vector<vector<int>> vec_block_c = vec[block_c];
        vector<vector<int>> vec_block_t = vec[block_t];
        // 精确到每个点
        float vec_block_c_h_sum = 0;
        float vec_block_c_s_sum = 0;
        float vec_block_c_v_sum = 0;
        float vec_block_c_h_avg = 0;
        float vec_block_c_s_avg = 0;
        float vec_block_c_v_avg = 0;
        float vec_block_t_h_sum = 0;
        float vec_block_t_s_sum = 0;
        float vec_block_t_v_sum = 0;
        float vec_block_t_h_avg = 0;
        float vec_block_t_s_avg = 0;
        float vec_block_t_v_avg = 0;
        for (int l = 0; l < vec_block_t.size(); l++) {
            //HSV_Arr[i][j]->H
            int index_height = vec_block_t[l][0];
            int index_width = vec_block_t[l][1];
            vec_block_t_h_sum = vec_block_t_h_sum + float(HSV_Arr[index_height][index_width]->H);
            vec_block_t_s_sum = vec_block_t_s_sum + float(HSV_Arr[index_height][index_width]->S);
            vec_block_t_v_sum = vec_block_t_v_sum + float(HSV_Arr[index_height][index_width]->V);
        }
        vec_block_t_h_avg = vec_block_t_h_sum / float(vec_block_t.size());
        vec_block_t_s_avg = vec_block_t_s_sum / float(vec_block_t.size());
        vec_block_t_v_avg = vec_block_t_v_sum / float(vec_block_t.size());

        //for (int l = 0; l < vec_block_y.size(); l++) {
        //    //HSV_Arr[i][j]->H
        //    int index_height = vec_block_y[l][0];
        //    int index_width = vec_block_y[1][1];
        //    vec_block_y_h_sum = vec_block_y_h_sum + float(HSV_Arr[index_height][index_width]->H);
        //    vec_block_y_s_sum = vec_block_y_s_sum + float(HSV_Arr[index_height][index_width]->S);
        //    vec_block_y_v_sum = vec_block_y_v_sum + float(HSV_Arr[index_height][index_width]->V);
        //}
        //vec_block_y_h_avg = vec_block_y_h_sum / float(vec_block_y.size());
        //vec_block_y_s_avg = vec_block_y_s_sum / float(vec_block_y.size());
        //vec_block_y_v_avg = vec_block_y_v_sum / float(vec_block_y.size());

        //饱和度差值 根据这个值定试纸检测结果
        if (vec_block_t_s_avg <= HSV_8[0]) {
            printf("8 强阳");
            return_value = 8;
        }
        else if (vec_block_t_s_avg <= HSV_7[0]) {
            printf("7 阳");
            return_value = 7;
        }
        else if (vec_block_t_s_avg <= HSV_6[0]) {
            printf("6 阳");
            return_value = 6;
        }
        else if (vec_block_t_s_avg <= HSV_5[0]) {
            printf("5 弱阳");
            return_value = 5;
        }
        else if (vec_block_t_s_avg <= HSV_4[0]) {
            printf("4 弱阳");
            return_value = 4;
        }
        else if (vec_block_t_s_avg <= HSV_3[0]) {
            printf("3 弱阳");
            return_value = 3;
        }
        else if (vec_block_t_s_avg <= HSV_2[0]) {
            printf("2 弱阳");
            return_value = 2;
        }
            /*else if (vec_block_t_s_avg <= HSV_8[0]) {
                printf("1 阴");
                return_value = 1;
            }*/
        else {
            printf("1 阴");
            return_value = 1;
        }
    }
    else if (vec_match_pairs.size() == 0) {
        return_value = 0;
        printf("0 该试纸不存在C线 检测无效");
    }
    else {
        printf("-1 存在多个相似块");
    }
    free(HSV_Arr);
    return return_value;
}
}


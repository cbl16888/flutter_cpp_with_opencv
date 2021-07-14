#include <opencv2/opencv.hpp>
#include <chrono>

#include<opencv2/core/core.hpp>
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

int zoomWidthPix = 500; //要将图片缩放到500像素的宽度比例

//定义结构体
struct HSVPixel {
    float H = 0;
    float S = 0;
    float V = 0;
}HSVPIXEL;

// 过滤得到每个连续块的行或者列
vector<vector<int>> filterVector(vector<int> vectorBefore) {
    vector<vector<int>> vec;
    vector<int> vecTmp;
    int vec_size = vectorBefore.size();
    for (int i = 0; i < (vec_size - 1); i++) {
        if (vecTmp.size() == 0) {
            vecTmp.push_back(vectorBefore[i]); //首先将首字母的第一个送进去
        }
        if ((vectorBefore[i] + 1) == vectorBefore[i + 1]) {
            vecTmp.push_back(vectorBefore[i + 1]);
        }
        else {
            //if (vecTmp.size() > 1) { //如果只有一行则不保存到里面，如果一行也要保存 删除本行
            vec.push_back(vecTmp);
            //}
            vecTmp.clear();
        }
    }
    if (vec.begin() == vec.end() or vecTmp != vec.back()) {
        vec.push_back(vecTmp);
        vecTmp.clear();
    }
    return vec;
}

//判断元素是否存在在vector当中
bool isContainVec(vector<int> vectorBefore, int item) {
    vector<int>::iterator it;
    it = find(vectorBefore.begin(), vectorBefore.end(), item);
    if (it != vectorBefore.end())
    {
        return true;
    }
    else
    {
        return false;
    }
}

//将这个区间的01矩阵全部置为0
void reduceNeverlessBlock(vector<vector<int>> vec, int*** oneHotMatrix) {
    for (int i = 0; i < vec.size(); i++) {
        *(oneHotMatrix[vec[i][0]][vec[i][1]]) = 0;
    }
}

//校验矩阵是否有效
bool checkVecValidByNormal(vector<vector<int>> vec, int*** oneHotMatrix) {
    int oneCount = 0;
    int heightMax = 0;
    int WidthMax = 0;
    for (int i = 0; i < vec.size(); i++) {
        if (*(oneHotMatrix[vec[i][0]][vec[i][1]])==1){
            oneCount = oneCount + 1;
        }
        if (heightMax< vec[i][0]){
            heightMax = vec[i][0];
        }
        if (WidthMax< vec[i][1]){
            WidthMax = vec[i][1];
        }
    }
    if (oneCount>(vec.size()/10)){
        //如果长宽比>10 也视为不规范
        int colNum = WidthMax - vec[0][1];
        int rowNum = heightMax - vec[0][0];
        //if ((colNum)<(10*(rowNum))){
        if (2*colNum < rowNum) {
            return true;
        }
        else {
            return false;
        }
    }
    else {
        return false;
    }
}

//校验矩阵是否有效
bool checkVecValidBySemiQuan(vector<vector<int>> vec, int*** oneHotMatrix) {
    int oneCount = 0;
    int heightMax = 0;
    int WidthMax = 0;
    for (int i = 0; i < vec.size(); i++) {
        if (*(oneHotMatrix[vec[i][0]][vec[i][1]]) == 1) {
            oneCount = oneCount + 1;
        }
        if (heightMax < vec[i][0]) {
            heightMax = vec[i][0];
        }
        if (WidthMax < vec[i][1]) {
            WidthMax = vec[i][1];
        }
    }
    if (oneCount > (vec.size() / 2)) {
        //如果长宽比>10 也视为不规范
        if ((WidthMax - vec[0][1]) < (10 * (heightMax - vec[0][0]))) {
            return true;
        }
        else {
            return false;
        }
    }
    else {
        return false;
    }
}


//画图逻辑 输入图片的长宽跟01矩阵
void drawImg(int imageHeight, int imageWidth, int*** oneHotMatrix) {
    printf("\n\n\n");
    for (int i = 0; i < imageHeight; i++) {
        for (int j = 0; j < imageWidth; j++) {
            printf("%d", *(oneHotMatrix[i][j]));
        }
        printf("\n");
    }
}

//画Vec
void drawVec(int imageHeight, int imageWidth, int*** oneHotMatrix,vector<vector<vector<int>>> vec) {
    printf("\n\n\n");
    for (int i = 0; i < imageHeight; i++) {
        for (int j = 0; j < imageWidth; j++) {
            vector<int> temp;
            temp.push_back(i);
            temp.push_back(j);
            bool flag = false;
            for (int k = 0; k < vec.size(); k++) {
                if (count(vec[k].begin(), vec[k].end(), temp) && *(oneHotMatrix[i][j]) == 1) {
                    flag = true;
                }
            }
            if (flag)
            {
                printf("%d", 1);
            }
            else {
                printf("%d", 0);
            }
        }
        printf("\n");
    }
}

//求最大值
float retmax(float a, float b, float c)
{
    float max = 0;
    max = a;
    if (max < b)
        max = b;
    if (max < c)
        max = c;
    return max;
}

//求最小值
float retmin(float a, float b, float c)
{
    float min = 0;
    min = a;
    if (min > b)
        min = b;
    if (min > c)
        min = c;
    return min;
}

//获取角度
float getAngle(Vec4i line)
{
    float result = atan2(line[3] - line[1], line[2] - line[0]);
    return result;
}

//RGB转为HSI 返回结果为[H S I]
vector<float> RGB2HSI(float red, float green, float blue) {
    float H, S, I;

    // 归一化
    float b = blue / 255.0;
    float g = green / 255.0;
    float r = red / 255.0;

    float num = 0.5 * ((r - g) + (r - b));
    float den = sqrt((r - g) * (r - g) + (r - b) * (g - b));
    float theta = acos(num / den);

    if (den == 0) {
        H = 0; // 分母不能为0
    }
    else {
        if (b <= g) {
            H = theta;
        }
        else {
            H = (2 * 3.14159265 - theta);
        }
    }

    float min_RGB = min(min(b, g), r); // min(R,G,B)
    float sum = b + g + r;
    if (sum == 0)
    {
        S = 0;
    }
    else {
        S = 1 - 3 * min_RGB / sum;
    }

    I = sum / 3.0;
    H = H / (2 * 3.14159265);
    vector<float> vecTmp;
    vecTmp.push_back(H);
    vecTmp.push_back(S);
    vecTmp.push_back(I);
    return vecTmp;
}

//RGB转为HSV 返回结果为[H,S,V]
vector<float> RGB2HSV(float red, float green, float blue) {
    float h, s, v;

    float max = 0, min = 0;
    red = red / 255;
    green = green / 255;
    blue = blue / 255;

    max = retmax(red, green, blue);
    min = retmin(red, green, blue);
    v = max;
    if (max == 0)
        s = 0;
    else
        s = 1 - (min / max);
    if (max == min)
        h = 0;
    else if (max == red && green >= blue)
        h = 60 * ((green - blue) / (max - min));
    else if (max == red && green < blue)
        h = 60 * ((green - blue) / (max - min)) + 360;
    else if (max == green)
        h = 60 * ((blue - red) / (max - min)) + 120;
    else if (max == blue)
        h = 60 * ((red - green) / (max - min)) + 240;
    vector<float> vecTmp;
    vecTmp.push_back(h);
    vecTmp.push_back(s);
    vecTmp.push_back(v);
    return vecTmp;
}

static double angle(Point pt1, Point pt2, Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

//检测矩形
//第一个参数是传入的原始图像，第二是输出的图像。
void findSquares(const Mat& image, Mat& out)
{
    int threshold1 = 60, threshold2 = 180, N = 5;
    vector<vector<Point>> squares;
    squares.clear();
    Mat src, dst, gray_one, gray;
    src = image.clone();
    out = image.clone();
    int imageHeight = src.rows;
    int imageWidth = src.cols;
    gray_one = Mat(src.size(), CV_8U);
    //滤波增强边缘检测
    medianBlur(src, dst, 1);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    //在图像的每个颜色通道中查找矩形
    for (int c = 0; c < image.channels(); c++) {
        int ch[] = { c, 0 };
        //通道分离
        mixChannels(&dst, 1, &gray_one, 1, ch, 1);
        // 尝试几个阈值
        for (int l = 0; l < N; l++) {
            // 用canny()提取边缘
            if (l == 0) {
                //检测边缘
                Canny(gray_one, gray, threshold1, threshold2);

                ////运行Sobel算子
                //Mat xdst, ydst;
                //Sobel(gray_one, xdst, CV_64F, 1, 0);
                //Sobel(gray_one, ydst, CV_64F, 0, 1);
                //subtract(xdst, ydst, gray);
                //convertScaleAbs(ydst, gray);
                //threshold(gray, gray, 0, 255, THRESH_OTSU); //OTSU自适应

                //膨脹
                dilate(gray, gray, Mat(), Point(-1, -1));
                //imshow("dilate", gray);
            }
            else {
                gray = gray_one >= (l + 1) * 255 / N;
            }

            // 轮廓查找
            //findContours(gray, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
            findContours(gray, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            // 检测所找到的轮廓
            for (size_t i = 0; i < contours.size(); i++)
            {
                //使用图像轮廓点进行多边形拟合
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true) * 0.02, true);
                //计算轮廓面积后，得到矩形4个顶点
                if (approx.size() == 4 && fabs(contourArea(Mat(approx))) > 50 && isContourConvex(Mat(approx)))
                {
                    double maxCosine = 0;
                    for (int j = 2; j < 5; j++) {
                        // 求轮廓边缘之间角度的最大余弦
                        double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }
                    if (maxCosine < 0.3
                        && (
                        approx[0].x > (imageWidth * 0.4) && approx[0].x < (imageWidth * 0.7) &&
                        approx[1].x >(imageWidth * 0.4) && approx[1].x < (imageWidth * 0.7) &&
                        approx[2].x >(imageWidth * 0.4) && approx[2].x < (imageWidth * 0.7) &&
                        approx[3].x >(imageWidth * 0.4) && approx[3].x < (imageWidth * 0.7) &&
                        approx[0].y >(imageHeight * 0.25) && approx[0].y < (imageHeight * 0.8) &&
                        approx[1].y >(imageHeight * 0.25) && approx[1].y < (imageHeight * 0.8) &&
                        approx[2].y >(imageHeight * 0.25) && approx[2].y < (imageHeight * 0.8) &&
                        approx[3].y >(imageHeight * 0.25) && approx[3].y < (imageHeight * 0.8)
                        )
                        ) {
                        squares.push_back(approx);
                    }
                }
            }
        }
    }
    vector<Point> targetSquare;//目标检测矩阵 依次为 左上左下右下右上的四个点

    int minHeight = 0;
    int maxHeight = 0;
    int minWidth = 0;
    int maxWidth = 0;
    for (int i = squares.size() - 1; i >= 0; i--)
    {
        if (minHeight == 0 && minWidth == 0)
        {
            minWidth = squares[i][0].x;
            minHeight = squares[i][0].y;
        }
        const Point* p = &squares[i][0];
        for (int s = 0; s < 4; s++) {
            if (squares[i][s].x < minWidth) {
                minWidth = squares[i][s].x;
            }
            if (squares[i][s].x > maxWidth) {
                maxWidth = squares[i][s].x;
            }
            if (squares[i][s].y < minHeight) {
                minHeight = squares[i][s].y;
            }
            if (squares[i][s].y > maxHeight) {
                maxHeight = squares[i][s].y;
            }
        }
        //const Point* p = &squares[i][0];

        //int n = (int)squares[i].size();
        ////dont detect the border
        //polylines(out, &p, &n, 1, true, Scalar(0, 0, 255), 3, LINE_AA);

    }
    targetSquare.push_back(Point(minWidth, minHeight));
    targetSquare.push_back(Point(minWidth, maxHeight));
    targetSquare.push_back(Point(maxWidth, maxHeight));
    targetSquare.push_back(Point(maxWidth, minHeight));
    const Point* p = &targetSquare[0];
    int n = (int)targetSquare.size();

    //画图
    polylines(out, &p, &n, 1, true, Scalar(0, 255, 0), 3, LINE_AA);
    //imshow("dst", out);
//    waitKey();
    //没有找到矩形框
    if (maxWidth == 0){
        out = out(Rect(int(imageWidth * 0.3), int(imageHeight * 0.2),int(imageWidth * 0.5), int(imageHeight * 0.6)));
    }
    else {
        out = out(Rect(minWidth, minHeight, maxWidth - minWidth, maxHeight - minHeight));
    }
    
}





// Avoiding name mangling
extern "C" {
    // Attributes to prevent 'unused' function from being removed and to make it visible
    __attribute__((visibility("default"))) __attribute__((used))
    const char* version() {
        return CV_VERSION;
    }

    __attribute__((visibility("default"))) __attribute__((used))
    //处理普通试纸图像主函数
    int process_image(char* imageUrl) {
        try{
            IplImage* imageOriginal = cvLoadImage(imageUrl);
            if (imageOriginal == NULL) {    //open image error
                printf("打开图像文件失败!");
                return -1;
            }

            /* 缩放图片 将图片大小统一处理图片宽度为200pix大小 */
            int orginalImageHeight = imageOriginal->height;
            int orginalImageWidth = imageOriginal->width;
            if (orginalImageWidth < zoomWidthPix) {
                int returnValue = -2;
                printf("-3 图片太小不清楚，请重拍");
                return returnValue;
            }
            double scale = double(double(zoomWidthPix) / orginalImageWidth);//缩放的倍数
            CvSize imageZoomSize;
            imageZoomSize.width = imageOriginal->width * scale;
            imageZoomSize.height = imageOriginal->height * scale;
            IplImage* imageZoom = cvCreateImage(imageZoomSize, imageOriginal->depth, imageOriginal->nChannels);  //缩放图片初始化
            cvResize(imageOriginal, imageZoom, CV_INTER_CUBIC);

            /* 获取图片的HSV矩阵 */
            IplImage* imageHsv = cvCreateImage(cvGetSize(imageZoom), IPL_DEPTH_8U, 3);
            cvCvtColor(imageZoom, imageHsv, CV_BGR2HSV);// 将RGB图像转为HSV图像的函数

            // 原图像
            //cvShowImage("原图像", imageZoom);
            Mat imageZoomMat = cvarrToMat(imageZoom);
            Mat DstPic, edge, grayImage, grad_Image, xdst, ydst, sobelImage, blurImage, cannyImage, marrImage, otsuImage, enhancePic, threshPic, imageROI;

            // 颜色均衡化
            //Mat ycrcb;
            //cvtColor(imageZoomMat, ycrcb, COLOR_BGR2YCrCb);
            //vector<Mat> channels;
            //split(ycrcb, channels);
            //equalizeHist(channels[0], channels[0]);
            //merge(channels, ycrcb);
            //cvtColor(ycrcb, imageZoomMat, COLOR_YCrCb2BGR);
            //imshow("颜色均衡化", imageZoomMat);

            //创建与src同类型和同大小的矩阵
            DstPic.create(imageZoomMat.size(), imageZoomMat.type());

            //将原始图转化为灰度图
            cvtColor(imageZoomMat, grayImage, COLOR_BGR2GRAY);
            blur(imageZoomMat, blurImage, cv::Size(3, 3));
            //运行Canny算子
            //Canny(blurImage, cannyImage, 250, 250);
            //imshow("cannyImage边缘提取效果", cannyImage);

            //运行Sobel算子
            Sobel(grayImage, xdst, CV_64F, 1, 0);
            Sobel(grayImage, ydst, CV_64F, 0, 1);
            subtract(xdst, ydst, sobelImage);
            convertScaleAbs(ydst, sobelImage);
            //imshow("Sobel边缘提取效果", sobelImage);
        
            // 填充空白区域，增强对比度
            Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
            morphologyEx(sobelImage, enhancePic, cv::MORPH_CLOSE, kernel);          // 去除噪声
            //imshow("enhancePic", enhancePic);
            blur(sobelImage, threshPic, cv::Size(10, 10));
            threshold(enhancePic, otsuImage, 0, 255, THRESH_OTSU); //OTSU自适应
            //imshow("OTSU", otsuImage);
            threshold(threshPic, threshPic, 140, 255, cv::THRESH_BINARY);//90  二值化
            //imshow("二值化处理", threshPic);

            //// 核函数
            //Mat element1 = getStructuringElement(MORPH_RECT, cv::Size(2, 2));
            //Mat element2 = getStructuringElement(MORPH_RECT, cv::Size(2, 2));
            //// 膨胀
            //dilate(threshPic, threshPic, element2);
            //imshow("第一次膨胀", threshPic);
            //// 腐蚀
            //erode(threshPic, threshPic, element1);
            //imshow("第一次腐蚀", threshPic);
            //// 膨胀
            //dilate(threshPic, threshPic, element2);
            //imshow("第二次膨胀", threshPic);

            //imshow("threshPic", threshPic);
            //查找轮廓
            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            //findContours(threshPic, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
            findContours(otsuImage, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
            Mat imageContours = Mat::zeros(imageZoomMat.size(), CV_8UC1);  //轮廓
            Mat marks(imageZoomMat.size(), CV_32S);
            marks = Scalar::all(0);
            int index = 0;
            int compCount = 0;
            for (; index >= 0; index = hierarchy[index][0], compCount++)
            {
                drawContours(imageContours, contours, index, Scalar(255), 1, 8, hierarchy);
            }
            //imshow("轮廓", imageContours);

            //霍夫变换进行直线检测
            vector<Vec4i> lines;
            /*
                rho 与像素相关单位的距离精度
                theta 弧度测量的角度精度
                threshold 阈值参数。如果相应的累计值大于 threshold， 则函数返回这条线段.
                minLineLength 最小直线长度，即如果小于该值，则不被认为是一条直线
                maxLineGap 最大直线间隙，如果有两条线段是在一条直线上，但它们之间因为有间隙，所以被认为是两个线段，如果这个间隙大于该值，则被认为是两条线段，否则是一条。
            */
            HoughLinesP(imageContours, lines, 1, CV_PI / 180, 100, 200, 80); //sobel
            //HoughLinesP(imageContours, lines, 10, CV_PI / 180, 100, 200, 20); //sobel test


            //// 画图
            //Mat imageLinePic = imageZoomMat;//画图的直线
            //for (size_t i = 0; i < lines.size(); i++) {
            //    Vec4i l = lines[i];
            //    line(imageLinePic, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 2, LINE_AA);
            //}
            //// Show result image
            //imshow("直线检测", imageLinePic);



            /*
                *******************************获取感兴趣区域*******************************
            */
            int imageHeight = imageZoomSize.height;
            int imageWidth = imageZoomSize.width;
            if (lines.size() >= 2){
                /*筛选出间隔最大的两条线*/
                Vec4i topLine = lines[0];
                Vec4i bottomLine = lines[0];
                for (int i = lines.size() - 1; i >= 0; i--) {
                    //获取直线斜率
                    float angleLine = getAngle(lines[i]);
                    //过滤掉角度大于20度的直线线段
                    if (fabs(angleLine) > 20.0) {
                        lines.erase(remove(lines.begin(), lines.end(), lines[i]), lines.end());
                        continue;
                    }
                    //过滤掉图片范围 如果直线不处在中间的60%范围，也过滤掉
                    if ((lines[i][1] < (float(imageHeight) * 0.2) and lines[i][3] < (float(imageHeight) * 0.2)) ||
                        (lines[i][1] > (float(imageHeight) * 0.8) and lines[i][3] > (float(imageHeight) * 0.8))) {
                        lines.erase(remove(lines.begin(), lines.end(), lines[i]), lines.end());
                        continue;
                    }
                    //选择最上面的线
                    if (((lines[i][1] < topLine[1]) && (lines[i][3] < topLine[3])) //左边的点更高 右边的点更高 代表整根线在上面
                        || ((lines[i][1] + lines[i][3]) < (topLine[1] + topLine[3]))) {//整体更高
                        topLine = lines[i];
                    }
                    //选择最下面的线
                    if (((lines[i][1] > bottomLine[1]) && (lines[i][3] > bottomLine[3])) //左边的点更高 右边的点更高 代表整根线在上面
                        || ((lines[i][1] + lines[i][3]) > (bottomLine[1] + bottomLine[3]))) {
                        bottomLine = lines[i];
                    }
                }

                //如果筛选过后 还有两条以上直线 则取最上面跟最下面的直线
                if (lines.size() >= 2){
                    float topPoint = 0;
                    if (topLine[1] < topLine[3]){
                        topPoint = topLine[1];
                    }
                    else{
                        topPoint = topLine[3];
                    }

                    float bottomPoint = 0;
                    if (bottomLine[1] > bottomLine[3]){
                        bottomPoint = bottomLine[1];
                    }
                    else{
                        bottomPoint = bottomLine[3];
                    }

                    //如果两跳直线太过接近，则默认选原先的
                    if ((bottomPoint - topPoint) < 10){
                        imageROI = imageZoomMat;
                    }
                    else{
                        imageROI = imageZoomMat(Rect(0, topPoint, imageWidth, bottomPoint - topPoint));
                    }
                }
                else {
                    imageROI = imageZoomMat;
                }
            }
            else {
                imageROI = imageZoomMat;
            }

            //imshow("imageROI", imageROI);
            //waitKey(0);
            /*
                *******************************处理裁剪后区域的逻辑*******************************
            */
            CvScalar scalar;
            //获取二维的HSV矩阵
            imageHeight = imageROI.rows;
            imageWidth = imageROI.cols;
            struct HSVPixel*** HSVArr = (struct HSVPixel***)malloc(imageHeight * sizeof(struct HSVPixel**));
            float b;
            float g;
            float r;
            for (int i = 0; i < imageHeight; i++) {    //HSV
                struct HSVPixel** HSVArr1 = (struct HSVPixel**)malloc(imageWidth * sizeof(struct HSVPixel*));
                HSVArr[i] = HSVArr1;
                for (int j = 0; j < imageWidth; j++) {
                    struct HSVPixel* HSVArr2 = (struct HSVPixel*)malloc(sizeof(struct HSVPixel));
                    b = imageROI.at<Vec3b>(i, j)[0];
                    g = imageROI.at<Vec3b>(i, j)[1];
                    r = imageROI.at<Vec3b>(i, j)[2];

                    //scalar = cvGet2D(imageZoom, i, j);    //获取像素点的RGB颜色分量
                    //printf("第%d个像素的H：%d°，S：%lf，V：%lf\n", (i * imageZoom->width) + j, (int)scalar.val[0] * 2, scalar.val[1] / 255, scalar.val[2] / 255);
                    vector<float> HSVPix = RGB2HSV(r, g, b);
                    HSVArr2->H = HSVPix[0];
                    HSVArr2->S = HSVPix[1];
                    HSVArr2->V = HSVPix[2];
                    HSVArr[i][j] = HSVArr2;
                    //getchar();    //防止打印速度太快，暂停一次打印一次
                }
                //printf("i:%d", i);
            }

            //产生最初始的01矩阵
            int*** oneHotMatrix = (int***)malloc(imageHeight * sizeof(int**));
            vector<vector<int>> heightValidList;  // 有效数组的集合 比如 [[1,2,3][7,8][11,12,14][17,18]]
            vector<vector<int>> widthValidList;
            vector<int>heightValidLines;  // 有效数组的列表
            vector<int>widthValidLines;
            for (int i = 0; i < imageHeight; i++) {
                int** oneHotLine = (int**)malloc(imageWidth * sizeof(int*));
                oneHotMatrix[i] = oneHotLine;
                for (int j = 0; j < imageWidth; j++) {
                    int* oneHot = (int*)malloc(sizeof(int));
                    // S-> 0.1
                    if (((HSVArr[i][j]->H >= 0 and HSVArr[i][j]->H <= 60) or (HSVArr[i][j]->H >= 300 and HSVArr[i][j]->H <= 360)) and HSVArr[i][j]->S >= 0.09 and HSVArr[i][j]->V >= 0.3) {
                        *oneHot = 1;
                    }
                    else {
                        *oneHot = 0;
                    }
                    oneHotMatrix[i][j] = oneHot;
                    //printf("第%d个像素的H：%d°，S：%lf，V：%lf，对应的O1值为：%d\n", (i* imageWidth) + j, HSVArr[i][j]->H, HSVArr[i][j]->S, HSVArr[i][j]->V, *oneHot);
                }
            }
            //drawImg(imageHeight, imageWidth, oneHotMatrix);
            //printf("总共有%d行，%d列", imageHeight, imageWidth);

            // 过滤图片中一些不符合条件的行
            for (int i = 0; i < imageHeight; i++) {
                int lineZeroCount = 0;
                int lineOneCount = 0;
                float proportionOne = 0;
                for (int j = 0; j < imageWidth; j++) {
                    if (*(oneHotMatrix[i][j]) == 0) {
                        lineZeroCount = lineZeroCount + 1;
                    }
                    else {
                        lineOneCount = lineOneCount + 1;
                    }
                }
                proportionOne = (float)lineOneCount / ((float)lineOneCount + (float)lineZeroCount);
                if (proportionOne >= 0.05) {
                    heightValidLines.push_back(i);
                }
            }
            heightValidList = filterVector(heightValidLines);
            // 过滤列

            for (int j = 0; j < imageWidth; j++) { // 列单位
                int colZeroCount = 0;
                int colOneCount = 0;
                float proportionOne = 0;
                for (int k = 0; k < heightValidList.size(); k++) {
                    for (int i = 0; i < heightValidList[k].size(); i++) { // 行单位
                        if (*(oneHotMatrix[(heightValidList[k][i])][j]) == 0) {
                            colZeroCount = colZeroCount + 1;
                        }
                        else {
                            colOneCount = colOneCount + 1;
                        }
                    }
                }
                proportionOne = (float)colOneCount / ((float)colOneCount + (float)colZeroCount);
                if (proportionOne >= 0.2) { // 如果列中1的比例大于0.2 则进入候选的有效段
                    widthValidLines.push_back(j);
                }
            }
            //将有效的列处理成连续集合的形式  [1,2,3,5,8,9,10] => [[1,2,3][5][8,9,10]]
            widthValidList = filterVector(widthValidLines);

            //----------将行跟列串成连续的数据块，并过滤出有效连续数据块-----------  [[[2,3][2,4][2,5][3,3]...]]
            //获取最右边的大块的整体高度
            int maxBlockHeight = 0;
            int maxBlockNum = 0;
            int maxBlockLeftIndex = 0; //最大块的左边坐标
            int maxBlockTopIndex = 0; //最大块的顶部坐标
            int maxBlockBottomIndex = 0; //最大块的顶部坐标
            //
            //获取最初始的vec块
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

                    //在未过滤时做最大块判断
                    if (maxBlockNum < vec1.size()) {
                        maxBlockNum = vec1.size();
                        maxBlockHeight = vec1[vec1.size() - 1][0] - vec1[0][0];
                        maxBlockLeftIndex = vec1[0][1];
                        maxBlockTopIndex = vec1[0][0];
                        maxBlockBottomIndex = vec1[vec1.size() - 1][0];
                    }

                    if (checkVecValidByNormal(vec1, oneHotMatrix)) {
                        vec.push_back(vec1);
                    }
                }
            }


            if (vec.size() < 1) {
                int returnValue = -2;
                printf("-2 未捕捉到试纸 请更换干净的背景重拍。");
                return returnValue;
            }

            // 将连续数据块中整行的0过滤掉
            for (int i = vec.size() - 1; i >= 0; i--) {
                if (vec[i].size() == 0)
                {
                    vec.erase(remove(vec.begin(), vec.end(), vec[i]), vec.end()); //删除vec中值为vec[i]的元素
                    continue;
                }
                int heightIndexBegin = vec[i][0][0];
                int heightIndexEnd = vec[i][(vec[i].size() - 1)][0];
                int widthIndexBegin = vec[i][0][1];
                int widthIndexEnd = vec[i][(vec[i].size() - 1)][1];
                int eachWidth = widthIndexEnd - widthIndexBegin + 1; //每行有多少个元素
                for (int j = heightIndexEnd; j >= heightIndexBegin; j--) { //行
                    bool flagWidth = true;//如果全部为0 则在vec中删除对应行
                    for (int k = widthIndexEnd; k >= widthIndexBegin; k--) { //列
                        if (*(oneHotMatrix[j][k]) == 1) {
                            flagWidth = false;
                            break;
                        }
                    }
                    if (flagWidth == true) {
                        //执行删除vec中对应行的操作
                        for (int k = widthIndexEnd; k >= widthIndexBegin; k--) { //列
                            int deleteIndex = ((j - heightIndexBegin) * eachWidth) + (k - widthIndexBegin);//获取要删除的元素对应的vec[i]中对应第几个
                            vec[i].erase(remove(vec[i].begin(), vec[i].end(), vec[i][deleteIndex]), vec[i].end()); //删除vec中值为vec[i]的元素
                        }
                    }
                }
            }
            // 将连续数据块中整列的0过滤掉
            for (int i = vec.size() - 1; i >= 0; i--) {
                int heightIndexBegin = vec[i][0][0];
                int heightIndexEnd = vec[i][(vec[i].size() - 1)][0];
                int widthIndexBegin = vec[i][0][1];
                int widthIndexEnd = vec[i][(vec[i].size() - 1)][1];
                int eachWidth = widthIndexEnd - widthIndexBegin + 1; //每列有多少个元素
                for (int k = widthIndexEnd; k >= widthIndexBegin; k--) { //列
                    bool flagHeight = true;//如果全部为0 则在vec中删除对应列
                    for (int j = heightIndexEnd; j >= heightIndexBegin; j--) { //行
                        if (*(oneHotMatrix[j][k]) == 1) {
                            flagHeight = false;
                            break;
                        }
                    }
                    if (flagHeight == true) {
                        //执行删除vec中对应行的操作
                        for (int j = heightIndexEnd; j >= heightIndexBegin; j--) {
                            int deleteIndex = ((j - heightIndexBegin) * eachWidth) + (k - widthIndexBegin);//获取要删除的元素对应的vec[i]中对应第几个
                            vec[i].erase(remove(vec[i].begin(), vec[i].end(), vec[i][deleteIndex]), vec[i].end()); //删除vec中值为vec[i]的元素
                        }
                        eachWidth = eachWidth - 1;
                    }
                }
            }


            //过滤杂块写的有点问题 需要改成判断1区域的  连续像素块中 标记为1的数量小于总像素块数量的2％，则将其排除；-------------------

            //for (int i = vec.size()-1; i >= 0; i--) { //遍历每个连续块
            //    //if (float(vec[i].size()) < ((float)imageHeight * (float)imageWidth * (float)(0.02))) { //每个连续块中包含几个点
            //    int count_one = 0; //计算每个连续块中偶遇
            //    for (int j = 0; j < vec[i].size(); j++) {
            //        int x = vec[i][j][0];
            //        int y = vec[i][j][1];
            //        if (*(oneHotMatrix[x][y]) == 1) {
            //            count_one = count_one + 1;
            //        }
            //    }
            //    if(float(count_one) < ((float)imageHeight * (float)imageWidth * (float)(0.01))){
            //        for (int p = 0; p < vec[i].size(); p++) {
            //            *(oneHotMatrix[vec[i][p][0]][vec[i][p][1]]) = 0;
            //        }
            //        vec.erase(remove(vec.begin(), vec.end(), vec[i]), vec.end()); //删除vec中值为vec[i]的元素
            //    }
            //}
            //drawImg(imageHeight, imageWidth, oneHotMatrix);

            
            
            //根据地方去除一些杂块 只取中间位置
            for (int i = vec.size() - 1; i >= 0; i--) {
                for (int j = 0; j < vec[i].size(); j++) {
                    //如果某一个连续块处于图片中间50%范围以外，则排除此连续块
                    if ((vec[i][j][1] < imageWidth * 0.25) || (vec[i][j][1] > imageWidth * 0.75)) {
                        reduceNeverlessBlock(vec[i], oneHotMatrix);
                        vec.erase(remove(vec.begin(), vec.end(), vec[i]), vec.end()); //删除vec中值为vec[i]的元素
                        break;
                    }
                }
            }
            

            if (vec.size() < 1) {
                int returnValue = 0;
                printf("0 未检测到有效试纸");
                return returnValue;
            }

            //两两对比 找到一对相似度高的区域块 同时排除最大的那块 标记为1的数量小于总像素块数量的2％
            vector<vector<int>> vecMatchPairs;  // [[1,2][5,6]]
            for (int i = 0; i < vec.size() - 1; i++) {
                int countOne_i = 0;//获取该数据块总共有多少个1
                int countOneHeight_i = 0;//    获取该数据块总共含有1的行数有几个
                int countOneWidth_i = 0;//    获取该数据块总共含有1的列数有几个
                //    计算数据块总共有多少个1
                vector<int> usedHeightNums_i;
                vector<int> usedWidthNums_i;
                for (int o = 0; o < vec[i].size(); o++) {
                    if (*(oneHotMatrix[vec[i][o][0]][vec[i][o][1]]) == 1) {
                        countOne_i = countOne_i + 1;
                        if (!isContainVec(usedHeightNums_i, vec[i][o][0])) {
                            usedHeightNums_i.push_back(vec[i][o][0]);
                            countOneHeight_i = countOneHeight_i + 1;
                        }
                        if (!isContainVec(usedWidthNums_i, vec[i][o][1])) {
                            usedWidthNums_i.push_back(vec[i][o][1]);
                            countOneWidth_i = countOneWidth_i + 1;
                        }
                    }
                }

                for (int j = i + 1; j < vec.size(); j++) {
                    int countOne_j = 0;//获取该数据块总共有多少个1
                    int countOneHeight_j = 0;//获取该数据块总共含有1的行数有几个
                    int countOneWidth_j = 0;//获取该数据块总共含有1的列数有几个
                    bool flagTotalCount = false; //判断总数是否符合
                    bool flagTotalRow = false; //判断行数是否符合
                    bool flagTotalCol = false; //判断列数是否符合
                    bool flagRowMatch = false; //判断行数是否与最大块拟合到一起
                    //计算数块总共有多少个1
                    vector<int> usedHeightNums_j;
                    vector<int> usedWidthNums_j;
                    for (int o = 0; o < vec[j].size(); o++) {
                        if (*(oneHotMatrix[vec[j][o][0]][vec[j][o][1]]) == 1) {
                            countOne_j = countOne_j + 1;
                            if (!isContainVec(usedHeightNums_j, vec[j][o][0])) {
                                usedHeightNums_j.push_back(vec[j][o][0]);
                                countOneHeight_j = countOneHeight_j + 1;
                            }
                            if (!isContainVec(usedWidthNums_j, vec[j][o][1])) {
                                usedWidthNums_j.push_back(vec[j][o][1]);
                                countOneWidth_j = countOneWidth_j + 1;
                            }
                        }
                    }

                    //判断总数是否符合
                    if (float(float(countOne_i) / float(countOne_j)) >= 0.4 and (float(countOne_i) / float(countOne_j)) <= 2.5) {
                        flagTotalCount = true;
                    }
                    //判断行数是否符合
                    if (float(float(countOneHeight_i) / float(countOneHeight_j)) >= 0.6 and (float(countOneHeight_i) / float(countOneHeight_j)) <= 1.666) {
                        flagTotalRow = true;
                    }
                    //判断列数是否符合
                    if (float(float(countOneWidth_i) / float(countOneWidth_j)) >= 0.33 and (float(countOneWidth_i) / float(countOneWidth_j)) <= 2) {
                        flagTotalCol = true;
                    }
                    //判断行数是否在最大块的行数的范围以内（最右边的大块）
                    if ((vec[i][0][0] >= maxBlockTopIndex*0.5 && vec[i][vec[i].size() - 1][0] <= maxBlockBottomIndex*1.2)&&(vec[j][0][0] >= maxBlockTopIndex*0.5 && vec[j][vec[j].size() - 1][0] <= maxBlockBottomIndex*1.2)){
                        flagRowMatch = true;
                    }
                    if ((flagTotalCount && flagTotalRow && flagTotalCol && flagRowMatch) || (flagRowMatch&& flagTotalCol)) {
                        vector<int > vecMatchPair; //记录
                        vecMatchPair.push_back(i);
                        vecMatchPair.push_back(j);
                        vecMatchPairs.push_back(vecMatchPair);
                    }
                }
            }
            int returnValue = -1;

            if (vecMatchPairs.size() >= 1) {
                //如果找到匹配度存在多块相似块 则取平均大小最大的那块，因为实际情况下图像存在很小小块的杂块影响了判断
                if (vecMatchPairs.size()>1)
                {
                    int maxVexPairIndex=0;
                    int maxVexPairNum = 0;
                    for (int i = 0; i < vecMatchPairs.size(); i++)
                    {
                        int pairTotalNum = vec[vecMatchPairs[i][0]].size() + vec[vecMatchPairs[i][1]].size();
                        if (maxVexPairNum< pairTotalNum)
                        {
                            maxVexPairNum = pairTotalNum;
                            maxVexPairIndex = i;
                        }
                    }
                    vector<int> vecMatchPairsTemp = vecMatchPairs[maxVexPairIndex];
                    vecMatchPairs.clear();
                    vecMatchPairs.push_back(vecMatchPairsTemp);
                }

                //1-8等级对应rgb的r的阈值
                float redLevel1 = 254;
                float redLevel2 = 252;
                float redLevel3 = 241;
                float redLevel4 = 235;
                float redLevel5 = 218;
                float redLevel6 = 190;
                float redLevel7 = 159;
                float redLevel8 = 139;

                //结果对应为1-8的颜色转为HSV
                vector<float> HSV_1 = RGB2HSV(redLevel1, 254, 254);
                vector<float> HSV_2 = RGB2HSV(redLevel2, 180, 209);
                vector<float> HSV_3 = RGB2HSV(redLevel3, 128, 177);
                vector<float> HSV_4 = RGB2HSV(redLevel4, 69, 141);
                vector<float> HSV_5 = RGB2HSV(redLevel5, 39, 117);
                vector<float> HSV_6 = RGB2HSV(redLevel6, 38, 105);
                vector<float> HSV_7 = RGB2HSV(redLevel7, 36, 92);
                vector<float> HSV_8 = RGB2HSV(redLevel8, 29, 76);

                //将左边的块定为T 右边的块定为C
                int blockT = 0; //T线在vec中对应的index
                int blockC = 0; //C线在vec中对应的index
                if (vec[vecMatchPairs[0][0]][0][1] > vec[vecMatchPairs[0][1]][0][1]) {
                    blockC = vecMatchPairs[0][0];
                    blockT = vecMatchPairs[0][1];
                }
                else {
                    blockT = vecMatchPairs[0][0];
                    blockC = vecMatchPairs[0][1];
                }

                // vec => [[[2,3][2,4][2,5][3,3]...]]
                vector<vector<int>> vecBlockC = vec[blockC];
                vector<vector<int>> vecBlockT = vec[blockT];
                // 精确到每个点
                float vecBlockCRSum = 0;
                float vecBlockCRAvg = 0;
                float vecBlockTRSum = 0;//rgb中r的总值
                float vecBlockTRAvg = 0;

                //通过rgb中的r的值进行计算
                for (int l = 0; l < vecBlockC.size(); l++) {
                    int indexHeight = vecBlockC[l][0];
                    int indexWidth = vecBlockC[l][1];
                    r = imageROI.at<Vec3b>(indexHeight, indexWidth)[2];
                    vecBlockCRSum = vecBlockCRSum + r;
                }
                vecBlockCRAvg = vecBlockCRSum / float(vecBlockC.size());

                //通过rgb中的r的值进行计算
                for (int l = 0; l < vecBlockT.size(); l++) {
                    int indexHeight = vecBlockT[l][0];
                    int indexWidth = vecBlockT[l][1];
                    r = imageROI.at<Vec3b>(indexHeight, indexWidth)[2];
                    vecBlockTRSum = vecBlockTRSum + r;
                }
                vecBlockTRAvg = vecBlockTRSum / float(vecBlockT.size());


                //根据HSV的色相差值 根据这个值定试纸检测结果
                if (vecBlockTRAvg <= vecBlockCRAvg) {
                    printf("8 强阳");
                    returnValue = 8;
                }
                else if (vecBlockTRAvg <= vecBlockCRAvg*1.15) {
                    printf("7 阳");
                    returnValue = 7;
                }
                else if (vecBlockTRAvg <= vecBlockCRAvg * 1.35) {
                    printf("6 阳");
                    returnValue = 6;
                }
                else if (vecBlockTRAvg <= vecBlockCRAvg * 1.57) {
                    printf("5 弱阳");
                    returnValue = 5;
                }
                else if (vecBlockTRAvg <= vecBlockCRAvg * 1.67) {
                    printf("4 弱阳");
                    returnValue = 4;
                }
                else if (vecBlockTRAvg <= vecBlockCRAvg * 1.71) {
                    printf("3 弱阳");
                    returnValue = 3;
                }
                else if (vecBlockTRAvg <= vecBlockCRAvg * 1.9) {
                    printf("2 弱阳");
                    returnValue = 2;
                }
                else {
                    printf("1 阴");
                    returnValue = 1;
                }
                printf("vecBlockTRAvg：%f", vecBlockTRAvg);
                printf("vecBlockCRAvg：%f", vecBlockCRAvg);
            }
            else if (vecMatchPairs.size() == 0) {
                //如果此时只有一个匹配块，则进入到只有T线或者只有C线的逻辑
                if (vec.size()==1){
                    int vecMarginMaxBlockPix = maxBlockLeftIndex - vec[0][vec[0].size() - 1][1];
                    if (float(vecMarginMaxBlockPix)<float(float(maxBlockHeight) * 1.7)){
                        printf("此时只有C线 ");
                        returnValue = 1;
                    }
                    else {
                        printf("此时只有T线 此时试纸无效");
                        returnValue = 0;
                    }
                }
                else if (vec.size() == 0) {
                    printf(" 没有检测到对应的T线与C线");
                    returnValue = 0;
                }
                else {
                    //检测到多个色素块，但是没有相似块，此时判断失败
                    //printf("-5 检测到多个色素块，但是没有相似块，请重拍");
                    printf("0 没有检测到对应的T线与C线");
                    returnValue = 0;
                }
            }
            else {
                printf("-99 存在未知异常");
            }
            free(HSVArr);
            return returnValue;
        }
        catch (...) {
            printf("-99 存在未知异常");
            return -99;
        }
    }



    __attribute__((visibility("default"))) __attribute__((used))
    int process_semi_quan_image(char* fileUrl) {
        try {
            IplImage* imageOriginal = cvLoadImage(fileUrl);
            Mat image = cvarrToMat(imageOriginal);
            Mat imageROI;//感兴趣区域图像
            findSquares(image, imageROI);
            int returnValue = -1; //返回值
            //imshow("imageROI", imageROI);
            //waitKey();
            CvScalar scalar;
            //获取二维的HSV矩阵
            int imageHeight = imageROI.rows;
            int imageWidth = imageROI.cols;
            struct HSVPixel*** HSVArr = (struct HSVPixel***)malloc(imageHeight * sizeof(struct HSVPixel**));
            float b;
            float g;
            float r;
            for (int i = 0; i < imageHeight; i++) {    //HSV
                struct HSVPixel** HSVArr1 = (struct HSVPixel**)malloc(imageWidth * sizeof(struct HSVPixel*));
                HSVArr[i] = HSVArr1;
                for (int j = 0; j < imageWidth; j++) {
                    struct HSVPixel* HSVArr2 = (struct HSVPixel*)malloc(sizeof(struct HSVPixel));
                    b = imageROI.at<Vec3b>(i, j)[0];
                    g = imageROI.at<Vec3b>(i, j)[1];
                    r = imageROI.at<Vec3b>(i, j)[2];

                    //scalar = cvGet2D(imageZoom, i, j);    //获取像素点的RGB颜色分量
                    //printf("第%d个像素的H：%d°，S：%lf，V：%lf\n", (i * imageZoom->width) + j, (int)scalar.val[0] * 2, scalar.val[1] / 255, scalar.val[2] / 255);
                    vector<float> HSVPix = RGB2HSV(r, g, b);
                    HSVArr2->H = HSVPix[0];
                    HSVArr2->S = HSVPix[1];
                    HSVArr2->V = HSVPix[2];
                    HSVArr[i][j] = HSVArr2;
                    //getchar();    //防止打印速度太快，暂停一次打印一次
                }
                //printf("i:%d", i);
            }

            //产生最初始的01矩阵
            int*** oneHotMatrix = (int***)malloc(imageHeight * sizeof(int**));
            vector<vector<int>> heightValidList;  // 有效数组的集合 比如 [[1,2,3][7,8][11,12,14][17,18]]
            vector<vector<int>> widthValidList;
            vector<int>heightValidLines;  // 有效数组的列表
            vector<int>widthValidLines;
            for (int i = 0; i < imageHeight; i++) {
                int** oneHotLine = (int**)malloc(imageWidth * sizeof(int*));
                oneHotMatrix[i] = oneHotLine;
                for (int j = 0; j < imageWidth; j++) {
                    int* oneHot = (int*)malloc(sizeof(int));
                    if (((HSVArr[i][j]->H >= 0 and HSVArr[i][j]->H <= 60) or (HSVArr[i][j]->H >= 300 and HSVArr[i][j]->H <= 360)) and HSVArr[i][j]->S >= 0.1 and HSVArr[i][j]->V >= 0.3) {
                        *oneHot = 1;
                    }
                    else {
                        *oneHot = 0;
                    }
                    oneHotMatrix[i][j] = oneHot;
                }
            }
            //drawImg(imageHeight, imageWidth, oneHotMatrix);
        
            //printf("总共有%d行，%d列", imageHeight, imageWidth);
            // 过滤图片中一些不符合条件的行
            for (int i = 0; i < imageHeight; i++) {
                int lineZeroCount = 0;
                int lineOneCount = 0;
                float proportionOne = 0;
                for (int j = 0; j < imageWidth; j++) {
                    if (*(oneHotMatrix[i][j]) == 0) {
                        lineZeroCount = lineZeroCount + 1;
                    }
                    else {
                        lineOneCount = lineOneCount + 1;
                    }
                }
                proportionOne = (float)lineOneCount / ((float)lineOneCount + (float)lineZeroCount);
                if (proportionOne >= 0.05) {
                    heightValidLines.push_back(i);
                }
            }
            heightValidList = filterVector(heightValidLines);
            // 过滤列
            for (int j = 0; j < imageWidth; j++) { // 列单位
                int colZeroCount = 0;
                int colOneCount = 0;
                float proportionOne = 0;
                for (int k = 0; k < imageHeight; k++) {// 行单位
                    if (*(oneHotMatrix[k][j]) == 0) {
                        colZeroCount = colZeroCount + 1;
                    }
                    else {
                        colOneCount = colOneCount + 1;
                    }
                }
                proportionOne = (float)colOneCount / ((float)colOneCount + (float)colZeroCount);
                if (proportionOne >= 0.1) { // 如果列中1的比例大于0.2 则进入候选的有效段
                    widthValidLines.push_back(j);
                }
            }
            //将有效的列处理成连续集合的形式  [1,2,3,5,8,9,10] => [[1,2,3][5][8,9,10]]
            widthValidList = filterVector(widthValidLines);

            //----------将行跟列串成连续的数据块，并过滤出有效连续数据块-----------  [[[2,3][2,4][2,5][3,3]...]]
            //获取最初始的vec块
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
                    //if (checkVecValidBySemiQuan(vec1, oneHotMatrix)) {
                    vec.push_back(vec1);
                    //}
                }
            }
            if (vec.size() < 1) {
                int returnValue = -2;
                printf("-2 图片不清晰，请重拍");
                return returnValue;
            }

            // 将连续数据块中整行的0过滤掉
            for (int i = vec.size() - 1; i >= 0; i--) {
                if (vec[i].size() == 0)
                {
                    vec.erase(remove(vec.begin(), vec.end(), vec[i]), vec.end()); //删除vec中值为vec[i]的元素
                    continue;
                }
                int heightIndexBegin = vec[i][0][0];
                int heightIndexEnd = vec[i][(vec[i].size() - 1)][0];
                int widthIndexBegin = vec[i][0][1];
                int widthIndexEnd = vec[i][(vec[i].size() - 1)][1];
                int eachWidth = widthIndexEnd - widthIndexBegin + 1; //每行有多少个元素
                for (int j = heightIndexEnd; j >= heightIndexBegin; j--) { //行
                    bool flagWidth = true;//如果全部为0 则在vec中删除对应行
                    for (int k = widthIndexEnd; k >= widthIndexBegin; k--) { //列
                        if (*(oneHotMatrix[j][k]) == 1) {
                            flagWidth = false;
                            break;
                        }
                    }
                    if (flagWidth == true) {
                        //执行删除vec中对应行的操作
                        for (int k = widthIndexEnd; k >= widthIndexBegin; k--) { //列
                            int deleteIndex = ((j - heightIndexBegin) * eachWidth) + (k - widthIndexBegin);//获取要删除的元素对应的vec[i]中对应第几个
                            vec[i].erase(remove(vec[i].begin(), vec[i].end(), vec[i][deleteIndex]), vec[i].end()); //删除vec中值为vec[i]的元素
                        }
                    }
                }
            }
            // 将连续数据块中整列的0过滤掉
            for (int i = vec.size() - 1; i >= 0; i--) {
                int heightIndexBegin = vec[i][0][0];
                int heightIndexEnd = vec[i][(vec[i].size() - 1)][0];
                int widthIndexBegin = vec[i][0][1];
                int widthIndexEnd = vec[i][(vec[i].size() - 1)][1];
                int eachWidth = widthIndexEnd - widthIndexBegin + 1; //每列有多少个元素
                for (int k = widthIndexEnd; k > widthIndexBegin; k--) { //列
                    //如果全部为0 则在vec中删除对应列
                    int oneColCount = 0;
                    for (int j = heightIndexEnd; j >= heightIndexBegin; j--) { //行
                        if (*(oneHotMatrix[j][k]) == 1) {
                            oneColCount = oneColCount + 1;
                        }
                    }
                    if (float(float(oneColCount) / float(eachWidth)) < 0.2){
                        //执行删除vec中对应行的操作
                        for (int j = heightIndexEnd; j >= heightIndexBegin; j--) {
                            int deleteIndex = ((j - heightIndexBegin) * eachWidth) + (k - widthIndexBegin);//获取要删除的元素对应的vec[i]中对应第几个
                            vec[i].erase(remove(vec[i].begin(), vec[i].end(), vec[i][deleteIndex]), vec[i].end()); //删除vec中值为vec[i]的元素
                        }
                        eachWidth = eachWidth - 1;
                    }
                }
            }
            
            //drawImg(imageHeight, imageWidth, oneHotMatrix);

            if (vec.size() < 1) {
                int returnValue = -2;
                printf("-2 图片不清晰，请重拍");
                return returnValue;
            }

            //两两对比 找到一对相似度高的区域块 同时排除最大的那块 标记为1的数量小于总像素块数量的2％
            vector<vector<int>> vecMatchPairs;  // [[1,2][5,6]]
            for (int i = 0; i < vec.size() - 1; i++) {
                int countOne_i = 0;//获取该数据块总共有多少个1
                int countOneHeight_i = 0;//    获取该数据块总共含有1的行数有几个
                int countOneWidth_i = 0;//    获取该数据块总共含有1的列数有几个
                //    计算数据块总共有多少个1
                vector<int> usedHeightNums_i;
                vector<int> usedWidthNums_i;
                for (int o = 0; o < vec[i].size(); o++) {
                    if (*(oneHotMatrix[vec[i][o][0]][vec[i][o][1]]) == 1) {
                        countOne_i = countOne_i + 1;
                        if (!isContainVec(usedHeightNums_i, vec[i][o][0])) {
                            usedHeightNums_i.push_back(vec[i][o][0]);
                            countOneHeight_i = countOneHeight_i + 1;
                        }
                        if (!isContainVec(usedWidthNums_i, vec[i][o][1])) {
                            usedWidthNums_i.push_back(vec[i][o][1]);
                            countOneWidth_i = countOneWidth_i + 1;
                        }
                    }
                }

                for (int j = i + 1; j < vec.size(); j++) {
                    int countOne_j = 0;//获取该数据块总共有多少个1
                    int countOneHeight_j = 0;//获取该数据块总共含有1的行数有几个
                    int countOneWidth_j = 0;//获取该数据块总共含有1的列数有几个
                    bool flagTotalCount = false; //判断总数是否符合
                    bool flagTotalRow = false; //判断行数是否符合
                    bool flagTotalCol = false; //判断列数是否符合
                    bool flagRowMatch = false; //判断列数是否重合
                    //计算数块总共有多少个1
                    vector<int> usedHeightNums_j;
                    vector<int> usedWidthNums_j;
                    for (int o = 0; o < vec[j].size(); o++) {
                        if (*(oneHotMatrix[vec[j][o][0]][vec[j][o][1]]) == 1) {
                            countOne_j = countOne_j + 1;
                            if (!isContainVec(usedHeightNums_j, vec[j][o][0])) {
                                usedHeightNums_j.push_back(vec[j][o][0]);
                                countOneHeight_j = countOneHeight_j + 1;
                            }
                            if (!isContainVec(usedWidthNums_j, vec[j][o][1])) {
                                usedWidthNums_j.push_back(vec[j][o][1]);
                                countOneWidth_j = countOneWidth_j + 1;
                            }
                        }
                    }

                    //判断总数是否符合
                    if (float(float(countOne_i) / float(countOne_j)) >= 0.4 and (float(countOne_i) / float(countOne_j)) <= 2.5) {
                        flagTotalCount = true;
                    }
                    //判断行数是否符合
                    if (float(float(countOneHeight_i) / float(countOneHeight_j)) >= 0.8 and (float(countOneHeight_i) / float(countOneHeight_j)) <= 1.2) {
                        flagTotalRow = true;
                    }
                    //判断列数是否符合
                    if (float(float(countOneWidth_i) / float(countOneWidth_j)) >= 0.33 and (float(countOneWidth_i) / float(countOneWidth_j)) <= 2) {
                        flagTotalCol = true;
                    }
                    //判断行数是否重合
                    if (
                        (vec[i][0][0] <= vec[j][0][0] && vec[i][vec[i].size() - 1][0] >= vec[j][vec[j].size() - 1][0]) ||
                        (vec[i][0][0] <= vec[j][0][0] && vec[i][vec[i].size() - 1][0] <= vec[j][vec[j].size() - 1][0]) ||
                        (vec[i][0][0] >= vec[j][0][0] && vec[i][vec[i].size() - 1][0] <= vec[j][vec[j].size() - 1][0]) ||
                        (vec[i][0][0] >= vec[j][0][0] && vec[i][vec[i].size() - 1][0] >= vec[j][vec[j].size() - 1][0])
                    ){
                        flagRowMatch = true;
                    }

                    if (flagTotalCount && flagTotalRow && flagTotalCol&& flagRowMatch) {
                        vector<int > vecMatchPair; //记录
                        vecMatchPair.push_back(i);
                        vecMatchPair.push_back(j);
                        vecMatchPairs.push_back(vecMatchPair);
                    }
                }
            }
        
            //drawVec(imageHeight, imageWidth, oneHotMatrix,vec);
            if (vecMatchPairs.size() == 1) {

                //1-6等级对应rgb的r的阈值
                float redLevel1 = 254;
                float redLevel2 = 237;
                float redLevel3 = 226;
                float redLevel4 = 189;
                float redLevel5 = 149;
                float redLevel6 = 146;

                //将左边的块定为T 右边的块定为C
                int blockT = 0; //T线在vec中对应的index
                int blockC = 0; //C线在vec中对应的index
                if (vec[vecMatchPairs[0][0]][0][1] > vec[vecMatchPairs[0][1]][0][1]) {
                    blockC = vecMatchPairs[0][0];
                    blockT = vecMatchPairs[0][1];
                }
                else {
                    blockT = vecMatchPairs[0][0];
                    blockC = vecMatchPairs[0][1];
                }

                // vec => [[[2,3][2,4][2,5][3,3]...]]
                vector<vector<int>> vecBlockC = vec[blockC];
                vector<vector<int>> vecBlockT = vec[blockT];
                // 精确到每个点
                float vecBlockCHSum = 0;
                float vecBlockCHAvg = 0;
                float vecBlockTHSum = 0;
                float vecBlockTHAvg = 0;
                float vecBlockTRSum = 0;//rgb中r的总值
                float vecBlockTRAvg = 0;


                //通过rgb中的r的值进行计算
                for (int l = 0; l < vecBlockT.size(); l++) {
                    int indexHeight = vecBlockT[l][0];
                    int indexWidth = vecBlockT[l][1];
                    r = imageROI.at<Vec3b>(indexHeight, indexWidth)[2];
                    vecBlockTRSum = vecBlockTRSum + r;
                }
                vecBlockTRAvg = vecBlockTRSum / float(vecBlockT.size());

                //根据HSV的色相差值 根据这个值定试纸检测结果
                if (vecBlockTRAvg <= redLevel6) {
                    printf("6");
                    returnValue = 6;
                }
                else if (vecBlockTRAvg <= redLevel5) {
                    printf("5");
                    returnValue = 5;
                }
                else if (vecBlockTRAvg <= redLevel4) {
                    printf("4");
                    returnValue = 4;
                }
                else if (vecBlockTRAvg <= redLevel3) {
                    printf("3");
                    returnValue = 3;
                }
                else if (vecBlockTRAvg <= redLevel2) {
                    printf("2");
                    returnValue = 2;
                }
                else if (vecBlockTRAvg <= redLevel1) {
                    printf("1");
                    returnValue = 1;
                }
                else {
                    printf("1 ");
                    returnValue = 1;
                }
                printf("vecBlockTRAvg：%f", vecBlockTRAvg);
            }
            else if (vecMatchPairs.size() == 0) {
                //如果此时只有一个匹配块，则进入到只有T线或者只有C线的逻辑
                if (vec.size() == 1) {
                    printf("此时只有C线 ");
                    returnValue = 1;
                }
                else if (vec.size() == 0) {
                    printf("-4 没有检测到对应的T线与C线，请重拍");
                    returnValue = -4;
                }
                else {
                    //检测到多个色素块，但是没有相似块，此时判断失败
                    printf("-5 检测到多个色素块，但是没有相似块，请重拍");
                    returnValue = -5;
                }
            }
            else {
                printf("-6 存在多个相似块 无法进行检测，请到干净的背景中重拍");
                returnValue = -6;
            }
            free(HSVArr);
            return returnValue;
        }
        catch (...) {
            printf("-99 存在未知异常");
            return -99;
        }
    }

}

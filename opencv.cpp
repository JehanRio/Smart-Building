#include"opencv.h"
#include<opencv2/dnn.hpp>
using namespace std;
using namespace cv;
void QuickDemo::colorSpace_Demo(cv::Mat& image)
{
	cv::Mat gray, hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	cvtColor(image, gray, COLOR_BGR2GRAY);
	namedWindow("HSV",WINDOW_FREERATIO);	// h：色度 s:饱和度 v：明度
	imshow("HSV", hsv); 
	imshow("gray", gray);
	//imwrite("E:/hsv.png", hsv);
}

void QuickDemo::mat_creation_demo(cv::Mat& image)
{
	cv::Mat m1, m2;
	m1 = image.clone();
	image.copyTo(m2);

	// 创建空白图像
	cv::Mat m3 = cv::Mat::ones(Size(255, 255), CV_8UC3);	//8:8行；1:单通道
	m3=Scalar(255,0,0);	// 赋值
	/*std::cout << "width:" << m3.cols << "  height:" << m3.rows << "  channel:" << m3.channels() << std::endl;
	std::cout << m3 <<std::endl;*/
	imshow("1", m3);

	cv::Mat m4 = m3;	// 浅拷贝
	cv::Mat m5;
	m5.copyTo(m4);	// 深拷贝

}

void QuickDemo::pixel_visit_demo(cv::Mat& image)
{
	int w = image.cols;
	int h = image.rows;
	int dims = image.channels();
	for (int row = 0; row < h; row++)
	{
		uchar* current_row = image.ptr<uchar>(row);	// 指针的方法更快
		for (int col = 0; col < w; col++)
		{
			if (dims == 1)	// 灰度图像
			{
				int pv = image.at<uchar>(row, col);	// 转换成int型
				image.at<uchar>(row, col) = 255 - pv;	// 取反

				pv = *current_row;
				*current_row++ = 255 - pv;	// 再次取反
			}
			else if (dims == 3)
			{
				Vec3b bgr = image.at<Vec3b>(row, col);
				image.at<Vec3b>(row, col)[0] = 255 - bgr[0];
				image.at<Vec3b>(row, col)[1] = 255 - bgr[1];
				image.at<Vec3b>(row, col)[2] = 255 - bgr[2];

				*current_row++ = 255 - *current_row;
				*current_row++ = 255 - *current_row;
				*current_row++ = 255 - *current_row;
			}
		}
	}
	imshow("像素读写显示", image);
}

void QuickDemo::operators_demo(cv::Mat& image)
{
	cv::Mat dst;
	cv::Mat m = cv::Mat::zeros(image.size(), image.type());
	m = Scalar(2, 2, 2);
	multiply(image, m, dst);		// 乘法
	//dst = image + Scalar(2, 2, 2);	// 加减
	add(dst, m, dst);
	imshow("dst", dst);
}

static void on_track(int b, void* userdata)	// 用于下方参数：回调函数
{
	cv::Mat image = *(cv::Mat*)userdata;
	cv::Mat dst = cv::Mat::zeros(image.size(), image.type());
	cv::Mat m = cv::Mat::zeros(image.size(), image.type());
	m = Scalar(b, b, b);
	add(image, m, dst);
	imshow("亮度调整", dst);
}

void QuickDemo::tracking_bar_demo(cv::Mat& image)
{
	namedWindow("亮度调整", WINDOW_AUTOSIZE);
	int lightness = 50;
	int max = 100;
	
	createTrackbar("Value Bar", "亮度调整", &lightness, max, on_track,(void*)(&image));
	//on_track(50, &image);
}

void QuickDemo::key_board_demo(cv::Mat& img)
{
	cv::Mat dst=cv::Mat::zeros(img.size(),img.type());
	while (true)
	{
		int c = waitKey(200);	// 等待200ms
		if (c == 27)
			break;
		if (c == 49)
		{
			std::cout << "you enter key #1" << std::endl;
			cvtColor(img, dst, COLOR_BGR2GRAY);
		}
		if (c == 50)
		{
			std::cout << "you enter key #2" << std::endl;
			cvtColor(img, dst, COLOR_BGR2HSV);
		}
		imshow("键盘响应", dst);
	}
	
}

void QuickDemo::bitwise_demo(cv::Mat& img)
{
	cv::Mat m1 = cv::Mat::zeros(Size(256, 256), CV_8UC3);
	cv::Mat m2 = cv::Mat::zeros(Size(256, 256), CV_8UC3);
	rectangle(m1, Rect(100, 100, 80, 80), Scalar(255, 255, 0), -1, LINE_8, 0);	// -1:填充 ；2：描边
	rectangle(m2, Rect(150, 150, 80, 80), Scalar(0, 255, 255), -1, LINE_8, 0);
	imshow("m1", m1);
	imshow("m2", m2);
	cv::Mat dst;
	//bitwise_or(m1, m2, dst);	// 保存在dst
	bitwise_not(m1,dst);	// 取反
	imshow("位操作", dst);
}

void QuickDemo::channels_demo(cv::Mat& img)
{
	std::vector<cv::Mat> mv;
	split(img, mv);
	imshow("blue", mv[0]);
	imshow("green", mv[1]);
	imshow("red", mv[2]);
	
	cv::Mat dst;
	mv[1] = 0;	// 全部为0
	mv[2] = 0;
	merge(mv, dst);	// 将mv合并到dst中
	imshow("blue", dst);

	int from_to[] = { 0,2,1,1,2,0 };
	mixChannels(&img,1, &dst,1, from_to,3);	//1: 矩阵数量 3:fromTo中索引对的数目
	imshow("通道混合", dst);
}

void QuickDemo::inrange_demo(cv::Mat& img)
{
	cv::Mat hsv;
	cvtColor(img, hsv, COLOR_BGR2HSV);	//cv	转换成hsv提取
	cv::Mat mask;
	inRange(hsv, Scalar(35, 43, 46), Scalar(77, 255, 255), mask);	// 传递给mask 区间内255白色，区间外0黑色

	cv::Mat redback = cv::Mat::zeros(img.size(), img.type());
	redback = Scalar(40, 40, 200);	// 做红色背景
	bitwise_not(mask, mask);		// 取反
	imshow("mask", mask);
	img.copyTo(redback, mask);	// mask和img重叠传递给redback 像素值为0的像素点都不会贴到redback上	mask就是蒙版
	imshow("区域提取", redback);
}

void QuickDemo::pixel_statistic_demo(cv::Mat& img)
{
	double minv, maxv;
	Point minLoc, maxLoc;
	std::vector<cv::Mat> mv;
	split(img, mv);
	for (int i = 0; i < 3; i++)
	{
		minMaxLoc(mv[i], &minv, &maxv, &minLoc, &maxLoc);	// 找到图中最大值和最小值	最后一个参数为蒙版
		std::cout <<"NO.channels:"<< i << "   min value:" << minv <<"   max value:" << maxv << std::endl;
	}
	
	cv::Mat mean, stddev;
	meanStdDev(img, mean, stddev);	// 求矩阵的均值和标准偏差
	std::cout << "means:" << mean << std::endl << "stddev:" << stddev << std::endl;
}

void QuickDemo::drawing_demo(cv::Mat& img)
{
	Rect rect;	// 矩形
	rect.x = 200;
	rect.y = 200;
	rect.width = 100;
	rect.height = 100;
	cv::Mat bg = cv::Mat::zeros(img.size(), img.type());
	rectangle(img, rect, Scalar(0, 0, 255), 2, 8, 0);	// 绘制矩形 红色
	circle(bg, Point(200, 150), 40, Scalar(255, 0, 0), -1, LINE_AA, 0);
	line(bg, Point(100, 100), Point(350, 400), Scalar(0, 255, 0), 2, LINE_AA, 0);
	
	cv::Mat dst;
	addWeighted(img, 0.7, bg, 0.3, 0, dst);
	imshow("绘制演示", dst);

}

void QuickDemo::random_drawing()
{
	cv::Mat canvas = cv::Mat::zeros(Size(512, 512), CV_8UC3);
	RNG rng(12345);	// 随机数种子
	while (true)
	{
		int c = waitKey(10);
		if (c == 27)
		{
			break;
		}
		int x1 = rng.uniform(0, canvas.cols);
		int y1 = rng.uniform(0, canvas.rows);
		int x2 = rng.uniform(0, canvas.cols);
		int y2 = rng.uniform(0, canvas.rows);
		canvas = Scalar(0, 0, 0);
		line(canvas, Point(x1,y1), Point(x2,y2), Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255)), 6, LINE_AA, 0); 
		imshow("随机绘制演示", canvas);
	}
}

void QuickDemo::norm_demo(cv::Mat& img)	// 转换成浮点数后必须要归一化
{	
	cv::Mat dst;	
	img.convertTo(img, CV_32F);		// 转换类型 从CV_8UC3到CV_32FC3 32位float channel3
	std::cout << img.type() << std::endl;
	std::cout << dst.type() << std::endl;
	normalize(img, dst, 1.0, 0, NORM_MINMAX);	// 1.0 0 :要归一化的上下界 NORM_MINMAX：最好的归一化类型
	imshow("图像数据归一化", dst);
	// 转换回来：乘以255后转换成CV_8UC3
}

void QuickDemo::resize_demo(cv::Mat& img)
{
	cv::Mat zoomin, zoomout;
	int h = img.rows, w = img.cols;
	resize(img, zoomout, Size(w / 2, h / 2), 0, 0, INTER_LINEAR);	// 线性插值
	imshow("变小", zoomout);
	resize(img, zoomin, Size(w*1.5, h*1.5), 0, 0, INTER_LINEAR);	// 线性插值
	imshow("变大", zoomin);
}

void QuickDemo::flip_demo(cv::Mat& mig)
{
	cv::Mat dst;
	flip(mig, dst, 0);	// 0:上下翻转 1：左右翻转 -1：对角线翻转
	imshow("flip", dst);
}

void QuickDemo::rotate_demo(cv::Mat& img)
{
	cv::Mat dst, M;
	int w = img.cols, h = img.rows;
	M = getRotationMatrix2D(Point2f(w / 2, h / 2), 45, 1.0); // 中心点，角度，放大大小	返回2行3列
	double sin = abs(M.at<double>(0, 1));
	double cos = abs(M.at<double>(0, 0));
	double nw = cos * w + sin * h;
	double nh = sin * w + cos * h;
	M.at<double>(0, 2) += (nw / 2 - w / 2);	// 得到新的中心位置
	M.at<double>(1, 2) += (nh / 2 - h / 2);
	warpAffine(img, dst, M,Size(nw,nh) , INTER_LINEAR,0,Scalar(255,0,0));	// 双线性插值
	imshow("旋转演示", dst);
}

void QuickDemo::video_demo(cv::Mat& img)
{
	VideoCapture capture("E:/视频/like a star/mmexport1629042587270.mp4");
	int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);
	int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
	int count = capture.get(CAP_PROP_FRAME_COUNT);		// 总数
	int fps = capture.get(CAP_PROP_FPS);	// 帧数
	std::cout << "frame width:" << frame_width << std::endl;
	std::cout << "frame height:" << frame_height << std::endl;
	std::cout << "Number of Rrames:" << count << std::endl;
	std::cout << "FPS:" << fps << std::endl;
	VideoWriter write("E:/test.mp4",capture.get(CAP_PROP_FOURCC),fps,Size(frame_width,frame_height),true);
	cv::Mat frame;
	while (1)
	{
		capture.read(frame);
		flip(frame, frame, 1);
		if (frame.empty())
			break;
		imshow("frame", frame);
		write.write(frame);
		// TO DO:do something......
		int c = waitKey(10);	 // 10ms
		if (c == 27)
			break;
	}
	capture.release();
	write.release();
}

void QuickDemo::showHistogram(cv::Mat& img)
{
	// 三通道分离
	std::vector<cv::Mat> bgr_plane;
	split(img, bgr_plane);
	// 定义参数变量
	const int channels[1] = { 0 };
	const int bins[1] = { 256 };
	float hranges[2] = { 0,255 };
	const float* ranges[1] = { hranges };
	cv::Mat b_hist;
	cv::Mat g_hist;
	cv::Mat r_hist;
	// 计算bgr通道的直方图
	calcHist(&bgr_plane[0], 1, 0, cv::Mat(), b_hist, 1, bins, ranges);	//1:一张图 cv::Mat():无蒙版
	calcHist(&bgr_plane[1], 1, 0, cv::Mat(), g_hist, 1, bins, ranges);
	calcHist(&bgr_plane[2], 1, 0, cv::Mat(), r_hist, 1, bins, ranges);
	// 显示直方图
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / bins[0]);
	cv::Mat histImage = cv::Mat::zeros(hist_h, hist_w, CV_8UC3);
	// 归一化直方图数据:为了将直方图的取值范围限制在画布的高度中
	normalize(b_hist, b_hist, histImage.rows, 0, NORM_MINMAX, -1, cv::Mat());
	normalize(g_hist, g_hist, histImage.rows, 0, NORM_MINMAX, -1, cv::Mat());
	normalize(r_hist, r_hist, histImage.rows, 0, NORM_MINMAX, -1, cv::Mat());
	// 绘制直方图曲线
	for (int i = 0; i < bins[0]; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w * i, hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w * i, hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w * i, hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0 ,255), 2, 8, 0);
	}
	// 显示直方图
	namedWindow("Histogram Demo", WINDOW_AUTOSIZE);
	imshow("Histogram Demo", histImage);
}

void QuickDemo::histogram_eq_demo(cv::Mat& img)
{
	cv::Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	imshow("灰度图像", gray);
	cv::Mat dst;
	equalizeHist(gray, dst);
	imshow("直方图均衡化演示", dst);
}

void QuickDemo::blur_demo(cv::Mat& img)
{
	cv::Mat dst;
	blur(img, dst, Size(3, 3), Point(-1, -1));	// -1，-1:默认为卷积核中心位置	最后默认处理方式
	imshow("模糊处理", dst); 
}

void QuickDemo::gaussian_blur_demo(cv::Mat& img)
{
	cv::Mat dst;
	GaussianBlur(img, dst, Size(5, 5), 15);
	imshow("高斯模糊", dst);
}

void QuickDemo::bifilter_demo(cv::Mat& img)	// 抑制噪声,增加边缘信息
{
	cv::Mat dst;
	bilateralFilter(img, dst, 0, 100, 10);
	imshow("高斯双边模糊", dst);
}

void QuickDemo::face_detection_demo()
{
	std::string root_dir = "E:/opencv/sources/samples/dnn/face_detector/";
	dnn::Net net=dnn::readNetFromTensorflow(root_dir+"opencv_face_detector_uint8.pb",root_dir+"opencv_face_detector.pbtxt");	// 模型、配置
	VideoCapture capture("E:/视频/like a star/mmexport1629042587270.mp4");
	cv::Mat frame;
	while (true)
	{
		capture.read(frame);
		flip(frame, frame, 1);
		if (frame.empty())
			break;
		
		cv::Mat blob = dnn::blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 177, 123), false, false); // 图像大小1.0；size模型要求300，300；均值为104，177，123；rgb是否交换：false；是否剪切：false。
		net.setInput(blob);	//NCHW 
		cv::Mat probs = net.forward();	// 7个值
		cv::Mat detectionMat(probs.size[2], probs.size[3], CV_32F, probs.ptr<float>());
		// 解析结果
		for (int i = 0; i < detectionMat.rows; i++)	// 检测出的人头
		{
			float confidence = detectionMat.at<float>(i, 2);
			if (confidence > 0.5)
			{	// 得到矩形的四个点
				int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
				int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
				int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
				int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
				Rect box(x1, y1, x2 - x1, y2 - y1);
				rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);
			}
		}
		imshow("人脸检测演示", frame);
		// TO DO:do something......
		int c = waitKey(10);	 // 10ms
		if (c == 27)
			break;
	}
}

int QuickDemo::face_match()
{
	string det_onnx_path = "E:/opencv/sources/samples/dnn/face_detector/face_detection_yunet_2022mar.onnx";	// 人脸检测模型
	string reg_onnx_path = "E:/opencv/sources/samples/dnn/face_detector/face_recognizer_fast.onnx";	// 人脸识别模型
	string image1_path = "E:/图片/ME/BF7806EA134CFE899D31C93E3D0B0FAD.jpg";
	string image2_path = "E:/图片/ME/FEB7B36BD1C5A98391F4CBD1CF2A6F86.jpg";
	cv::Mat img1 = imread(image1_path);
	cv::Mat img2 = imread(image2_path);
	double cosine_similar_thresh = 0.363;
	double l2norm_similar_thresh = 1.128;

	// Initialize FaceDetector 人脸检测
	Ptr<FaceDetectorYN> faceDetector;
	faceDetector = FaceDetectorYN::create(det_onnx_path, "", img1.size());
	cv::Mat faces_1;
	faceDetector->detect(img1, faces_1);
	if (faces_1.rows < 1)
	{
		std::cerr << "Cannot find a face in " << image1_path << "\n";
		return -1;
	}

	faceDetector = FaceDetectorYN::create(det_onnx_path, "", img2.size());
	cv::Mat faces_2;
	faceDetector->detect(img2, faces_2);
	if (faces_2.rows < 1)
	{
		std::cerr << "Cannot find a face in " << image2_path << "\n";
		return -1;
	}
	

	// Initialize FaceRecognizerSF  人脸对齐
	Ptr<FaceRecognizerSF> faceRecognizer = FaceRecognizerSF::create(reg_onnx_path, "");

	// 在人脸检测部分的基础上, 对齐检测到的首个人脸(faces.row(0)), 保存至aligned_face。
	cv::Mat aligned_face1, aligned_face2;
	faceRecognizer->alignCrop(img1, faces_1.row(0), aligned_face1);
	faceRecognizer->alignCrop(img2, faces_2.row(0), aligned_face2);


	// 展示效果
	cv::Mat show1, show2;
	resize(aligned_face1, show1, Size(300, 200), 0, 0, INTER_AREA);
	resize(aligned_face2, show2, Size(300, 200), 0, 0, INTER_AREA);
	imshow("图1", show1);
	imshow("图2", show2);
	cv::waitKey(0);

	// 特征提取
	cv::Mat feature1, feature2;
	faceRecognizer->feature(aligned_face1, feature1);
	feature1 = feature1.clone();	// 深复制
	faceRecognizer->feature(aligned_face2, feature2);
	feature2 = feature2.clone();

	// 比对人脸特征相似度，判断是否为同一个人
	double cos_score = faceRecognizer->match(feature1, feature2, FaceRecognizerSF::DisType::FR_COSINE);
	double L2_score = faceRecognizer->match(feature1, feature2, FaceRecognizerSF::DisType::FR_NORM_L2);

	
	if (cos_score >= cosine_similar_thresh)  // 当使用consine距离时，值越大，则人脸越相似身份越接近
	{
		std::cout << "They have the same identity;";
	}
	else
	{
		std::cout << "They have different identities;";
	}
	std::cout << " Cosine Similarity: " << cos_score << ", threshold: " << cosine_similar_thresh << ". (higher value means higher similarity, max 1.0)\n";

	if (L2_score <= l2norm_similar_thresh)   // 当使用normL2距离时，值越小，则人脸越相似身份越接近
	{ 
		std::cout << "They have the same identity;";
	}
	else
	{
		std::cout << "They have different identities.";
	}
	std::cout << " NormL2 Distance: " << L2_score << ", threshold: " << l2norm_similar_thresh << ". (lower value means higher similarity, min 0.0)\n";

	return 0;
}

cv::Mat QuickDemo::get_face(cv::Mat& img)
{
	string det_onnx_path = "face_detection_yunet_2022mar.onnx";	// 人脸检测模型
	string reg_onnx_path = "face_recognizer_fast.onnx";	// 人脸识别模型
	Ptr<FaceDetectorYN> faceDetector;
	faceDetector = FaceDetectorYN::create(det_onnx_path, "", img.size());
	cv::Mat faces_1;
	faceDetector->detect(img, faces_1);
	if (faces_1.rows < 1)
	{
		std::cout << "Cannot find a face in the img" << "\n";
		return img;
	}
	// Initialize FaceRecognizerSF  人脸对齐
	Ptr<FaceRecognizerSF> faceRecognizer = FaceRecognizerSF::create(reg_onnx_path, "");

	// 在人脸检测部分的基础上, 对齐检测到的首个人脸(faces.row(0)), 保存至aligned_face。
	cv::Mat aligned_face1;
	faceRecognizer->alignCrop(img, faces_1.row(0), aligned_face1);
	return aligned_face1;
}

cv::Mat QuickDemo::get_feature(cv::Mat& img)
{
	string reg_onnx_path = "face_recognizer_fast.onnx";	// 人脸识别模型
	Ptr<FaceRecognizerSF> faceRecognizer = FaceRecognizerSF::create(reg_onnx_path, "");
	cv::Mat feature1;
	faceRecognizer->feature(img, feature1);
	return feature1;
}


bool QuickDemo::check_person(cv::Mat& feature1, cv::Mat& feature2)
{
	string reg_onnx_path = "face_recognizer_fast.onnx";	// 人脸识别模型
	double cosine_similar_thresh = 0.363;
	double l2norm_similar_thresh = 1.128;

	Ptr<FaceRecognizerSF> faceRecognizer = FaceRecognizerSF::create(reg_onnx_path, "");
	double cos_score = faceRecognizer->match(feature1, feature2, FaceRecognizerSF::DisType::FR_COSINE);
	double L2_score = faceRecognizer->match(feature1, feature2, FaceRecognizerSF::DisType::FR_NORM_L2);

	bool check1 = false, check2 = false;
	if (cos_score >= cosine_similar_thresh)  // 当使用consine距离时，值越大，则人脸越相似身份越接近
	{
		check1 = true;
	}
	if (L2_score <= l2norm_similar_thresh)   // 当使用normL2距离时，值越小，则人脸越相似身份越接近
	{
		check2 = true;
	}
	
	if (check1 && check2)
		return true;
	else
		return false;
}

bool QuickDemo::register_info(string name, string img_path)
{
	ConnectionPool* pool = ConnectionPool::getConnectPool();
	shared_ptr<MysqlConn> conn = pool->getConnection();

	// 编码设置为gbk
	string s = "set names gbk;";
	conn->update(s);

	char sql[1024] = { 0 };
	sprintf_s(sql, "insert into person_img (name,face_path) values('%s','%s');", name.c_str(),img_path.c_str());
	bool flag = conn->update(sql);
	if (flag)
	{
		cout << "注册成功！" << endl;
		return true;
	}
	else
	{
		cout << "注册失败！" << endl;
		return false;
	}
}

void QuickDemo::recognize(cv::VideoCapture &capture)
{
	check_flag = 0;	//重置
	capture.set(cv::CAP_PROP_FPS, 10);
	while (1)
	{
		if (check_flag)
			break;
		cv::Mat frame;
		capture >> frame;// 读取当前帧
		cv::flip(frame, frame, 1);
		cv::imshow("capture", frame);

		if (mutexQ.try_lock())
		{
			thread t1(&QuickDemo::video_thread, this, frame);
			t1.detach();
		}

		if (cv::waitKey(20) == 27)
			break;
	}
	capture.release();
	cout << "欢迎回家！" << endl;
}

void QuickDemo::video_thread(cv::Mat frame)
{
	ConnectionPool* pool = ConnectionPool::getConnectPool();
	shared_ptr<MysqlConn> conn = pool->getConnection();


	// 编码设置为gbk
	string sql = "set names gbk;";
	conn->update(sql);

	frame = QuickDemo::get_face(frame);	// 获取人脸
	cv::Mat feature1 = QuickDemo::get_feature(frame);	// 获取到特征

	sql = "select * from person_img;";
	conn->query(sql);
	string s;
	while (conn->next())
	{
		cout << conn->Value(0) << "," << conn->Value(1) << "," << conn->Value(2) << endl;
		s = conn->Value(2);
		cv::Mat img_sql = cv::imread(s);
		img_sql = QuickDemo::get_face(img_sql);
		cv::Mat feature2 = QuickDemo::get_feature(img_sql);	// 获取到特征
		if (QuickDemo::check_person(feature1, feature2))
		{
			check_flag = 1;	// 标志着结束
			//cout << "线程结束！" << endl;
			cv::imshow("test", frame);
			cv::waitKey(0);
			mutexQ.unlock();

			break;
		}
	}
	mutexQ.unlock();
}





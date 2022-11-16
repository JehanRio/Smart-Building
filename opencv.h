#pragma once
#include<string>
#include<opencv2/opencv.hpp>
#include"ConnectionPool.h"


class QuickDemo
{
public:
	void colorSpace_Demo(cv::Mat& image);
	void mat_creation_demo(cv::Mat& image);	// 图像像素的读写操作
	void pixel_visit_demo(cv::Mat& image);	// 图像像素的算数操作
	void operators_demo(cv::Mat& image);	// 调整图像亮度
	void tracking_bar_demo(cv::Mat& image); // 滚动条操作演示-参数传递值
	void key_board_demo(cv::Mat& img);				// 键盘响应
	void bitwise_demo(cv::Mat& img);		// 图像像素的逻辑操作
	void channels_demo(cv::Mat& img);		// 通道分离与合并
	void inrange_demo(cv::Mat& img);		// 图像色彩空间转换
	void pixel_statistic_demo(cv::Mat& img);// 图像像素值统计
	void drawing_demo(cv::Mat& img);		// 图像几何形状绘制
	void random_drawing();				// 随机数和随机颜色
	void mouse_drawing();				// 鼠标操作与响应 (未写)
	void norm_demo(cv::Mat& img);			// 图像像素类型转换和归一化
	void resize_demo(cv::Mat& img);			// 图像放缩与插值
	void flip_demo(cv::Mat& mig);			// 图像翻转
	void rotate_demo(cv::Mat& img);			// 图像旋转
	void video_demo(cv::Mat& img);			// 视频文件摄像头使用
	void showHistogram(cv::Mat& img);		// 图像直方图
	
	void histogram_eq_demo(cv::Mat& img);	// 直方图均衡化：图像直方图均衡化可用于图像增强（对比度拉伸）
	void blur_demo(cv::Mat& img);			// 卷积 模糊操作
	void gaussian_blur_demo(cv::Mat& img);	// 高斯模糊: 考虑图像空间位置对权重的影响
	void bifilter_demo(cv::Mat& img);		// 高斯双边模糊：考虑了像素分布的影响，对像素值空间分布差异较大的进行保留从而完整的保留了图像的边缘信息

	void face_detection_demo();			// 人脸检测
	int face_match();					// 人脸匹配

	cv::Mat get_face(cv::Mat& img);						// 获取人脸
	cv::Mat get_feature(cv::Mat& img);					// 获取特征
	bool check_person(cv::Mat& img1, cv::Mat& img2);	// 判断两个人的特征是否为同一个人

	bool register_info(std::string name, std::string img_path);		// 注册信息，录入数据库

	void recognize(cv::VideoCapture &capture);		// 人脸识别
private:
	mutex mutexQ;	// 用于下方的线程函数
	bool check_flag;	// 标志着结束
	void video_thread(cv::Mat frame);	// 线程函数
};
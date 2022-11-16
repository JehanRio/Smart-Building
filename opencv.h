#pragma once
#include<string>
#include<opencv2/opencv.hpp>
#include"ConnectionPool.h"


class QuickDemo
{
public:
	void colorSpace_Demo(cv::Mat& image);
	void mat_creation_demo(cv::Mat& image);	// ͼ�����صĶ�д����
	void pixel_visit_demo(cv::Mat& image);	// ͼ�����ص���������
	void operators_demo(cv::Mat& image);	// ����ͼ������
	void tracking_bar_demo(cv::Mat& image); // ������������ʾ-��������ֵ
	void key_board_demo(cv::Mat& img);				// ������Ӧ
	void bitwise_demo(cv::Mat& img);		// ͼ�����ص��߼�����
	void channels_demo(cv::Mat& img);		// ͨ��������ϲ�
	void inrange_demo(cv::Mat& img);		// ͼ��ɫ�ʿռ�ת��
	void pixel_statistic_demo(cv::Mat& img);// ͼ������ֵͳ��
	void drawing_demo(cv::Mat& img);		// ͼ�񼸺���״����
	void random_drawing();				// ������������ɫ
	void mouse_drawing();				// ����������Ӧ (δд)
	void norm_demo(cv::Mat& img);			// ͼ����������ת���͹�һ��
	void resize_demo(cv::Mat& img);			// ͼ��������ֵ
	void flip_demo(cv::Mat& mig);			// ͼ��ת
	void rotate_demo(cv::Mat& img);			// ͼ����ת
	void video_demo(cv::Mat& img);			// ��Ƶ�ļ�����ͷʹ��
	void showHistogram(cv::Mat& img);		// ͼ��ֱ��ͼ
	
	void histogram_eq_demo(cv::Mat& img);	// ֱ��ͼ���⻯��ͼ��ֱ��ͼ���⻯������ͼ����ǿ���Աȶ����죩
	void blur_demo(cv::Mat& img);			// ��� ģ������
	void gaussian_blur_demo(cv::Mat& img);	// ��˹ģ��: ����ͼ��ռ�λ�ö�Ȩ�ص�Ӱ��
	void bifilter_demo(cv::Mat& img);		// ��˹˫��ģ�������������طֲ���Ӱ�죬������ֵ�ռ�ֲ�����ϴ�Ľ��б����Ӷ������ı�����ͼ��ı�Ե��Ϣ

	void face_detection_demo();			// �������
	int face_match();					// ����ƥ��

	cv::Mat get_face(cv::Mat& img);						// ��ȡ����
	cv::Mat get_feature(cv::Mat& img);					// ��ȡ����
	bool check_person(cv::Mat& img1, cv::Mat& img2);	// �ж������˵������Ƿ�Ϊͬһ����

	bool register_info(std::string name, std::string img_path);		// ע����Ϣ��¼�����ݿ�

	void recognize(cv::VideoCapture &capture);		// ����ʶ��
private:
	mutex mutexQ;	// �����·����̺߳���
	bool check_flag;	// ��־�Ž���
	void video_thread(cv::Mat frame);	// �̺߳���
};
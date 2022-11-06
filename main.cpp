#include"opencv.h"
#include"ConnectionPool.h"
#include<thread>
#include<mutex>
using namespace std;

QuickDemo qd;

mutex mutexQ;
bool check_flag = 0;

void cv_test()
{
	QuickDemo qd;
	cv::Mat img1 = cv::imread("E:/图片/ME/mmexport1597321139729.jpg");
	cv::Mat img2 = cv::imread("E:/图片/ME/mmexport1597321161740.jpg");
	img1 = qd.get_face(img1);
	img2 = qd.get_face(img2);
	cv::imshow("img1", img1);
	cv::imshow("img2", img2);
	cv::waitKey(0);
	/*resize(img, img, Size(100, 60), 0, 0, INTER_LINEAR);
	imshow("img", img);
	cout << "L= (numpy)" << endl << format(img, Formatter::FMT_NUMPY) << endl << endl;
	cout << img.data << endl;*/
	cv::Mat feature1 = qd.get_feature(img1);
	cv::Mat feature2 = qd.get_feature(img2);

	qd.check_person(feature1, feature2);
	qd.face_match();
}

void cv_mysql_test()
{
	ConnectionPool* pool = ConnectionPool::getConnectPool();
	shared_ptr<MysqlConn> conn = pool->getConnection();

	QuickDemo qd;

	// 编码设置为gbk
	string sql = "set names gbk;";
	conn->update(sql);

	sql = "insert into person_img (name,face_path) values('李佳函','E:/图片/ME/FEB7B36BD1C5A98391F4CBD1CF2A6F86.jpg');";
	if (!conn->update(sql))
		return;

	cv::Mat test_face = cv::imread("E:/图片/ME/0199D3A18002AB74FF83E7D8AF111861.jpg"); // 测试图片
	test_face = qd.get_face(test_face);	// 获取人脸
	cv::Mat feature1 = qd.get_feature(test_face);	// 获取到特征

	sql = "select * from person_img;";
	conn->query(sql);
	string s;
	while (conn->next())
	{
		cout << conn->Value(0) << "," << conn->Value(1) << "," << conn->Value(2) << endl;
		s = conn->Value(2);
		cv::Mat img_sql = cv::imread(s);
		img_sql = qd.get_face(img_sql);
		cv::Mat feature2 = qd.get_feature(img_sql);	// 获取到特征
		if (qd.check_person(feature1, feature2))
		{
			cv::imshow("test", test_face);
			cv::waitKey(0);
			break;
		}

	}
}

bool video_test(cv::Mat frame)
{
	ConnectionPool* pool = ConnectionPool::getConnectPool();
	shared_ptr<MysqlConn> conn = pool->getConnection();
	

	// 编码设置为gbk
	string sql = "set names gbk;";
	conn->update(sql);

	frame = qd.get_face(frame);	// 获取人脸
	cv::Mat feature1 = qd.get_feature(frame);	// 获取到特征

	sql = "select * from person_img;";
	conn->query(sql);
	string s;
	while (conn->next())
	{
		cout << conn->Value(0) << "," << conn->Value(1) << "," << conn->Value(2) << endl;
		s = conn->Value(2);
		cv::Mat img_sql = cv::imread(s);
		img_sql = qd.get_face(img_sql);
		cv::Mat feature2 = qd.get_feature(img_sql);	// 获取到特征
		if (qd.check_person(feature1, feature2))
		{
			check_flag = 1;	// 标志着结束
			cout << "线程结束！" << endl;
			cv::imshow("test", frame);
			cv::waitKey(0);
			mutexQ.unlock();

			break;
		}
	}
	mutexQ.unlock();
}


int main(int argc, char** argv)
{
	
	cv::VideoCapture capture(0);
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
			thread t1(video_test,frame);
			t1.detach();
		}

		if (cv::waitKey(20) == 27)
			break;
	}
	capture.release();
	cout << "欢迎回家！" << endl;
	return 0;
}
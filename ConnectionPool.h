#pragma once
#include<queue>
#include<string>
#include<mutex>
#include<thread>
#include<condition_variable>
#include"MysqlConn.h"
using namespace std;
class ConnectionPool
{
public:
	static ConnectionPool* getConnectPool();	// ��̬�������õ�Ψһ�ĵ�������
	ConnectionPool(const ConnectionPool& obj) = delete;	// ɾ����������
	ConnectionPool& operator=(const ConnectionPool& obj) = delete;	// ɾ�����Ʋ���������
	shared_ptr<MysqlConn> getConnection();	// ��������ָ�룬��ȡ���ӣ�Ϊ������
	~ConnectionPool();
private:
	ConnectionPool(string user, string passwd, string dbName, string ip, unsigned short port);
	void produceConnection();	// �������ݿ����� �߳�
	void recycleConnection();	// �������� �߳�
	void addConnection();		// �������

	string m_ip;
	string m_user;
	string m_passwd;
	string m_dbName;
	unsigned short m_port;
	int m_minSize=5, m_maxSize=100;	// ����������
	int m_timeout = 1000;	// ��ʱʱ��
	int m_maxIdleTime = 5000;	// ������ʱ��

	queue<MysqlConn*> m_connectionQ;		// ���ӳض���
	mutex m_mutexQ;	// ����������������
	condition_variable m_cond;	// �����������˴������ߺ�������ʹ��ͬһ����������
};


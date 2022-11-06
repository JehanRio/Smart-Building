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
	static ConnectionPool* getConnectPool();	// 静态方法：得到唯一的单例对象
	ConnectionPool(const ConnectionPool& obj) = delete;	// 删除拷贝构造
	ConnectionPool& operator=(const ConnectionPool& obj) = delete;	// 删除复制操作符重载
	shared_ptr<MysqlConn> getConnection();	// 共享智能指针，获取连接，为消费者
	~ConnectionPool();
private:
	ConnectionPool(string user, string passwd, string dbName, string ip, unsigned short port);
	void produceConnection();	// 生产数据库连接 线程
	void recycleConnection();	// 销毁连接 线程
	void addConnection();		// 添加连接

	string m_ip;
	string m_user;
	string m_passwd;
	string m_dbName;
	unsigned short m_port;
	int m_minSize=5, m_maxSize=100;	// 连接上下限
	int m_timeout = 1000;	// 超时时长
	int m_maxIdleTime = 5000;	// 最大空闲时常

	queue<MysqlConn*> m_connectionQ;		// 连接池队列
	mutex m_mutexQ;	// 互斥锁，保护队列
	condition_variable m_cond;	// 条件变量：此处生产者和消费者使用同一个条件变量
};

